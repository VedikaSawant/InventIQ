# src/forecasting/forecasting.py
import os
import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


# =========================================================
# CONSTANTS (centralized — very important)
# =========================================================

SEQ_LEN = 28
HORIZON = 1
N_FEATURES = 17


# =========================================================
# MODEL
# =========================================================

class DemandLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=N_FEATURES,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(0.1)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, HORIZON)
        )

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)
        out = self.dropout(hidden[-1])
        return self.head(out)


class WeightedMSELoss(nn.Module):

    def __init__(self, nonzero_weight=2.0):
        super().__init__()
        self.nonzero_weight = nonzero_weight

    def forward(self, pred, target):

        weights = torch.ones_like(target)

        weights[torch.abs(target) > 0.05] = self.nonzero_weight

        loss = weights * (pred - target) ** 2

        return loss.mean()

# =========================================================
# TRAINING SYSTEM
# =========================================================

class DemandForecastingSystem:

    def __init__(
        self,
        lr=5e-5,
        weight_decay=1e-5,
        device="cpu",
        checkpoint_path="outputs/models/best_model.pt"
    ):

        self.device = device

        self.model = DemandLSTM().to(device)

        self.target_scaler = None

        self.criterion = nn.HuberLoss(delta=0.5)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # self.scheduler = ReduceLROnPlateau(
        #     self.optimizer,
        #     mode="min",
        #     factor=0.5,
        #     patience=4
        # )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-6
        )

        self.checkpoint_path = checkpoint_path

        self.best_val_rmse = float("inf")

    # =====================================================
    # TRAINING
    # =====================================================

    def train(
        self,
        train_loader,
        val_loader,
        epochs=60,
        early_stop=15
    ):

        patience = 0

        for epoch in range(epochs):

            train_loss = self._train_epoch(train_loader)

            (
                val_loss,
                val_rmse,
                val_nrmse,
                val_rmsse,
                val_mae,
                val_r2,
                naive_rmse
            ) = self._validate(val_loader)

            #self.scheduler.step(val_loss)
            self.scheduler.step()

            print(
                f"Epoch {epoch+1} | "
                f"Train {train_loss:.4f} | "
                f"Val {val_loss:.4f} | "
                f"RMSE {val_rmse:.4f} | "
                f"NRMSE {val_nrmse:.4f} | "
                f"RMSSE {val_rmsse:.4f} | "
                f"NaiveRMSE {naive_rmse:.4f} | "
                f"MAE {val_mae:.4f} | "
                f"R2 {val_r2:.4f}"
            )

            # In train(), replace the save condition:
            if val_rmse < self.best_val_rmse:  # track RMSE, not loss
                self.best_val_rmse = val_rmse
                self.save()
                patience = 0
            else:
                patience += 1

            if patience >= early_stop:
                print("Early stopping triggered")
                break

    # =====================================================
    # INTERNAL TRAIN STEP
    # =====================================================

    def _train_epoch(self, loader):

        self.model.train()

        total_loss = 0

        for x, y in loader:

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            preds = self.model(x)

            loss = self.criterion(preds, y)

            loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate(self, loader):
        self.model.eval()

        total_loss = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():

            for x, y in loader:

                x = x.to(self.device)
                y = y.to(self.device)

                preds = self.model(x)

                loss = self.criterion(preds, y)

                total_loss += loss.item()

                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

        # =====================================================
        # VALIDATION LOSS
        # =====================================================

        val_loss = total_loss / len(loader)

        # =====================================================
        # CONCATENATE ALL PREDICTIONS
        # =====================================================

        preds_all = torch.cat(all_preds)
        targets_all = torch.cat(all_targets)

        preds_np = preds_all.numpy()
        targets_np = targets_all.numpy()

        # =====================================================
        # INVERSE TRANSFORM TO REAL SALES SPACE
        # =====================================================

        preds_real = np.expm1(
            self.target_scaler.inverse_transform(
                preds_np.reshape(-1, 1)
            )
        )

        targets_real = np.expm1(
            self.target_scaler.inverse_transform(
                targets_np.reshape(-1, 1)
            )
        )

        targets_real = targets_real.flatten()
        preds_real = preds_real.flatten()

        # =====================================================
        # NAIVE BASELINE (last 28-step mean)
        # =====================================================

        naive_preds = []

        for i in range(28, len(targets_real)):
            naive_preds.append(
                targets_real[i-28:i].mean()
            )

        naive_targets = targets_real[28:]

        naive_rmse = np.sqrt(
            np.mean(
                (naive_targets - np.array(naive_preds)) ** 2
            )
        )

        # =====================================================
        # REAL-SCALE METRICS
        # =====================================================

        rmse = np.sqrt(
            np.mean((targets_real - preds_real) ** 2)
        )

        # =====================================================
        # RMSSE
        # =====================================================

        naive_diff = np.diff(targets_real)

        scale = np.mean(naive_diff ** 2)

        rmsse = np.sqrt(
            np.mean((targets_real - preds_real) ** 2)
            / (scale + 1e-8)
        )

        # =====================================================
        # NORMALIZED RMSE
        # =====================================================

        nrmse = rmse / (targets_real.mean() + 1e-8)

        mae = np.mean(
            np.abs(targets_real - preds_real)
        )

        # =====================================================
        # R² SCORE
        # =====================================================

        ss_res = np.sum((targets_real - preds_real) ** 2)

        ss_tot = np.sum(
            (targets_real - targets_real.mean()) ** 2
        )

        r2 = 1 - ss_res / ss_tot

        return (
            val_loss,
            rmse,
            nrmse,
            rmsse,
            mae,
            r2.item(),
            naive_rmse
        )

    # =====================================================
    # INFERENCE
    # =====================================================

    def forecast(self, window):

        self.model.eval()

        with torch.no_grad():

            if window.dim() == 2:
                window = window.unsqueeze(0)

            window = window.to(self.device)

            output = self.model(window)

        return output.squeeze(0)

    # =====================================================
    # CHECKPOINT
    # =====================================================

    def save(self):

        # Create directory if missing
        os.makedirs(
            os.path.dirname(self.checkpoint_path),
            exist_ok=True
        )

        torch.save(
            self.model.state_dict(),
            self.checkpoint_path
        )

    def load(self):

        self.model.load_state_dict(
            torch.load(
                self.checkpoint_path,
                map_location=self.device
            )
        )

    # =====================================================
    # RL HANDOFF
    # =====================================================

    def freeze_for_rl(self):

        self.load()

        # Freeze weights properly
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        return self.model