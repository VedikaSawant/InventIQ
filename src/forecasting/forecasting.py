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

SEQ_LEN = 14
HORIZON = 7
N_FEATURES = 14


# =========================================================
# MODEL
# =========================================================

class DemandLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=N_FEATURES,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(0.3)

        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, HORIZON)
        )

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)
        out = self.dropout(hidden[-1])
        return self.head(out)


class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.log(torch.cosh(pred - target + 1e-8)))

# =========================================================
# TRAINING SYSTEM
# =========================================================

class DemandForecastingSystem:

    def __init__(
        self,
        lr=5e-4,
        weight_decay=1e-5,
        device="cpu",
        checkpoint_path="outputs/models/best_model.pt"
    ):

        self.device = device

        self.model = DemandLSTM().to(device)

        self.target_scaler = None

        self.criterion = LogCoshLoss()

        self.optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=4
        )

        self.checkpoint_path = checkpoint_path

        self.best_val_loss = float("inf")

    # =====================================================
    # TRAINING
    # =====================================================

    def train(
        self,
        train_loader,
        val_loader,
        epochs=25,
        early_stop=10
    ):

        patience = 0

        for epoch in range(epochs):

            train_loss = self._train_epoch(train_loader)

            val_loss, val_rmse, val_mae, val_r2 = self._validate(val_loader)

            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1} | "
                f"Train {train_loss:.4f} | "
                f"Val {val_loss:.4f} | "
                f"RMSE {val_rmse:.4f} | "
                f"MAE {val_mae:.4f} | "
                f"R2 {val_r2:.4f}"
            )

            if val_loss < self.best_val_loss:

                self.best_val_loss = val_loss

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

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate(self, loader):

        self.model.eval()

        total_loss = 0
        total_sq = 0
        total_abs = 0

        all_preds = []
        all_targets = []

        n = 0

        with torch.no_grad():

            for x, y in loader:

                x = x.to(self.device)
                y = y.to(self.device)

                preds = self.model(x)

                loss = self.criterion(preds, y)

                total_loss += loss.item()

                total_sq += ((preds - y) ** 2).sum().item()
                total_abs += (preds - y).abs().sum().item()

                n += y.numel()

                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

        mse = total_loss / len(loader)

        rmse = (total_sq / n) ** 0.5

        mae = total_abs / n

        preds_all = torch.cat(all_preds)
        targets_all = torch.cat(all_targets)

        # Convert tensors → numpy FIRST

        preds_np = preds_all.cpu().numpy()
        targets_np = targets_all.cpu().numpy()

        # Now inverse scale

        preds_real = np.expm1(self.target_scaler.inverse_transform(
            preds_np.reshape(-1, 1)
        ))

        targets_real = np.expm1(self.target_scaler.inverse_transform(
            targets_np.reshape(-1, 1)
        ))

        targets_real = targets_real.reshape(-1, HORIZON)
        preds_real = preds_real.reshape(-1, HORIZON)

        targets_real = targets_real.flatten()
        preds_real = preds_real.flatten()

        # R² Score
        ss_res = np.sum((targets_real - preds_real) ** 2)
        ss_tot = np.sum((targets_real - targets_real.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        return mse, rmse, mae, r2.item()

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