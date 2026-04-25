# src/forecasting/forecasting.py
import os
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


# =========================================================
# CONSTANTS (centralized — very important)
# =========================================================

SEQ_LEN = 28
HORIZON = 7
N_FEATURES = 9


# =========================================================
# MODEL
# =========================================================

class DemandLSTM(nn.Module):

    def __init__(
        self,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=N_FEATURES,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, HORIZON)
        )

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        context = hidden[-1]

        context = self.layer_norm(context)

        return self.head(context)

    def freeze(self):

        for p in self.parameters():
            p.requires_grad = False

        self.eval()


# =========================================================
# TRAINING SYSTEM
# =========================================================

class DemandForecastingSystem:

    def __init__(
        self,
        lr=1e-3,
        weight_decay=1e-4,
        device="cpu",
        checkpoint_path="outputs/models/best_model.pt"
    ):

        self.device = device

        self.model = DemandLSTM().to(device)

        self.criterion = nn.MSELoss()

        self.optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5
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
        epochs=50,
        early_stop=10
    ):

        patience = 0

        for epoch in range(epochs):

            train_loss = self._train_epoch(train_loader)

            val_loss, val_rmse = self._validate(val_loader)

            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1} | "
                f"Train {train_loss:.4f} | "
                f"Val {val_loss:.4f} | "
                f"RMSE {val_rmse:.4f}"
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
        n = 0

        with torch.no_grad():

            for x, y in loader:

                x = x.to(self.device)
                y = y.to(self.device)

                preds = self.model(x)

                loss = self.criterion(preds, y)

                total_loss += loss.item()

                total_sq += ((preds - y) ** 2).sum().item()

                n += y.numel()

        mse = total_loss / len(loader)

        rmse = (total_sq / n) ** 0.5

        return mse, rmse

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

        self.model.freeze()

        return self.model