import os
import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# =========================================================
# CONFIG (replaces YAML)
# =========================================================

CONFIG = {
    "raw_dir": "data/raw",
    "processed_path": "data/processed/m5_processed.csv",
    "store_ids": ["CA_1"],
    "n_items": 50
}

SEQ_LEN = 28
HORIZON = 7

FEATURE_COLS = [
    "sales", "sell_price",
    "wday", "month",
    "is_event", "is_snap",
    "lag_7", "lag_28", "rolling_7"
]

TARGET_COL = "sales"


# =========================================================
# DATA PIPELINE
# =========================================================

def run_data_pipeline():
    """Full raw → processed pipeline"""

    raw_dir = CONFIG["raw_dir"]

    print("Loading raw data...")

    sales = pd.read_csv(
        os.path.join(raw_dir, "sales_train_evaluation.csv")
    )

    calendar = pd.read_csv(
        os.path.join(raw_dir, "calendar.csv")
    )

    prices = pd.read_csv(
        os.path.join(raw_dir, "sell_prices.csv")
    )

    # Filter stores
    sales = sales[
        sales["store_id"].isin(CONFIG["store_ids"])
    ]

    if CONFIG["n_items"]:
        sales = (
            sales.groupby("store_id")
            .head(CONFIG["n_items"])
        )

    # Melt wide → long
    id_cols = [
        "id", "item_id",
        "dept_id", "cat_id",
        "store_id", "state_id"
    ]

    day_cols = [
        c for c in sales.columns
        if c.startswith("d_")
    ]

    df = sales.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales"
    )

    # Merge calendar
    df = df.merge(
        calendar[
            ["d", "date", "wm_yr_wk",
             "wday", "month",
             "event_name_1",
             "snap_CA", "snap_TX", "snap_WI"]
        ],
        on="d"
    )

    # Merge prices
    df = df.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"]
    )

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["id", "date"], inplace=True)

    # Feature engineering
    df["sell_price"] = (
        df.groupby("item_id")["sell_price"]
        .transform(lambda x: x.fillna(x.median()))
    )

    df["is_event"] = df["event_name_1"].notna().astype(int)
    df["is_snap"] = df["snap_CA"].astype(int)

    g = df.groupby("id")["sales"]

    df["lag_7"] = g.shift(7)
    df["lag_28"] = g.shift(28)

    df["rolling_7"] = (
        g.transform(
            lambda x: x.shift(1).rolling(7).mean()
        )
    )

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(
        os.path.dirname(CONFIG["processed_path"]),
        exist_ok=True
    )

    df.to_csv(CONFIG["processed_path"], index=False)

    print("Processed data saved.")

    return df


# =========================================================
# DATASET
# =========================================================

class M5Dataset(Dataset):

    def __init__(
        self,
        df,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        scaler=None,
        fit_scaler=True
    ):

        self.seq_len = seq_len
        self.horizon = horizon

        self.scaler = scaler or StandardScaler()

        features = df[FEATURE_COLS].values.astype(np.float32)

        if fit_scaler:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)

        targets = df[TARGET_COL].values.astype(np.float32)

        self.X, self.y = self.build_windows(
            df,
            features,
            targets
        )

    def build_windows(self, df, features, targets):

        X, y = [], []

        start = 0

        for item_id, group in df.groupby("id"):

            group_len = len(group)

            end = start + group_len

            group_features = features[start:end]
            group_targets = targets[start:end]

            if group_len < self.seq_len + self.horizon:
                start = end
                continue

            for i in range(
                group_len - self.seq_len - self.horizon + 1
            ):

                X.append(
                    group_features[
                        i : i + self.seq_len
                    ]
                )

                y.append(
                    group_targets[
                        i + self.seq_len :
                        i + self.seq_len + self.horizon
                    ]
                )

            start = end

        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.y[idx])
        )


# =========================================================
# DATALOADER FACTORY
# =========================================================

def get_dataloaders(batch_size=64):

    df = pd.read_csv(
        CONFIG["processed_path"],
        parse_dates=["date"]
    )

    df = df.sort_values(["id", "date"])

    # ---------- GLOBAL SPLIT (correct) ----------

    n = len(df)

    train_end = int(n * 0.8)
    val_end   = int(n * 0.9)

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    # ---------- DATASETS ----------

    train_ds = M5Dataset(train)

    val_ds = M5Dataset(
        val,
        scaler=train_ds.scaler,
        fit_scaler=False
    )

    test_ds = M5Dataset(
        test,
        scaler=train_ds.scaler,
        fit_scaler=False
    )

    # ---------- DEBUG PRINTS ----------

    print("Train size:", len(train))
    print("Val size:", len(val))
    print("Test size:", len(test))

    print("Train windows:", len(train_ds))
    print("Val windows:", len(val_ds))
    print("Test windows:", len(test_ds))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
        train_ds.scaler
    )