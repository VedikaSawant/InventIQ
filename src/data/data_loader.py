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

    "store_ids": [
        "CA_1",
        "CA_2",
        "TX_1",
        "WI_3"
    ],

    "n_items": None,

    "selected_items": [
        "FOODS_3_586",
        "FOODS_3_090",
        "FOODS_3_555",
        "FOODS_3_252",
        "FOODS_3_587",
        "FOODS_3_714",
        "FOODS_3_694",
        "FOODS_2_360",
        "FOODS_3_150",
        "FOODS_3_080"
    ]
}

SEQ_LEN = 14
HORIZON = 7

FEATURE_COLS = [
    "sell_price",
    "wday", "month",
    "is_event", "is_snap",
    "lag_1", "lag_7", "lag_14", "lag_28",      # added lag_14
    "rolling_7", "rolling_14", "rolling_28",    # added rolling_14
    "rolling_7_std",                             # added volatility
    "item_idx"                                   # added item identity
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

    # Filter by store
    sales = sales[
        sales["store_id"].isin(CONFIG["store_ids"])
    ]

    # NEW — filter specific items
    if CONFIG.get("selected_items"):
        sales = sales[
            sales["item_id"].isin(CONFIG["selected_items"])
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

    # FIX 1: Left join so items with missing prices keep all their days
    # Inner join was dropping ~15k rows where price history didn't exist yet
    df = df.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["id", "date"], inplace=True)

    # Feature engineering
    # sell_price fill must come AFTER the left join so NaNs are filled
    df["sell_price"] = (
        df.groupby("item_id")["sell_price"]
        .transform(lambda x: x.fillna(x.median()))
    )

    # After groupby id, add label-encoded item ID as a feature
    df["item_idx"] = df.groupby("id").ngroup().astype(np.float32)
    df["item_idx"] = df["item_idx"] / df["item_idx"].max()  # normalize 0-1

    df["is_event"] = df["event_name_1"].notna().astype(int)
    df["is_snap"] = df["snap_CA"].astype(int)

    g = df.groupby("id")["sales"]

    df["lag_1"]  = g.shift(1)
    df["lag_7"]  = g.shift(7)
    df["lag_28"] = g.shift(28)

    df["lag_14"]       = g.shift(14)
    df["rolling_14"]   = g.transform(lambda x: x.shift(1).rolling(14).mean())
    df["rolling_7_std"] = g.transform(lambda x: x.shift(1).rolling(7).std())

    df["rolling_7"] = (
        g.transform(
            lambda x: x.shift(1).rolling(7).mean()
        )
    )

    df["rolling_28"] = (
        g.transform(
            lambda x: x.shift(1).rolling(28).mean()
        )
    )

    # FIX 2: Drop NaN rows per-item instead of globally.
    # Global dropna was nuking entire items because lag_28 creates 28 NaN
    # rows at the START of each item — those early rows should be dropped,
    # but the rest of the item's history is perfectly valid and must be kept.
    feature_cols_with_target = FEATURE_COLS + [TARGET_COL]
    df = (
        df.groupby("id", group_keys=False)
        .apply(lambda g: g.dropna(subset=feature_cols_with_target))
    )

    df.reset_index(drop=True, inplace=True)

    print(f"After pipeline: {df.shape}")
    print(f"Days per item:\n{df.groupby('id').size().describe()}")

    os.makedirs(
        os.path.dirname(CONFIG["processed_path"]),
        exist_ok=True
    )

    # Keep only items with meaningful sales activity
    item_mean = df.groupby("id")["sales"].mean()
    active_items = item_mean[item_mean >= 0.5].index  # at least 1 sale/day average
    df = df[df["id"].isin(active_items)]

    print(f"Active items kept: {len(active_items)} / {item_mean.shape[0]}")
    print(f"After filtering: {df.shape}")

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
        target_scaler=None,
        fit_scaler=True
    ):

        self.seq_len = seq_len
        self.horizon = horizon

        # --- Feature scaler (no target column inside) ---
        self.scaler = scaler or StandardScaler()

        features = df[FEATURE_COLS].values.astype(np.float32)

        if fit_scaler:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)

        # --- Target scaler (separate, only for sales) ---
        self.target_scaler = target_scaler or StandardScaler()

        targets = np.log1p(df[TARGET_COL].values.astype(np.float32)).reshape(-1, 1)

        if fit_scaler:
            targets = self.target_scaler.fit_transform(targets)
        else:
            targets = self.target_scaler.transform(targets)

        targets = targets.flatten()

        self.X, self.y = self.build_windows(df, features, targets)

    def inverse_transform_targets(self, y_scaled):
        """Call this on predictions before computing MAPE/MAE for reporting."""
        y_scaled = np.array(y_scaled).reshape(-1, 1)
        return self.target_scaler.inverse_transform(y_scaled).flatten()

    def build_windows(self, df, features, targets):

        X, y = [], []

        df = df.reset_index(drop=True)

        for item_id, group in df.groupby("id", sort=True):

            idx = group.index.to_numpy()

            group_features = features[idx]
            group_targets  = targets[idx]
            group_len      = len(idx)

            if group_len < self.seq_len + self.horizon:
                print(f"  ⚠ Skipping {item_id}: {group_len} days < {self.seq_len + self.horizon} needed")
                continue

            for i in range(group_len - self.seq_len - self.horizon + 1):
                X.append(group_features[i : i + self.seq_len])
                y.append(group_targets[i + self.seq_len : i + self.seq_len + self.horizon])

        if len(X) == 0:
            raise ValueError(
                f"No windows built — all groups shorter than "
                f"seq_len({self.seq_len}) + horizon({self.horizon}) = {self.seq_len + self.horizon}. "
                f"Check your date split ratios or reduce SEQ_LEN."
            )

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

def get_dataloaders(batch_size=128):

    df = pd.read_csv(
        CONFIG["processed_path"],
        parse_dates=["date"]
    )

    df = df.sort_values(["id", "date"])

    # Split by date so every item is cut at the same calendar point
    dates = np.sort(df["date"].unique())

    n = len(dates)
    train_cutoff = dates[int(n * 0.7)]
    val_cutoff   = dates[int(n * 0.85)]

    train = df[df["date"] <  train_cutoff]
    val   = df[(df["date"] >= train_cutoff) & (df["date"] < val_cutoff)]
    test  = df[df["date"] >= val_cutoff]

    # ---------- DATASETS ----------

    train_ds = M5Dataset(train, fit_scaler=True)

    val_ds = M5Dataset(
        val,
        scaler=train_ds.scaler,
        target_scaler=train_ds.target_scaler,
        fit_scaler=False
    )

    test_ds = M5Dataset(
        test,
        scaler=train_ds.scaler,
        target_scaler=train_ds.target_scaler,
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
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, num_workers=4, pin_memory=True),
        DataLoader(test_ds,  batch_size=batch_size, num_workers=4, pin_memory=True),
        train_ds.scaler,
        train_ds.target_scaler
    )