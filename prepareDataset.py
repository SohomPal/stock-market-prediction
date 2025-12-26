import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATA_PATH = Path("/pscratch/sd/s/sp2160/StockPrediction/Datasets/stocks_all_labeled_1d.parquet")
OUT_DIR = Path("/pscratch/sd/s/sp2160/StockPrediction/LSTM_Data")

LOOKBACK = 30
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
MIN_ROWS_PER_TICKER = 60

OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD DATA
# =========================
print("📂 Loading labeled dataset...")
df = pd.read_parquet(DATA_PATH)

initial_rows = len(df)
initial_tickers = df["ticker"].nunique()

# =========================
# DATE COLUMN HANDLING
# =========================
if "Date" in df.columns:
    date_col = "Date"
elif "date" in df.columns:
    date_col = "date"
elif df.index.name in ("Date", "date"):
    df = df.reset_index()
    date_col = df.index.name
else:
    raise RuntimeError("❌ No date column found.")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.sort_values(["ticker", date_col]).reset_index(drop=True)

# =========================
# FILTER SHORT TICKERS
# =========================
counts = df["ticker"].value_counts()
valid_tickers = counts[counts >= MIN_ROWS_PER_TICKER].index
df = df[df["ticker"].isin(valid_tickers)].copy()

print(f"✅ Retained {len(valid_tickers):,} tickers after min-row filter")

# =========================
# NUMERIC FEATURE SELECTION
# =========================
NUMERIC_COLS = df.select_dtypes(include=["number"]).columns.tolist()
TARGET_COL = "target_log_return_1d"

FEATURE_COLS = [c for c in NUMERIC_COLS if c != TARGET_COL]

print(f"📊 Using {len(FEATURE_COLS)} numeric features")

# =========================
# HARD NaN / INF CHECK (PRE-NORMALIZATION)
# =========================
bad_mask = ~np.isfinite(df[FEATURE_COLS + [TARGET_COL]]).all(axis=1)
bad_tickers = df.loc[bad_mask, "ticker"].unique()

if len(bad_tickers) > 0:
    print(f"🚨 Dropping {len(bad_tickers):,} tickers with NaN/inf values (pre-normalization)")
    df = df[~df["ticker"].isin(bad_tickers)].copy()

# =========================
# ROBUST NORMALIZATION
# =========================
print("📐 Computing robust normalization stats...")
median = df[FEATURE_COLS].median()
iqr = df[FEATURE_COLS].quantile(0.75) - df[FEATURE_COLS].quantile(0.25)
iqr = iqr.replace(0, 1e-6)

df[FEATURE_COLS] = (df[FEATURE_COLS] - median) / iqr

# =========================
# HARD NaN / INF CHECK (POST-NORMALIZATION)
# =========================
bad_mask = ~np.isfinite(df[FEATURE_COLS + [TARGET_COL]]).all(axis=1)
bad_tickers_post = df.loc[bad_mask, "ticker"].unique()

if len(bad_tickers_post) > 0:
    print(f"🚨 Dropping {len(bad_tickers_post):,} tickers with NaN/inf values (post-normalization)")
    df = df[~df["ticker"].isin(bad_tickers_post)].copy()

# =========================
# BUILD LSTM WINDOWS
# =========================
print("🧠 Building LSTM windows...")

X_list, y_list = [], []

for _, group in tqdm(df.groupby("ticker"), desc="Tickers"):
    group = group.reset_index(drop=True)

    X_vals = group[FEATURE_COLS].values.astype(np.float32)
    y_vals = group[TARGET_COL].values.astype(np.float32)

    for i in range(LOOKBACK, len(group)):
        X_list.append(X_vals[i - LOOKBACK:i])
        y_list.append(y_vals[i])

X = np.stack(X_list)
y = np.array(y_list)

# =========================
# FINAL SAFETY ASSERTIONS
# =========================
assert np.isfinite(X).all(), "❌ NaN or inf in X"
assert np.isfinite(y).all(), "❌ NaN or inf in y"

print(f"✅ Total samples: {len(X):,}")

# =========================
# TIME SPLIT
# =========================
n = len(X)
train_end = int(n * TRAIN_SPLIT)
val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

splits = {
    "train": (X[:train_end], y[:train_end]),
    "val": (X[train_end:val_end], y[train_end:val_end]),
    "test": (X[val_end:], y[val_end:]),
}

# =========================
# SAVE
# =========================
print("💾 Saving .npy files...")
for split, (Xs, ys) in splits.items():
    np.save(OUT_DIR / f"{split}_X.npy", Xs)
    np.save(OUT_DIR / f"{split}_y.npy", ys)

# =========================
# REPORT
# =========================
print("\n🎉 LSTM dataset ready!")
print("-" * 40)
print(f"Initial rows     : {initial_rows:,}")
print(f"Final rows       : {len(df):,}")
print(f"Initial tickers  : {initial_tickers:,}")
print(f"Final tickers    : {df['ticker'].nunique():,}")
print(f"Total samples    : {len(X):,}")
print(f"Saved to         : {OUT_DIR}")
