#!/usr/bin/env python
import pandas as pd
import numpy as np
from pathlib import Path
import os

# -----------------------------
# Config
# -----------------------------
MIN_ROWS_PER_TICKER = 120
MIN_DATE = pd.Timestamp("1990-01-01")
CLIP_LOG_RETURN = 0.20

DATASET_DIR = Path(os.environ["PSCRATCH"]) / "StockPrediction" / "Datasets"
INPUT_PATH = DATASET_DIR / "stocks_all_features.parquet"
OUTPUT_PATH = DATASET_DIR / "stocks_all_labeled_1d.parquet"

# -----------------------------
# Load dataset
# -----------------------------
print("📂 Loading feature dataset...")
df = pd.read_parquet(INPUT_PATH)

initial_rows = len(df)
initial_tickers = df["ticker"].nunique()

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# -----------------------------
# Date sanity
# -----------------------------
df = df[df["date"] >= MIN_DATE].copy()
rows_dropped_by_date = initial_rows - len(df)

# Sort correctly
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

# -----------------------------
# Pre-label min rows per ticker
# -----------------------------
counts = df["ticker"].value_counts()
valid_tickers = counts[counts >= MIN_ROWS_PER_TICKER].index
df = df[df["ticker"].isin(valid_tickers)].copy()

print(f"✅ Retained {len(valid_tickers):,} tickers after pre-label min-row filter")

# -----------------------------
# HARD PRICE SANITY CHECK
# -----------------------------
bad_price_tickers = df.loc[df["close"] <= 0, "ticker"].unique()

if len(bad_price_tickers) > 0:
    print(
        f"🚨 Dropping {len(bad_price_tickers):,} tickers "
        "with non-positive close prices"
    )

df = df[~df["ticker"].isin(bad_price_tickers)].copy()

# -----------------------------
# Vectorized 1D log return
# -----------------------------
df["target_log_return_1d"] = (
    np.log(df["close"].shift(-1)) -
    np.log(df["close"])
)

# Invalidate last row per ticker
df.loc[
    df["ticker"] != df["ticker"].shift(-1),
    "target_log_return_1d"
] = np.nan

# -----------------------------
# Identify bad tickers (ANY NaN / inf beyond last row)
# -----------------------------
# Identify unexpected invalid log returns (ignore last row per ticker)
is_last_row = df["ticker"] != df["ticker"].shift(-1)

bad_mask = (
    ~np.isfinite(df["target_log_return_1d"]) &
    ~is_last_row
)

bad_tickers = df.loc[bad_mask, "ticker"].unique()


if len(bad_tickers) > 0:
    print(f"🚨 Dropping {len(bad_tickers):,} tickers with invalid log returns")

df = df[~df["ticker"].isin(bad_tickers)].copy()

# Drop expected NaNs (last rows)
df = df.dropna(subset=["target_log_return_1d"]).reset_index(drop=True)

# -----------------------------
# Clip extreme returns
# -----------------------------
df["target_log_return_1d"] = df["target_log_return_1d"].clip(
    -CLIP_LOG_RETURN,
    CLIP_LOG_RETURN
)

# -----------------------------
# Post-label min rows per ticker
# -----------------------------
counts = df["ticker"].value_counts()
final_tickers = counts[counts >= MIN_ROWS_PER_TICKER].index
dropped_post = df["ticker"].nunique() - len(final_tickers)

df = df[df["ticker"].isin(final_tickers)].reset_index(drop=True)

# -----------------------------
# FINAL ASSERTIONS
# -----------------------------
assert df["target_log_return_1d"].isna().sum() == 0
assert np.isfinite(df["target_log_return_1d"]).all()
assert (df["close"] > 0).all()

# -----------------------------
# Save
# -----------------------------
df.to_parquet(
    OUTPUT_PATH,
    engine="pyarrow",
    compression="snappy"
)

# -----------------------------
# Report
# -----------------------------
print("\n📊 Labeled dataset summary")
print("-" * 40)
print(f"Initial rows               : {initial_rows:,}")
print(f"Rows dropped (date filter) : {rows_dropped_by_date:,}")
print(f"Final rows                 : {len(df):,}")

print(f"Initial tickers            : {initial_tickers:,}")
print(f"Tickers dropped (bad data) : {len(bad_tickers):,}")
print(f"Tickers dropped post-label : {dropped_post:,}")
print(f"Final tickers              : {df['ticker'].nunique():,}")

print(f"Date range                 : {df['date'].min()} → {df['date'].max()}")
print(
    f"Return stats               : "
    f"mean={df['target_log_return_1d'].mean():.5f}, "
    f"std={df['target_log_return_1d'].std():.5f}"
)

print(f"\n💾 Saved labeled dataset to:\n{OUTPUT_PATH}")
