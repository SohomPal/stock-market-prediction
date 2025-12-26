import pandas as pd
from pathlib import Path
import os

# Path to feature dataset
DATASET_PATH = (
    Path(os.environ["PSCRATCH"])
    / "StockPrediction"
    / "Datasets"
    / "stocks_all_features.parquet"
)

print(f"📂 Loading dataset from:\n{DATASET_PATH}\n")

df = pd.read_parquet(DATASET_PATH)

print("📊 Dataset summary")
print("-" * 40)
print(f"Total rows      : {len(df):,}")
print(f"Unique tickers  : {df['ticker'].nunique():,}")
print(f"Date range      : {df['date'].min()} → {df['date'].max()}")
print(f"Columns         : {len(df.columns)}")

# Check for NaNs
nan_cols = df.columns[df.isna().any()].tolist()
print("\n🧪 NaN check")
if nan_cols:
    print("⚠️ Columns with NaNs:")
    for c in nan_cols:
        print(f"  - {c}")
else:
    print("✅ No NaNs found")

# Show sample tickers
print("\n🔍 Sample tickers:")
print(df["ticker"].drop_duplicates().head(20).tolist())

# Optional: rows per ticker stats
rows_per_ticker = df.groupby("ticker").size()
print("\n📈 Rows per ticker")
print(f"Min rows : {rows_per_ticker.min():,}")
print(f"Median   : {int(rows_per_ticker.median()):,}")
print(f"Max rows : {rows_per_ticker.max():,}")
