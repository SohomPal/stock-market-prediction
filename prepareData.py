import pandas as pd
from pathlib import Path
import os

# =============================
# Paths
# =============================
BASE_DIR = Path(os.environ.get("PSCRATCH")) / "StockPrediction"
INPUT_ROOT = BASE_DIR / "Data" / "stock_market_data"
OUTPUT_DIR = BASE_DIR / "Datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "stocks_all_ohlcv.parquet"

# Exchanges to load
EXCHANGES = ["forbes2000", "nasdaq", "nyse", "sp500"]

# Required columns (case-insensitive)
REQUIRED_COLUMNS = {
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}

# Optional but highly recommended
OPTIONAL_COLUMNS = {
    "adj close": "adj_close",
    "adjusted close": "adj_close",
    "adj_close": "adj_close",
}

MIN_ROWS_PER_TICKER = 100

all_data = []
seen_tickers = set()

total_files = 0
skipped_files = 0
dropped_small_tickers = 0

print("🚀 Starting OHLCV aggregation (unique tickers, min 100 rows)...")

# =============================
# Load CSVs
# =============================
for exchange in EXCHANGES:
    csv_dir = INPUT_ROOT / exchange / "csv"
    if not csv_dir.exists():
        print(f"⚠️ Missing directory: {csv_dir}")
        continue

    for csv_file in csv_dir.glob("*.csv"):
        total_files += 1
        ticker = csv_file.stem.upper()

        # Enforce UNIQUE ticker rule
        if ticker in seen_tickers:
            skipped_files += 1
            continue

        try:
            df = pd.read_csv(csv_file)

            # Normalize column names
            df.columns = [c.lower().strip() for c in df.columns]

            # Check required columns
            if not REQUIRED_COLUMNS.keys() <= set(df.columns):
                skipped_files += 1
                continue

            # Rename required columns
            df = df.rename(columns=REQUIRED_COLUMNS)

            # Handle adjusted close
            adj_col = None
            for k, v in OPTIONAL_COLUMNS.items():
                if k in df.columns:
                    adj_col = k
                    df = df.rename(columns={k: v})
                    break

            if adj_col is None:
                df["adj_close"] = df["close"]

            # Parse date
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            # Convert numeric columns
            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop invalid rows
            df = df.dropna(
                subset=["date", "open", "high", "low", "close", "adj_close", "volume"]
            )

            # Enforce MIN ROW constraint
            if len(df) < MIN_ROWS_PER_TICKER:
                dropped_small_tickers += 1
                continue

            # Sort by time
            df = df.sort_values("date")

            # Add metadata
            df["ticker"] = ticker
            df["exchange"] = exchange

            all_data.append(df)
            seen_tickers.add(ticker)

        except Exception as e:
            skipped_files += 1
            print(f"❌ Error reading {csv_file.name}: {e}")

# =============================
# Combine + Save
# =============================
if not all_data:
    raise RuntimeError("No valid OHLCV data found.")

combined_df = pd.concat(all_data, ignore_index=True)

# Final dedup safety
combined_df = combined_df.drop_duplicates(
    subset=["ticker", "date"], keep="last"
)

# Enforce dtypes
combined_df = combined_df.astype({
    "open": "float32",
    "high": "float32",
    "low": "float32",
    "close": "float32",
    "adj_close": "float32",
    "volume": "float32",
})

# Save as Parquet
combined_df.to_parquet(
    OUTPUT_FILE,
    engine="pyarrow",
    compression="snappy",
)

# =============================
# Stats
# =============================
print("\n✅ Dataset creation complete")
print(f"CSV files scanned: {total_files}")
print(f"Files skipped (duplicate/invalid): {skipped_files}")
print(f"Tickers dropped (< {MIN_ROWS_PER_TICKER} rows): {dropped_small_tickers}")
print(f"Unique tickers kept: {combined_df['ticker'].nunique():,}")
print(f"Total rows: {len(combined_df):,}")
print(f"Date range: {combined_df['date'].min()} → {combined_df['date'].max()}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nSample:")
print(combined_df.head())
