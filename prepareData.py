import pandas as pd
from pathlib import Path

# Set up paths
data_dir = Path("data")
stocks_dir = data_dir / "stocks"
etfs_dir = data_dir / "etfs"
meta_file = data_dir / "symbols_valid_meta.csv"
output_file = data_dir / "nasdaq_all_ohlcv.parquet"

# Load symbol metadata
meta_df = pd.read_csv(meta_file)
tickers_df = meta_df[["NASDAQ Symbol", "ETF"]].dropna()
tickers_df.columns = ["symbol", "is_etf"]

# Initialize list to collect all stock/ETF DataFrames
all_data = []

# Load each individual CSV and tag with ticker
for _, row in tickers_df.iterrows():
    ticker = row["symbol"]
    is_etf = row["is_etf"].strip().upper() == "Y"
    folder = etfs_dir if is_etf else stocks_dir
    file_path = folder / f"{ticker}.csv"

    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            df["ticker"] = ticker
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Combine and save
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_parquet(output_file)
    
    # Print first and last 10 rows
    print("First 10 rows:")
    print(combined_df.head(10))
    print("\nLast 10 rows:")
    print(combined_df.tail(10))
    print(f"\nSaved combined dataset to {output_file} with {len(combined_df)} rows.")
else:
    print("No CSV data found in provided folders.")
