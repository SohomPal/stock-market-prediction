import pandas as pd
from pathlib import Path

# Path to feature-rich dataset
data_path = Path("data") / "nasdaq_all.parquet"

# Load dataset
df = pd.read_parquet(data_path)

# Print shape and columns
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
print("Column Names:")
print(df.columns.tolist())

# Show first and last few rows
print("\nFirst 10 rows:")
print(df.head(10))

print("\nLast 10 rows:")
print(df.tail(10))
