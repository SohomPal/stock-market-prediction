import pandas as pd
import numpy as np
from pathlib import Path

# Load feature dataset
data_path = Path("data") / "nasdaq_all.parquet"
output_path = Path("data") / "nasdaq_all_labeled.parquet"
df = pd.read_parquet(data_path)

# Sort before group-wise shifting
df = df.sort_values(["ticker", "Date"])

# Function to compute labels per stock
def compute_targets(group):
    group = group.copy()
    group["target_log_return_1d"] = np.log(group["Close"].shift(-1) / group["Close"])
    group["target_log_return_5d"] = np.log(group["Close"].shift(-5) / group["Close"])

    # Classify 1-day return
    threshold = 0.0025  # ≈0.25% change
    group["class_1d"] = pd.cut(
        group["target_log_return_1d"],
        bins=[-np.inf, -threshold, threshold, np.inf],
        labels=["negative", "neutral", "positive"]
    )
    return group

# Apply to each stock
df = df.groupby("ticker", group_keys=False).apply(compute_targets)

# Drop rows without target labels
df = df.dropna(subset=["target_log_return_1d", "target_log_return_5d", "class_1d"])

# Save and preview
df.to_parquet(output_path)

print("First 10 rows with labels:")
print(df.head(10))
print("\nLast 10 rows with labels:")
print(df.tail(10))
print(f"\nSaved labeled dataset to {output_path} with {len(df)} rows.")
