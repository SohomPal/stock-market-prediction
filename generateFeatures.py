import pandas as pd
import numpy as np
from pathlib import Path
import ta

# Paths
data_dir = Path("data")
input_path = data_dir / "nasdaq_all_ohlcv.parquet"
output_path = data_dir / "nasdaq_all.parquet"

# Load base data
df = pd.read_parquet(input_path)

# Ensure date column exists and is datetime
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["ticker", "Date"])

# Function to compute indicators for each stock
def add_technical_indicators(group):
    if len(group) < 30:
        return pd.DataFrame()  # skip too-short histories
    group = group.copy()

    # Trend
    group["sma_20"] = ta.trend.SMAIndicator(group["Close"], window=20).sma_indicator()
    group["sma_50"] = ta.trend.SMAIndicator(group["Close"], window=50).sma_indicator()
    group["ema_12"] = ta.trend.EMAIndicator(group["Close"], window=12).ema_indicator()
    group["ema_26"] = ta.trend.EMAIndicator(group["Close"], window=26).ema_indicator()
    macd = ta.trend.MACD(group["Close"])
    group["macd"] = macd.macd()
    group["macd_signal"] = macd.macd_signal()

    # Momentum
    group["rsi_14"] = ta.momentum.RSIIndicator(group["Close"], window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(group["High"], group["Low"], group["Close"])
    group["stoch_k"] = stoch.stoch()
    group["stoch_d"] = stoch.stoch_signal()
    group["roc"] = ta.momentum.ROCIndicator(group["Close"]).roc()

    # Volatility
    group["atr_14"] = ta.volatility.AverageTrueRange(group["High"], group["Low"], group["Close"]).average_true_range()
    boll = ta.volatility.BollingerBands(group["Close"])
    group["bollinger_h"] = boll.bollinger_hband()
    group["bollinger_l"] = boll.bollinger_lband()

    # Volume
    group["obv"] = ta.volume.OnBalanceVolumeIndicator(group["Close"], group["Volume"]).on_balance_volume()
    group["mfi"] = ta.volume.MFIIndicator(group["High"], group["Low"], group["Close"], group["Volume"]).money_flow_index()
    group["vpt"] = ta.volume.VolumePriceTrendIndicator(group["Close"], group["Volume"]).volume_price_trend()

    # Custom features
    group["daily_return"] = group["Close"].pct_change(fill_method=None)
    group["log_return"] = np.log(group["Close"] / group["Close"].shift(1)).replace([np.inf, -np.inf], 0)
    group["volatility_5"] = group["daily_return"].rolling(5).std()
    group["volatility_20"] = group["daily_return"].rolling(20).std()
    group["price_range_pct"] = (group["High"] - group["Low"]) / group["Close"]
    group["volume_delta"] = group["Volume"].diff()

    return group

# Apply feature generation per stock
df = df.groupby("ticker", group_keys=False).apply(add_technical_indicators)

# Drop rows with any missing values caused by indicator lags
df = df.dropna()

# Save enriched dataset
df.to_parquet(output_path)

# Print preview
print("First 10 rows:")
print(df.head(10))
print("\nLast 10 rows:")
print(df.tail(10))
print(f"\nSaved feature-rich dataset to {output_path} with {len(df)} rows.")
