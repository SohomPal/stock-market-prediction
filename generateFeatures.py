import pandas as pd
import numpy as np
from pathlib import Path
import os
import ta

# =============================
# Paths
# =============================
BASE_DIR = Path(os.environ["PSCRATCH"]) / "StockPrediction"
INPUT_PATH = BASE_DIR / "Datasets" / "stocks_all_ohlcv.parquet"
OUTPUT_PATH = BASE_DIR / "Datasets" / "stocks_all_features.parquet"

# =============================
# Config
# =============================
MIN_HISTORY = 60
MIN_ROWS_PER_TICKER = 100
MIN_DATE = pd.Timestamp("1990-01-01")
EPS = 1e-8

print("📥 Loading OHLCV dataset...")
df = pd.read_parquet(INPUT_PATH)

initial_rows = len(df)
initial_tickers = df["ticker"].nunique()

# =============================
# Date sanity filter
# =============================
print("🗓️ Dropping invalid / pre-1990 dates...")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df[df["date"] >= MIN_DATE].copy()

rows_dropped_by_date = initial_rows - len(df)

# =============================
# Sorting + dtypes
# =============================
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

FLOAT_COLS = ["open", "high", "low", "close", "adj_close", "volume"]
df[FLOAT_COLS] = df[FLOAT_COLS].astype("float32")

# =============================
# Feature generation per ticker
# =============================
def add_technical_indicators(group: pd.DataFrame) -> pd.DataFrame:
    if len(group) < MIN_HISTORY:
        return pd.DataFrame()

    g = group.copy()

    close = g["close"]
    high = g["high"]
    low = g["low"]
    volume = g["volume"]

    # -------- Trend --------
    g["sma_20"] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    g["sma_50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    g["ema_12"] = ta.trend.EMAIndicator(close, window=12).ema_indicator()
    g["ema_26"] = ta.trend.EMAIndicator(close, window=26).ema_indicator()

    macd = ta.trend.MACD(close)
    g["macd"] = macd.macd()
    g["macd_signal"] = macd.macd_signal()

    # -------- Momentum --------
    g["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    stoch = ta.momentum.StochasticOscillator(high, low, close)
    g["stoch_k"] = stoch.stoch()
    g["stoch_d"] = stoch.stoch_signal()

    g["roc"] = ta.momentum.ROCIndicator(close, window=10).roc()

    # -------- Volatility --------
    atr = ta.volatility.AverageTrueRange(high, low, close)
    g["atr_14"] = atr.average_true_range()

    boll = ta.volatility.BollingerBands(close)
    g["bollinger_h"] = boll.bollinger_hband()
    g["bollinger_l"] = boll.bollinger_lband()
    g["bollinger_w"] = boll.bollinger_wband()

    # -------- Volume --------
    g["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    g["mfi"] = ta.volume.MFIIndicator(high, low, close, volume).money_flow_index()
    g["vpt"] = ta.volume.VolumePriceTrendIndicator(close, volume).volume_price_trend()

    # -------- Returns --------
    g["return_1d"] = close.pct_change()
    g["log_return"] = np.log((close + EPS) / (close.shift(1) + EPS))

    g["volatility_5"] = g["log_return"].rolling(5).std()
    g["volatility_20"] = g["log_return"].rolling(20).std()

    # -------- Microstructure --------
    g["price_range_pct"] = (high - low) / (close + EPS)
    g["volume_delta"] = volume.diff()
    g["volume_z"] = (volume - volume.rolling(20).mean()) / (
        volume.rolling(20).std() + EPS
    )

    return g

print("⚙️ Computing technical indicators...")
df = df.groupby("ticker", group_keys=False).apply(add_technical_indicators)

# =============================
# Final cleanup
# =============================
print("🧹 Cleaning invalid values...")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)

rows_after_features = len(df)

# =============================
# Enforce MIN ROWS per ticker (POST-CLEAN)
# =============================
ticker_counts = df["ticker"].value_counts()
valid_tickers = ticker_counts[ticker_counts >= MIN_ROWS_PER_TICKER].index

dropped_tickers_post = df["ticker"].nunique() - len(valid_tickers)

df = df[df["ticker"].isin(valid_tickers)].reset_index(drop=True)

# Downcast to save memory
for col in df.columns:
    if df[col].dtype == "float64":
        df[col] = df[col].astype("float32")

# =============================
# Save
# =============================
df.to_parquet(
    OUTPUT_PATH,
    engine="pyarrow",
    compression="snappy"
)

# =============================
# Report
# =============================
print("\n✅ Feature dataset ready")
print(f"Initial rows: {initial_rows:,}")
print(f"Rows dropped (pre-1990 dates): {rows_dropped_by_date:,}")
print(f"Rows after feature cleaning: {rows_after_features:,}")
print(f"Final rows: {len(df):,}")

print(f"Initial tickers: {initial_tickers:,}")
print(f"Tickers dropped post-clean (< {MIN_ROWS_PER_TICKER} rows): {dropped_tickers_post:,}")
print(f"Final tickers: {df['ticker'].nunique():,}")

print(f"Date range: {df['date'].min()} → {df['date'].max()}")
print(f"Saved to: {OUTPUT_PATH}")

print("\nSample:")
print(df.head())
