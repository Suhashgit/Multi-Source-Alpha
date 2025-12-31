import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRICES_DIR = REPO_ROOT / "data" / "prices"

# Prefer a "wide panel" volume file if you have it; else fall back to downloading.
VOLUME_PATH = PRICES_DIR / "sp500_volume.csv"
ADJ_CLOSE_PATH = PRICES_DIR / "sp500_adj_close.csv"

OUT_DIR = REPO_ROOT / "data" / "volume" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RAW = OUT_DIR / "volume_shock_raw.parquet"
OUT_Z = OUT_DIR / "volume_shock_z.parquet"

def load_volume_panel() -> pd.DataFrame:
    if not VOLUME_PATH.exists():
        raise FileNotFoundError(f"Volume data not found at {VOLUME_PATH}. Please run 'scripts/get_volume_yfinance.py' to download it.")
    vol = pd.read_csv(VOLUME_PATH, index_col = 0, parse_dates=True)
    vol = vol.sort_index()
    return vol

def winsorize_df(df: pd.DataFrame, lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    lo = df.quantile(lower_q, axis=1)
    hi = df.quantile(upper_q, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)

def rolling_zscore(df: pd.DataFrame, window = 60, min_periods = 40) -> pd.DataFrame:
    mu = df.rolling(window=window, min_periods=min_periods).mean()
    sd = df.rolling(window=window, min_periods=min_periods).std(ddof=0).replace(0,np.nan)
    return (df-mu)/sd


def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    mu = panel.mean(axis=1)
    sd = panel.std(axis=1).replace(0, np.nan)
    return panel.sub(mu, axis=0).div(sd, axis=0)

def compute_volume_shock(volume: pd.DataFrame,
                         window = 60,
                         min_periods = 40,
                         winsorize = True) -> pd.DataFrame:
    v = volume.copy()
    v = v.apply(pd.to_numeric, errors='coerce')
    v = v.where(v > 0, np.nan)
    logv = np.log(v)
    if winsorize:
        logv = winsorize_df(logv, 0.01,0.99)
    shock = rolling_zscore(logv, window=window, min_periods=min_periods)
    return shock

def main():
    vol = load_volume_panel()
    shock = compute_volume_shock(vol, window=60, min_periods=40, winsorize=True)

    # Save raw rolling z-score signal
    shock.to_parquet(OUT_RAW)

    # Save cross-sectional standardized version (useful if you want to rank stocks each day)
    shock_cs = cross_sectional_zscore(shock)
    shock_cs.to_parquet(OUT_Z)

    print(f"Saved volume shock to:\n  {OUT_RAW}\n  {OUT_Z}")
    print("Shape:", shock.shape)
    print("Date range:", shock.index.min(), "to", shock.index.max())


if __name__ == "__main__":
    main()