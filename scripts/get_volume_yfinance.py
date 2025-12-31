import pandas as pd
import yfinance as yf
from pathlib import Path
import sys
import os
# Add the parent directory of 'multi_source_alpha' to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
REPO_ROOT = Path(__file__).resolve().parents[1]          
DATA_DIR = REPO_ROOT / "data"
PRICES_DIR = DATA_DIR / "prices"
PRICES_DIR.mkdir(parents=True, exist_ok=True)

TICKERS_PATH = DATA_DIR / "Universe" / "sp500_constituents.csv"

OUT_PATH = PRICES_DIR / "sp500_volume.csv"


def load_tickers():
    df = pd.read_csv(TICKERS_PATH, header=None)
    tickers = df.iloc[:, 0].astype(str).str.strip().tolist()
    return tickers


def main():
    tickers = load_tickers()

    df = yf.download(
        tickers=tickers,
        start="2000-01-01",
        auto_adjust=False,
        group_by="column",
        threads=True,
        progress=True,
    )

    # yfinance returns MultiIndex columns: (Field, Ticker)
    # We want the 'Volume' layer.
    if isinstance(df.columns, pd.MultiIndex):
        if "Volume" not in df.columns.get_level_values(0):
            raise KeyError(f"'Volume' not found in yfinance output. Top level fields: {df.columns.levels[0]}")
        vol = df["Volume"].copy()
    else:
        # single ticker case
        if "Volume" not in df.columns:
            raise KeyError(f"'Volume' not found in yfinance output columns: {df.columns}")
        vol = df[["Volume"]].copy()

    vol.index.name = "Date"
    vol = vol.sort_index()

    vol.to_csv(OUT_PATH, index=True)
    print(f"Saved volume panel to: {OUT_PATH}")
    print("Shape:", vol.shape)
    print("Date range:", vol.index.min(), "to", vol.index.max())


if __name__ == "__main__":
    main()