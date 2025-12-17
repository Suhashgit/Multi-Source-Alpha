import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
# Add the parent directory of 'multi_source_alpha' to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from multi_source_alpha.data_providers.earnings_finnhub import fetch_earnings_history
from multi_source_alpha.signals.momentum import load_sp500_adj_close
from multi_source_alpha.signals.sentiment.earnings import (
    compute_eps_surprise,
    standardize_surprise_within_ticker,
    build_daily_decayed_sentiment,
)

RAW_DIR = Path("data/sentiment/raw")
PROCESSED_DIR = Path("data/sentiment/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 1) Load Kaggle as canonical
# -----------------------------
def load_kaggle_as_canonical(path: Path) -> pd.DataFrame:
    """
    Kaggle columns (from your screenshot):
      symbol, date, qtr, eps_est, eps, release_time

    Canonical output:
      symbol, date, epsActual, epsEstimate, source
    """
    df = pd.read_csv(path)

    # rename into canonical names
    df = df.rename(columns={
        "eps": "epsActual",
        "eps_est": "epsEstimate",
    })

    # clean + types
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["epsActual"] = pd.to_numeric(df["epsActual"].replace("NULL", np.nan), errors="coerce")
    df["epsEstimate"] = pd.to_numeric(df["epsEstimate"].replace("NULL", np.nan), errors="coerce")

    df["source"] = "kaggle"
    return df[["symbol", "date", "epsActual", "epsEstimate", "source"]]


# -----------------------------
# 2) Finnhub to canonical
# -----------------------------
def finnhub_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finnhub earningsCalendar usually includes:
      symbol, date, epsActual, epsEstimate, ...

    Canonical output:
      symbol, date, epsActual, epsEstimate, source
    """
    out = df.copy()

    if out.empty:
        return out

    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["epsActual"] = pd.to_numeric(out.get("epsActual"), errors="coerce")
    out["epsEstimate"] = pd.to_numeric(out.get("epsEstimate"), errors="coerce")

    out["source"] = "finnhub"
    return out[["symbol", "date", "epsActual", "epsEstimate", "source"]]


# -----------------------------
# 3) Merge, Finnhub priority
# -----------------------------
def merge_canonical(kaggle_df: pd.DataFrame, finnhub_df: pd.DataFrame) -> pd.DataFrame:
    all_df = pd.concat([kaggle_df, finnhub_df], ignore_index=True)

    # drop garbage rows
    all_df = all_df.dropna(subset=["symbol", "date"])

    # Finnhub wins on same (symbol, date)
    priority = {"finnhub": 0, "kaggle": 1}
    all_df["priority"] = all_df["source"].map(priority).fillna(9)

    all_df = (
        all_df.sort_values(["symbol", "date", "priority"])
             .drop_duplicates(subset=["symbol", "date"], keep="first")
             .drop(columns=["priority"])
             .sort_values(["date", "symbol"])
             .reset_index(drop=True)
    )

    return all_df


def main():
    # Trading index for PEAD decay (aligns to your prices file)
    prices = load_sp500_adj_close()
    trading_index = prices.index

    # --- Load Kaggle ---
    kaggle_path = RAW_DIR / "kaggle_earnings.csv"
    if not kaggle_path.exists():
        raise FileNotFoundError(f"Missing Kaggle file: {kaggle_path}")

    print("[Kaggle] Loading → canonical")
    kaggle = load_kaggle_as_canonical(kaggle_path)
    print(f"[Kaggle] rows: {len(kaggle):,}")

    # --- Fetch Finnhub ---
    print("[Finnhub] Fetching earnings history")
    finnhub_raw = fetch_earnings_history(start="2021-01-01")
    finnhub_raw.to_csv(RAW_DIR / "finnhub_earnings.csv", index=False)
    print(f"[Finnhub] rows: {len(finnhub_raw):,}")

    finnhub = finnhub_to_canonical(finnhub_raw)

    # --- Merge ---
    print("[Merge] Kaggle + Finnhub (Finnhub priority)")
    events = merge_canonical(kaggle, finnhub)
    print(f"[Merge] unique (symbol,date): {len(events):,}")

    # Save canonical events
    events_out = PROCESSED_DIR / "earnings_events.parquet"
    events.to_parquet(events_out, index=False)
    print(f"[Saved] {events_out}")

    # --- Build surprise + z ---
    print("[Signal] Compute EPS surprise")
    event_surprise = compute_eps_surprise(
        events,
        date_col="date",
        ticker_col="symbol",
        actual_col="epsActual",
        estimate_col="epsEstimate",
    )

    print("[Signal] Standardize within ticker (expanding z-score)")
    event_z = standardize_surprise_within_ticker(event_surprise)

    # --- Daily decayed sentiment on trading days ---
    print("[Signal] Build daily decayed sentiment (trading days)")
    daily = build_daily_decayed_sentiment(
        event_z,
        trading_index=trading_index,
        half_life_days=42,
        active_window_days=126,
    )

    daily_out = PROCESSED_DIR / "earnings_sentiment_daily.parquet"
    daily.to_parquet(daily_out)
    print(f"[Saved] {daily_out}")

    print("✅ build_earnings_sentiment.py complete.")


if __name__ == "__main__":
    main()



