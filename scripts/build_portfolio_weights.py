import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
# Add the parent directory of 'multi_source_alpha' to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# ------------------------
# Paths (relative to repo root when you run `python -m ...`)
# ------------------------
DATA = Path("data")
OUT_DIR = DATA / "portfolio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MOM_PATH = Path("C:/Users/Suhas/Quant projects/multi_source_alpha/data/signals/momentum_z.parquet")
SENT_PATH = Path("C:/Users/Suhas/Quant projects/multi_source_alpha/data/sentiment/processed/earnings_sentiment_daily.parquet")
VOL_PATH = Path("C:/Users/Suhas/Quant projects/multi_source_alpha/data/volume/processed/volume_shock_z.parquet")

OUT_PATH = OUT_DIR / "weights_long_only.parquet"


def normalize_long_only(W: pd.DataFrame, cap: float = 0.02) -> pd.DataFrame:
    """
    Long-only normalization that preserves sleeve hierarchy.

    Steps per day:
      1) Normalize raw scores so sum(weights)=1
      2) Cap each name to `cap`
      3) Renormalize so sum(weights)=1 again
    """
    W = W.clip(lower=0.0)

    # 1) normalize
    s = W.sum(axis=1).replace(0, np.nan)
    W = W.div(s, axis=0).fillna(0.0)

    # 2) cap
    if cap is not None:
        W = W.clip(upper=cap)
        # 3) renormalize after cap
        s2 = W.sum(axis=1).replace(0, np.nan)
        W = W.div(s2, axis=0).fillna(0.0)

    return W


def main():
    print("[Load] Signals")
    mom = pd.read_parquet(MOM_PATH)
    sent = pd.read_parquet(SENT_PATH)
    vol = pd.read_parquet(VOL_PATH)

    # Align on common dates + tickers
    common_dates = mom.index.intersection(sent.index).intersection(vol.index)
    common_cols = mom.columns.intersection(sent.columns).intersection(vol.columns)

    mom = mom.loc[common_dates, common_cols]
    sent = sent.loc[common_dates, common_cols]
    vol = vol.loc[common_dates, common_cols]

    print("Aligned shape:", mom.shape)

    # ------------------------
    # Cross-sectional thresholds (row-wise, broadcast-safe)
    # ------------------------
    mom_hi = mom.ge(mom.quantile(0.8, axis=1), axis=0)   # top 20%
    mom_lo = mom.le(mom.quantile(0.2, axis=1), axis=0)   # bottom 20%

    # "Neutral sentiment" = in the middle by magnitude (avoid extremes)
    sent_neutral = sent.abs().le(sent.abs().quantile(0.6, axis=1), axis=0)

    # "Very negative sentiment" = bottom 20% (mean-reversion sleeve)
    sent_very_neg = sent.le(sent.quantile(0.2, axis=1), axis=0)

    # High volume shock = top 20%
    vol_hi = vol.ge(vol.quantile(0.8, axis=1), axis=0)
    vol_not_hi = ~vol_hi

    # ------------------------
    # Sleeve masks (your intended logic)
    # ------------------------
    core_long = mom_hi & sent_neutral & vol_not_hi
    mr_long = mom_lo & sent_very_neg & vol_hi

    # ------------------------
    # Raw scores (core > MR)
    # ------------------------
    raw = pd.DataFrame(0.0, index=common_dates, columns=common_cols)
    raw[core_long] = 1.0
    raw[mr_long] += 0.3
    # Normalize + cap
    W = normalize_long_only(raw, cap=0.02)

    # Save
    W.to_parquet(OUT_PATH)
    print(f"[Saved] {OUT_PATH}")

    # Diagnostics
    print("Median #positions/day:", (W > 0).sum(axis=1).median())
    print("Median max weight/day:", W.max(axis=1).median())
    print("Core longs/day (median):", core_long.sum(axis=1).median())
    print("MR longs/day (median):", mr_long.sum(axis=1).median())


if __name__ == "__main__":
    main()

