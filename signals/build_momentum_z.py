import pandas as pd
from pathlib import Path

from multi_source_alpha.signals.momentum import (
    load_sp500_adj_close,
    compute_raw_momentum,
    compute_momentum_zscore,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "signals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "momentum_z.parquet"


def main():
    print("[Load] Prices (Adj Close)")
    prices = load_sp500_adj_close()

    print("[Compute] Raw momentum (12m lookback, 1m skip)")
    raw = compute_raw_momentum(prices, short_gap=21, lookback=252)

    print("[Compute] Cross-sectional z-score momentum")
    mom_z = compute_momentum_zscore(raw)

    print("[Save] momentum_z.parquet")
    mom_z.to_parquet(OUT_PATH)

    print("Saved:", OUT_PATH)
    print("Shape:", mom_z.shape)
    print("Date range:", mom_z.index.min(), "to", mom_z.index.max())


if __name__ == "__main__":
    main()
