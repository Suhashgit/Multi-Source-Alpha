import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

VOL_PATH = REPO_ROOT / "data/volume/processed/volume_shock_z.parquet"
PRICES_PATH = REPO_ROOT / "data/prices/sp500_adj_close.csv"


def load_forward_returns(horizon=21):
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    fwd = prices.shift(-horizon) / prices - 1.0
    return fwd


def compute_ic_series(signal, fwd):
    ics = []

    for date in signal.index:
        s = signal.loc[date]
        r = fwd.loc[date]

        valid = s.dropna().index.intersection(r.dropna().index)
        if len(valid) < 30:
            ics.append(np.nan)
            continue

        ic, _ = spearmanr(s.loc[valid], r.loc[valid])
        ics.append(ic)

    return pd.Series(ics, index=signal.index)


def main():
    print("[Load] Volume shock")
    vol = pd.read_parquet(VOL_PATH)

    print("[Load] Forward returns")
    fwd63 = load_forward_returns(63)

    common_dates = vol.index.intersection(fwd63.index)
    vol = vol.loc[common_dates]
    fwd63 = fwd63.loc[common_dates]
    print("[Compute] IC series")
    ic = compute_ic_series(vol, fwd63)

    mean_ic = ic.mean()
    t_stat = mean_ic / (ic.std() / np.sqrt(ic.count()))
    pct_pos = (ic > 0).mean() * 100

    print("\n=== Volume Shock IC Summary ===")
    print(f"Mean IC      : {mean_ic:.5f}")
    print(f"IC t-stat   : {t_stat:.2f}")
    print(f"% Positive  : {pct_pos:.2f}%")

    print("\nLast 10 IC values:")
    print(ic.tail(10))


if __name__ == "__main__":
    main()