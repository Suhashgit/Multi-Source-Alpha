import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

VOL_PATH = REPO_ROOT / "data/volume/processed/volume_shock_z.parquet"
PRICES_PATH = REPO_ROOT / "data/prices/sp500_adj_close.csv"


def load_forward_returns(horizon=21):
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    fwd = prices.shift(-horizon) / prices - 1.0
    return fwd


def compute_decile_returns(signal, fwd, n_deciles=10):
    decile_returns = {d: [] for d in range(1, n_deciles + 1)}

    for date in signal.index:
        s = signal.loc[date]
        r = fwd.loc[date]

        valid = s.dropna().index.intersection(r.dropna().index)
        if len(valid) < 50:
            continue

        s = s.loc[valid]
        r = r.loc[valid]

        ranks = s.rank(method="first")
        bin_size = len(valid) / n_deciles
        decile = np.floor((ranks - 1) / bin_size) + 1
        decile = decile.clip(1, n_deciles)

        for d in range(1, n_deciles + 1):
            members = decile[decile == d].index
            decile_returns[d].append(r.loc[members].mean())

    return pd.DataFrame(decile_returns).mean()


def main():
    print("[Load] Volume shock")
    vol = pd.read_parquet(VOL_PATH)

    print("[Load] Forward returns")
    fwd63 = load_forward_returns(63)

    common_dates = vol.index.intersection(fwd63.index)
    vol = vol.loc[common_dates]
    fwd63 = fwd63.loc[common_dates]
    print("[Compute] Decile returns")
    mean_deciles = compute_decile_returns(vol, fwd63)

    print("\nMean forward return by volume shock decile:")
    print(mean_deciles)
    print("\nD10 - D1:", mean_deciles[10] - mean_deciles[1])


if __name__ == "__main__":
    main()