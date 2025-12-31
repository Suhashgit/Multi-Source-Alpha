import pandas as pd
import numpy as np
import sys
import os
# Add the parent directory of 'multi_source_alpha' to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pathlib import Path
from multi_source_alpha.signals.momentum import load_sp500_adj_close
from multi_source_alpha.signals.returns import compute_forward_returns
REPO_ROOT = Path(__file__).resolve().parents[1]
SENT_PATH = REPO_ROOT / "data" / "sentiment" / "processed" / "earnings_sentiment_daily.parquet"
sent = pd.read_parquet(SENT_PATH)



def compute_decile_returns(sent: pd.DataFrame, fwd: pd.DataFrame, n_deciles=10):
    decile_returns = {d: [] for d in range(1, n_deciles + 1)}
    dates_used = []

    for date in sent.index:
        s_t = sent.loc[date]
        r_t = fwd.loc[date]

        valid = s_t.replace(0, np.nan).dropna().index.intersection(
            r_t.dropna().index
        )

        if len(valid) < 50:
            continue

        s = s_t.loc[valid]
        r = r_t.loc[valid]

        ranks = s.rank(method="first")
        bin_size = len(ranks) / n_deciles
        decile = np.floor((ranks - 1) / bin_size) + 1
        decile = decile.clip(1, n_deciles)

        for d in range(1, n_deciles + 1):
            members = decile[decile == d].index
            decile_returns[d].append(r.loc[members].mean())

        dates_used.append(date)

    return pd.DataFrame(decile_returns, index=pd.DatetimeIndex(dates_used))


def main():
    print("[Load] Earnings sentiment")
    sent = pd.read_parquet(SENT_PATH)

    print("[Load] Prices & forward returns")
    prices = load_sp500_adj_close()
    fwd = compute_forward_returns(prices, horizons=(63,))[63]

    common_dates = sent.index.intersection(fwd.index)
    sent = sent.loc[common_dates]
    fwd = fwd.loc[common_dates]

    print("[Compute] Decile returns")
    deciles = compute_decile_returns(sent, fwd)

    print("\nMean forward return by sentiment decile:")
    print(deciles.mean())


if __name__ == "__main__":
    main()