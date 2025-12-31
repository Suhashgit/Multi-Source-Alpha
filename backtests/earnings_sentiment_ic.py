import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import sys
import os
# Add the parent directory of 'multi_source_alpha' to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from multi_source_alpha.signals.momentum import load_sp500_adj_close
from multi_source_alpha.signals.returns import compute_forward_returns
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
SENT_PATH = REPO_ROOT / "data" / "sentiment" / "processed" / "earnings_sentiment_daily.parquet"
sent = pd.read_parquet(SENT_PATH)

def compute_ic_series(sent: pd.DataFrame, fwd: pd.DataFrame, min_obs=50) -> pd.Series:
    """
    Cross-sectional Spearman IC at each date.
    """
    ics = []

    for date in sent.index:
        s_t = sent.loc[date]
        r_t = fwd.loc[date]

        valid = s_t.replace(0, np.nan).dropna().index.intersection(
            r_t.dropna().index
        )

        if len(valid) < min_obs:
            ics.append(np.nan)
            continue

        ic, _ = spearmanr(s_t.loc[valid], r_t.loc[valid])
        ics.append(ic)

    return pd.Series(ics, index=sent.index)


def summarize_ic(ic: pd.Series) -> None:
    ic = ic.dropna()

    mean_ic = ic.mean()
    std_ic = ic.std(ddof=0)
    t_stat = mean_ic / (std_ic / np.sqrt(len(ic)))
    pct_pos = (ic > 0).mean() * 100

    print("\n=== Earnings Sentiment IC Summary ===")
    print(f"Mean IC        : {mean_ic:.5f}")
    print(f"IC Std Dev    : {std_ic:.5f}")
    print(f"IC t-stat     : {t_stat:.2f}")
    print(f"% Positive IC : {pct_pos:.2f}%")


def main():
    print("[Load] Earnings sentiment")
    sent = pd.read_parquet(SENT_PATH)

    print("[Load] Prices & forward returns")
    prices = load_sp500_adj_close()
    fwd = compute_forward_returns(prices, horizons=(63,))[63]

    # Align dates
    common_dates = sent.index.intersection(fwd.index)
    sent = sent.loc[common_dates]
    fwd = fwd.loc[common_dates]

    print("[Compute] IC series")
    ic_series = compute_ic_series(sent, fwd)

    summarize_ic(ic_series)

    print("\nLast 10 IC values:")
    print(ic_series.tail(10))


if __name__ == "__main__":
    main()