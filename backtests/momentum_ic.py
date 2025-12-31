import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from multi_source_alpha.research.combine_factors import combine_momentum_and_returns


def compute_ic_series(mom: pd.DataFrame, fwd21: pd.DataFrame) -> pd.Series:
    """
    Compute daily Spearman IC between momentum and 21d forward returns.
    """
    ics = []

    for date in mom.index:
        mom_t = mom.loc[date]
        fwd_t = fwd21.loc[date]

        # Drop NaNs
        valid = mom_t.dropna().index.intersection(fwd_t.dropna().index)
        if len(valid) < 30:
            ics.append(np.nan)
            continue

        mom_v = mom_t.loc[valid]
        fwd_v = fwd_t.loc[valid]

        # Spearman returns an object; take .correlation
        ic = spearmanr(mom_v, fwd_v).correlation
        ics.append(ic)

    ic_series = pd.Series(ics, index=mom.index)
    return ic_series


def summarize_ic(ic_series: pd.Series) -> dict:
    """
    Compute mean IC, std, t-stat, and percent positive IC.
    """
    valid = ic_series.dropna()

    mean_ic = valid.mean()
    std_ic = valid.std()
    t_stat = mean_ic / (std_ic / np.sqrt(valid.count()))
    pct_positive = (valid[valid > 0].count() / valid.count()) * 100

    return {
        "Mean IC": mean_ic,
        "IC Std Dev": std_ic,
        "IC t-stat": t_stat,
        "Pct Positive IC": pct_positive,
    }


if __name__ == "__main__":
    combined = combine_momentum_and_returns()
    mom = combined["mom"]
    fwd63 = combined["fwd_63d"]

    ic_series = compute_ic_series(mom, fwd63)
    summary = summarize_ic(ic_series)

    print("IC Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nLast few IC values:")
    print(ic_series.tail())