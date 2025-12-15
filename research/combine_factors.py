import pandas as pd
from multi_source_alpha.signals.momentum import (
    load_sp500_adj_close,
    compute_raw_momentum,
    compute_momentum_zscore,
)
from multi_source_alpha.signals.returns import compute_forward_returns


def combine_momentum_and_returns():
    """
    Build a panel with:
      - mom: momentum z-scores
      - fwd_1d, fwd_5d, fwd_21d: forward returns
    Columns: MultiIndex [panel_name, ticker]
    Index: dates
    """
    # --- Load data ---
    prices = load_sp500_adj_close()

    # --- Compute momentum ---
    raw = compute_raw_momentum(prices)
    mom_z = compute_momentum_zscore(raw)

    # --- Compute forward returns ---
    fwd = compute_forward_returns(prices, horizons=(1, 5, 21,63))
    fwd_1d = fwd[1]
    fwd_5d = fwd[5]
    fwd_21d = fwd[21]
    fwd_63d = fwd[63]

    # --- Align dates ---
    common_index = mom_z.index
    for df in (fwd_1d, fwd_5d, fwd_21d):
        common_index = common_index.intersection(df.index)

    mom = mom_z.loc[common_index]
    f1 = fwd_1d.loc[common_index]
    f5 = fwd_5d.loc[common_index]
    f21 = fwd_21d.loc[common_index]
    f63 = fwd_63d.loc[common_index]

    # --- Assign MultiIndex columns ---
    mom.columns = pd.MultiIndex.from_product([["mom"], mom.columns])
    f1.columns = pd.MultiIndex.from_product([["fwd_1d"], f1.columns])
    f5.columns = pd.MultiIndex.from_product([["fwd_5d"], f5.columns])
    f21.columns = pd.MultiIndex.from_product([["fwd_21d"], f21.columns])
    f63.columns = pd.MultiIndex.from_product([["fwd_63d"], f63.columns])
    # --- Combine everything ---
    combined = pd.concat([mom, f1, f5, f21,f63], axis=1)
    return combined

if __name__ == "__main__":
    combined = combine_momentum_and_returns()
    print("Combined DataFrame:")
    print(combined.head())
    print("Column structure:")
    print(combined.columns)