from multi_source_alpha.research.combine_factors import combine_momentum_and_returns
import pandas as pd
import numpy as np


def extract_momentum_and_fwd(df, horizon_label="fwd_63d"):
    mom = df["mom"]
    fwd = df[horizon_label]
    return mom, fwd


def compute_decile_returns(mom, fwd, n_deciles=10, debug_date_idx=300):
    decile_returns = {d: [] for d in range(1, n_deciles + 1)}
    dates_used = []

    for idx, date in enumerate(mom.index):
        mom_t = mom.loc[date]
        fwd_t = fwd.loc[date]

        valid = mom_t.dropna().index.intersection(fwd_t.dropna().index)
        if len(valid) < 50:
            continue

        mom_t = mom_t.loc[valid]
        fwd_t = fwd_t.loc[valid]

        # Rank momentum cross-sectionally (ascending=True => lowest mom gets rank 1)
        ranks = mom_t.rank(method="first")

        bin_size = len(valid) / n_deciles
        decile_cut = np.floor((ranks - 1) / bin_size) + 1
        decile_cut = decile_cut.clip(1, n_deciles)

        # --- DEBUG / SANITY CHECK (runs once) ---
        if idx == debug_date_idx:
            print(f"\nDEBUG DATE: {date}")
            print("Momentum mean by decile (should be increasing if Decile 1 = lowest momentum):")
            for d in range(1, n_deciles + 1):
                members = decile_cut[decile_cut == d].index
                print(d, mom_t.loc[members].mean())
            print("If this is decreasing, your decile labels are flipped.\n")
            # no break here; you can break if you want only this date
            # break

        for d in range(1, n_deciles + 1):
            members = decile_cut[decile_cut == d].index
            decile_returns[d].append(fwd_t.loc[members].mean())

        dates_used.append(date)

    return pd.DataFrame(decile_returns, index=pd.DatetimeIndex(dates_used)).sort_index()


if __name__ == "__main__":
    combined = combine_momentum_and_returns()
    mom, fwd = extract_momentum_and_fwd(combined, horizon_label="fwd_63d")

    decile_df = compute_decile_returns(mom, fwd, n_deciles=10, debug_date_idx=300)

    print("Mean forward return by decile:")
    print(decile_df.mean())
