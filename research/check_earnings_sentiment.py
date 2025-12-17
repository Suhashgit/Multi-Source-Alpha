import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Add the parent directory of 'multi_source_alpha' to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

SENT_PATH = "data/sentiment/processed/earnings_sentiment_daily.parquet"


def main():
    # ----------------------------
    # Load sentiment panel
    # ----------------------------
    sent = pd.read_parquet(SENT_PATH)

    print("Shape:", sent.shape)
    print("Date range:", sent.index.min(), "→", sent.index.max())

    # ----------------------------
    # CHECK 1: Non-zero clustering
    # ----------------------------
    print("\n=== CHECK 1: Non-zero clustering (single stock) ===")
    ticker = "A" if "A" in sent.columns else sent.columns[0]
    series = sent[ticker].replace(0, np.nan).dropna()

    print(f"Ticker used: {ticker}")
    print(series.head(20))

    # Optional visual
    series.iloc[:100].plot(
        title=f"Earnings sentiment decay (first 100 non-zero days) — {ticker}",
        figsize=(10, 4),
    )
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # CHECK 2: Distribution sanity
    # ----------------------------
    print("\n=== CHECK 2: Distribution sanity ===")
    stats = sent.stack().describe()
    print(stats)

    # ----------------------------
    # Optional diagnostics
    # ----------------------------
    print("\n=== OPTIONAL DIAGNOSTICS ===")

    # Sparsity
    nonzero_frac = (sent != 0).mean().mean()
    print(f"Fraction of non-zero entries: {nonzero_frac:.4f}")

    # Cross-sectional snapshot
    sample_date = sent.index[len(sent) // 2]
    cs = sent.loc[sample_date].replace(0, np.nan).dropna()
    print(f"\nCross-section on {sample_date.date()}:")
    print(cs.describe())


if __name__ == "__main__":
    main()
