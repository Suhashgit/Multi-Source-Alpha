import pandas as pd
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = REPO_ROOT / "data/prices/sp500_adj_close.csv"

def load_sp500_adj_close() -> pd.DataFrame:
    df = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df

def compute_raw_momentum(prices:pd.DataFrame,
                         short_gap: int = 21,
                         lookback: int = 252) -> pd.DataFrame:
    p_short = prices.shift(short_gap)
    p_long = prices.shift(lookback)
    raw_mom = (p_short/p_long) - 1.0
    return raw_mom

def compute_momentum_zscore(raw_mom: pd.DataFrame) -> pd.DataFrame:
    mean = raw_mom.mean(axis=1)
    std = raw_mom.std(axis=1)
    z = (raw_mom.sub(mean, axis=0)).div(std, axis=0)
    return z

if __name__ == "__main__":
    prices = load_sp500_adj_close()
    print("Prices shape:", prices.shape)

    raw = compute_raw_momentum(prices)
    print("Raw momentum tail:")
    print(raw.tail())

    z = compute_momentum_zscore(raw)
    print("Momentum z-score tail:")
    print(z.tail())