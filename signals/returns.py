import pandas as pd
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = REPO_ROOT / 'data' / 'prices' / 'sp500_adj_close.csv'
def load_sp500_adj_close() -> pd.DataFrame:
    df = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df

def compute_forward_returns(prices:pd.DataFrame,
                            horizons = (1,5,21,63),
                            ) -> dict[int,pd.DataFrame]:
    fwd = {}
    for h in horizons:
        fwd[h] = (prices.shift(-h)/prices) - 1.0
    return fwd

if __name__ == "__main__":
    prices = load_sp500_adj_close()
    fwd = compute_forward_returns(prices)
    for h, df_h in fwd.items():
        print(f"Horizon {h} days, shape: {df_h.shape}")
        print(df_h.tail())