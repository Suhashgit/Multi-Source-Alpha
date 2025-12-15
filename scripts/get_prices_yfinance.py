import pandas as pd
import yfinance as yf
from pathlib import Path

Data_dir = Path('data/prices')
Data_dir.mkdir(parents=True, exist_ok=True)
sp500_tickers_path = Path('data/Universe/sp500_constituents.csv')
tickers = pd.read_csv(sp500_tickers_path, header=None)[0].tolist()
df = yf.download(
    tickers,
    start = "2000-01-01",
    end = None,
    auto_adjust=False
)
adj_close = df["Adj Close"]
out_path = Data_dir / 'sp500_adj_close.csv'
adj_close.to_csv(out_path, index=True, header=True)
print(f'Saved prices to {out_path}')