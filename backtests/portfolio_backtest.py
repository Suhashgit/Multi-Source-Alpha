import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ------------------------
# Paths (robust: absolute repo root)
# ------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]  # .../multi_source_alpha
DATA_DIR = REPO_ROOT / "data"

WEIGHTS_PATH = DATA_DIR / "portfolio" / "weights_long_only.parquet"
PRICES_PATH = DATA_DIR / "prices" / "sp500_adj_close.csv"

OUT_DIR = DATA_DIR / "portfolio"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_EQUITY_PATH = OUT_DIR / "equity_curve.parquet"
OUT_METRICS_PATH = OUT_DIR / "backtest_metrics.csv"
OUT_PLOT_PATH = OUT_DIR / "equity_curve.png"

TRADING_DAYS = 252


# ------------------------
# Metrics helpers
# ------------------------
def annualized_sharpe(daily_rets: pd.Series, freq: int = TRADING_DAYS) -> float:
    daily_rets = daily_rets.dropna()
    if daily_rets.empty or daily_rets.std(ddof=0) == 0:
        return np.nan
    return np.sqrt(freq) * daily_rets.mean() / daily_rets.std(ddof=0)


def max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna()
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return dd.min()


def turnover(weights: pd.DataFrame) -> pd.Series:
    return weights.diff().abs().sum(axis=1)


def apply_transaction_costs(pnl: pd.Series, to: pd.Series, bps: float) -> pd.Series:
    """
    Linear costs: cost_t = (bps/10000) * turnover_t
    """
    if bps is None or bps <= 0:
        return pnl
    cost = (bps / 10000.0) * to.fillna(0.0)
    return pnl - cost


def plot_equity_curves(out_df: pd.DataFrame, save_path: Path):
    """
    Plots portfolio gross/net and benchmark equity curves and saves PNG.
    """
    plt.figure()
    out_df[["portfolio_gross", "portfolio_net", "benchmark_ew"]].dropna().plot()
    plt.xlabel("Date")
    plt.ylabel("Equity (Growth of $1)")
    plt.title("Equity Curve: Portfolio vs Equal-Weight Benchmark")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ------------------------
# Main backtest
# ------------------------
def main(tc_bps: float = 0.0):
    print("[Load] Weights:", WEIGHTS_PATH)
    W = pd.read_parquet(WEIGHTS_PATH)
    W.index = pd.to_datetime(W.index)

    print("[Load] Prices:", PRICES_PATH)
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True).sort_index()

    # Align tickers
    common_cols = W.columns.intersection(prices.columns)
    W = W[common_cols].sort_index()
    prices = prices[common_cols].sort_index()

    # Daily simple returns
    rets = prices.pct_change()

    # Align dates
    W, rets = W.align(rets, join="inner", axis=0)

    # Avoid lookahead: use yesterday's weights for today's returns
    W_lag = W.shift(1).fillna(0.0)

    # Portfolio daily return
    pnl = (W_lag * rets).sum(axis=1)

    # Turnover + optional transaction costs
    to = turnover(W).reindex(pnl.index)
    pnl_net = apply_transaction_costs(pnl, to, bps=tc_bps)

    # Equity curves
    equity = (1.0 + pnl).cumprod()
    equity_net = (1.0 + pnl_net).cumprod()

    # Benchmark: equal-weight return proxy from same universe
    bench_ret = rets.mean(axis=1).reindex(pnl.index)
    bench_equity = (1.0 + bench_ret).cumprod()

    # Metrics
    metrics = {
        "Sharpe (gross)": annualized_sharpe(pnl),
        "Sharpe (net)": annualized_sharpe(pnl_net),
        "Max Drawdown (gross)": max_drawdown(equity),
        "Max Drawdown (net)": max_drawdown(equity_net),
        "Avg Daily Turnover": float(to.mean()),
        "Benchmark Sharpe (EW)": annualized_sharpe(bench_ret),
        "Benchmark Max DD (EW)": max_drawdown(bench_equity),
        "Start": str(pnl.index.min().date()) if len(pnl) else "",
        "End": str(pnl.index.max().date()) if len(pnl) else "",
        "TC (bps)": tc_bps,
    }

    print("\n=== Portfolio Performance ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            if "Drawdown" in k:
                print(f"{k:24s}: {v: .2%}")
            else:
                print(f"{k:24s}: {v: .4f}")
        else:
            print(f"{k:24s}: {v}")

    # Save equity curves + diagnostics
    out = pd.DataFrame(
        {
            "portfolio_gross": equity,
            "portfolio_net": equity_net,
            "benchmark_ew": bench_equity,
            "pnl_gross": pnl,
            "pnl_net": pnl_net,
            "turnover": to,
        }
    )
    out.to_parquet(OUT_EQUITY_PATH)
    pd.DataFrame([metrics]).to_csv(OUT_METRICS_PATH, index=False)

    # Plot
    plot_equity_curves(out, OUT_PLOT_PATH)

    print("\n[Saved] Equity curve:", OUT_EQUITY_PATH)
    print("[Saved] Metrics:", OUT_METRICS_PATH)
    print("[Saved] Plot:", OUT_PLOT_PATH)


if __name__ == "__main__":
    # Set tc_bps to e.g. 5.0 for 5 bps per unit turnover
    main(tc_bps=0.0)
