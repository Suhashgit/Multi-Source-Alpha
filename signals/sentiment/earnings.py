import numpy as np
import pandas as pd 
def compute_eps_surprise(events: pd.DataFrame,
                         date_col = "date",
                         ticker_col = "symbol",
                         actual_col = "epsActual",
                         estimate_col = "epsEstimate",
                         eps = 1e-6) -> pd.DataFrame:
    df = events.copy()
    df = df.rename(columns={
        ticker_col: "ticker",
        date_col: "event_date",
        actual_col: "eps_actual",
        estimate_col: "eps_est"
    })