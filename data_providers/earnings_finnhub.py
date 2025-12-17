import os 
import time 
import pandas as pd
import requests
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
from datetime import date,timedelta
Base_URL = "https://finnhub.io/api/v1/"
def _get_key() -> str:
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("Finnhub API key not found. Please set the FINNHUB_API_KEY environment variable.")
    return key

def fetch_earnings_calendar(from_date:str, to_date:str) -> pd.DataFrame:
    key = _get_key()
    url = f"{Base_URL}calendar/earnings"
    params = {
        "from": from_date,
        "to": to_date,
    }
    headers = {
        "X-Finnhub-Token": key}
    
    r = requests.get(url,headers=headers,params=params,timeout=30)
    if r.status_code == 429:
        raise RuntimeError("Rate limit exceeded. Please try again later.")
    r.raise_for_status()
    payload = r.json()
    rows = payload.get("earningsCalendar",[])
    df = pd.DataFrame(rows)
    return df

def fetch_earnings_history(start ="2021-01-01", end = None, chunk_days = 90, sleep_s=1.0) -> pd.DataFrame:
    if end is None:
        end = date.today().isoformat()
    cur = pd.to_datetime(start).date()
    end_dt = pd.to_datetime(end).date()
    out =[]
    while cur <= end_dt:
        nxt = min(cur + timedelta(days=chunk_days - 1), end_dt)
        print(f"Fetching earnings from {cur} to {nxt}")
        df = fetch_earnings_calendar(cur.isoformat(), nxt.isoformat())
        if not df.empty:
            out.append(df)
        cur = nxt + timedelta(days=1)
        time.sleep(sleep_s)
    if not out:
        return pd.DataFrame()
    all_df = pd.concat(out, ignore_index=True).drop_duplicates()
    return all_df