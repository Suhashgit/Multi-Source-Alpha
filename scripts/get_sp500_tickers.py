import pandas as pd
import requests
import os
repo_root = os.path.dirname(os.path.dirname(__file__))
output_dir = os.path.join(repo_root,"data","Universe")
os.makedirs(output_dir,exist_ok=True)
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
resp.raise_for_status()
tables = pd.read_html(resp.text)
df = tables[0]
tickers = df.get('Symbol',df.iloc[:,0]).astype(str).str.replace('.','-',regex=False)
out_path = os.path.join(output_dir,"sp500_constituents.csv")
tickers.to_csv(out_path,index = False, header=False,encoding='utf-8')
print('Saved S&P 500 tickers!')