# cd ml
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt

import os
import yfinance as yf
import pandas as pd
import json
from pathlib import Path
import time

stock_data_path = "./stock-data"


def fetch_stock_data():
    print("fetch_stock_data")
    if not os.path.exists(stock_data_path):
        print("2")
        os.makedirs(stock_data_path)

    # 가장 최근 데이터 날짜 확인
    ticker_directory = f"{stock_data_path}/^IXIC"
    last_date = get_last_date(ticker_directory)

    # yfinance에서 데이터 가져오기
    stock = yf.Ticker("^IXIC")
    if last_date:
        # 마지막 데이터 날짜 다음날부터 데이터 가져오기
        df = stock.history(start=last_date + timedelta(days=1))
    else:
        # 처음부터 모든 데이터 가져오기
        df = stock.history(period="max")

    df.reset_index(inplace=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Drop Dividends and Stock Splits columns
    columns_to_drop = ["Dividends", "Stock Splits"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    print(df)


fetch_stock_data()
