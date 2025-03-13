import os
import yfinance as yf
import pandas as pd
import json
from pathlib import Path
import time

from datetime import datetime, timedelta


# 환경별 설정
ENV_CONFIG = {
    'local': {
        'STOCK_DATA_PATH': str(Path.home() / 'project/stock-prediction/airflow/data/stock-history'),
        'HADOOP_URL': 'http://localhost:9870'
    },
}

stock_data_path = 'project/stock-prediction/airflow/data/stock-history'
hadoop_url = 'http://localhost:9870'

hdfs_base_path = "/stock-history"
tickers = ['^IXIC']

def fetch_stock_data(stock_data_path, tickers):
    if not os.path.exists(stock_data_path):
        os.makedirs(stock_data_path)
    
    for ticker in tickers:
        # 가장 최근 데이터 날짜 확인
        ticker_directory = f"{stock_data_path}/{ticker}"
        
        # yfinance에서 데이터 가져오기
        stock = yf.Ticker(ticker)
        
        df = stock.history(period="max")
        
        print(df)

fetch_stock_data(stock_data_path, tickers)

