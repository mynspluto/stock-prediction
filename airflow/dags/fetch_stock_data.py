import os
import yfinance as yf
import pandas as pd

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

local_path = '/home/mynspluto/Project/stock-prediction/airflow'
tickers = ['^IXIC']

def fetch_stock_data(local_path, tickers):
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df = stock.history(period='max')
        df.reset_index(inplace=True)
        df.to_csv(f"{local_path}/{ticker}.csv", index=False)

with DAG(
    "stock-predict",
    default_args={
        "owner": "airflow",
        "depend_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="Examples of Kafka Operators with HDFS",
    schedule=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:
    
    fetch_stock_task = PythonOperator(
        task_id="fetch_stock_data_task",
        python_callable=fetch_stock_data,
        op_args=[local_path, tickers],  # 로컬 경로와 티커 리스트를 인자로 전달
    )
    
    fetch_stock_task
