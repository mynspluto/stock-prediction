import os
import yfinance as yf
import pandas as pd
import json
from pathlib import Path
import time

from hdfs import InsecureClient
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# 환경 설정
ENVIRONMENT = os.getenv('AIRFLOW_ENV', 'local')  # 기본값은 local

# 환경별 설정
ENV_CONFIG = {
    'local': {
        'STOCK_DATA_PATH': str(Path.home() / 'project/stock-prediction/airflow/data/stock-history'),
        'HADOOP_URL': 'http://localhost:9870'
    },
    'kubernetes': {
        'STOCK_DATA_PATH': '/opt/airflow/stock_data',
        'HADOOP_URL': 'http://host.minikube.internal:9870'
    },
    'ec2-kubernetes': {
        'STOCK_DATA_PATH': '/opt/airflow/stock_data',
        'HADOOP_URL': 'http://18.190.148.99:9870'
    },
}

# 현재 환경의 설정 가져오기
current_config = ENV_CONFIG.get(ENVIRONMENT, ENV_CONFIG['local'])

# 환경변수 설정
# stock_data_path = '/home/mynspluto/Project/stock-prediction/airflow/data/stock-history'
# hadoop_url = 'http://localhost:9870'
stock_data_path = os.getenv('STOCK_DATA_PATH', current_config['STOCK_DATA_PATH'])
hadoop_url = os.getenv('HADOOP_URL', current_config['HADOOP_URL'])

hdfs_base_path = "/stock-history"
tickers = ['^IXIC']

def fetch_stock_data(stock_data_path, tickers):
    if not os.path.exists(stock_data_path):
        os.makedirs(stock_data_path)
    
    for ticker in tickers:
        # 가장 최근 데이터 날짜 확인
        ticker_directory = f"{stock_data_path}/{ticker}"
        last_date = get_last_date(ticker_directory)
        
        # yfinance에서 데이터 가져오기
        stock = yf.Ticker(ticker)
        if last_date:
            # 마지막 데이터 날짜 다음날부터 데이터 가져오기
            df = stock.history(start=last_date + timedelta(days=1))
        else:
            # 처음부터 모든 데이터 가져오기
            df = stock.history(period="max")
        
        if df.empty:
            print(f"No new data found for {ticker}.")
            continue
        
        df.reset_index(inplace=True)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Drop Dividends and Stock Splits columns
        columns_to_drop = ['Dividends', 'Stock Splits']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Save new data
        save_to_monthly_files(df, ticker_directory, ticker)

def get_last_date(ticker_directory):
    """가장 최근 데이터의 날짜를 반환"""
    if not os.path.exists(ticker_directory):
        return None
    
    latest_date = None
    for filename in os.listdir(ticker_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(ticker_directory, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                if data:  # 파일에 데이터가 있는 경우
                    df = pd.DataFrame(data)
                    max_date = pd.to_datetime(df['Date']).max()
                    if latest_date is None or max_date > latest_date:
                        latest_date = max_date
    
    return latest_date

def save_to_monthly_files(df, ticker_directory, ticker):
    """새로운 데이터를 월별 파일로 저장"""
    if not os.path.exists(ticker_directory):
        os.makedirs(ticker_directory)
        print(f"Created directory: {ticker_directory}")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year-Month'] = df['Date'].dt.to_period('M')
    
    # 월별로 그룹화하여 저장
    for year_month, group in df.groupby('Year-Month'):
        month_file_path = f"{ticker_directory}/{year_month}.json"
        
        # 기존 파일이 있는 경우 데이터 병합
        if os.path.exists(month_file_path):
            with open(month_file_path, 'r') as f:
                existing_data = json.load(f)
            existing_df = pd.DataFrame(existing_data)
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
            
            # 중복 제거하여 병합
            combined_df = pd.concat([existing_df, group])
            combined_df = combined_df.drop_duplicates(subset=['Date'])
            combined_df = combined_df.sort_values('Date')
            
            # Year-Month 컬럼 제거
            combined_df = combined_df.drop(columns=['Year-Month'])
            
            # 저장
            combined_df.to_json(month_file_path, orient='records', date_format='iso')
            print(f"Updated data for {ticker} for {year_month} in {month_file_path}")
        else:
            # 새로운 파일 생성
            group = group.drop(columns=['Year-Month'])
            group.to_json(month_file_path, orient='records', date_format='iso')
            print(f"Created new file for {ticker} for {year_month} in {month_file_path}")


def split_json_by_month(stock_data_path, tickers):
    for ticker in tickers:
        json_file_path = f"{stock_data_path}/{ticker}.json"
        ticker_directory = f"{stock_data_path}/{ticker}"

        # Check if the JSON file exists
        if os.path.exists(json_file_path):
            # Create ticker directory if it doesn't exist
            if not os.path.exists(ticker_directory):
                os.makedirs(ticker_directory)
                print(f"Created directory: {ticker_directory}")

            with open(json_file_path, "r") as file:
                data = json.load(file)  # Load the JSON content as a list of dictionaries

            # Convert the data to a DataFrame
            df = pd.DataFrame(data)

            # Convert Date column to datetime format
            df['Date'] = pd.to_datetime(df['Date'])

            # Extract Year-Month for grouping
            df['Year-Month'] = df['Date'].dt.to_period('M')

            # Group by Year-Month and save each group as a separate JSON file
            for year_month, group in df.groupby('Year-Month'):
                # Remove the Year-Month column for clean data
                group = group.drop(columns=['Year-Month'])

                # Save the data to a new JSON file
                month_file_path = f"{ticker_directory}/{year_month}.json"
                group.to_json(month_file_path, orient='records', date_format='iso')
                print(f"Data for {ticker} for {year_month} saved to {month_file_path}")
        else:
            print(f"JSON file not found: {json_file_path}")
            
def upload_json_to_hdfs(stock_data_path, tickers):
    client = InsecureClient(
        hadoop_url,
        user='hadoop'
    )
    
    for ticker in tickers:
        hdfs_ticker_data_path = f"{hdfs_base_path}/{ticker}/monthly"
        try:
            if not client.status(hdfs_ticker_data_path, strict=False):
                client.makedirs(hdfs_ticker_data_path)
                print(f"Created HDFS directory: {hdfs_ticker_data_path}")
        except Exception as e:
            print(f"Creating directory {hdfs_ticker_data_path}: {str(e)}")
            
            client.makedirs(hdfs_ticker_data_path)

        # Upload files
        for root, dirs, files in os.walk(f"{stock_data_path}/{ticker}"):
            for file in files:
                if file.endswith('.json'):
                    local_file_path = os.path.join(root, file)
                    hdfs_path = f"{hdfs_ticker_data_path}/{file}"
                    
                    try:
                        # Check if file exists
                        if client.status(hdfs_path, strict=False):
                            print(f"File {hdfs_path} already exists, overwriting...")
                            client.delete(hdfs_path)
                        
                        # Upload the file
                        client.upload(hdfs_path, local_file_path, overwrite=True)
                        print(f"Uploaded {local_file_path} to HDFS path {hdfs_path}")
                    except Exception as e:
                        print(f"Error uploading {local_file_path}: {str(e)}")
                        
with DAG(
    "download_upload_stock_data",
    default_args={
        "owner": "airflow",
        "depend_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="주가 데이터를 수집하여 월별로 분할하여 파일 저장",
    schedule=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:
    
    download_stock_data_task = PythonOperator(
        task_id="download_stock_data_task",
        python_callable=fetch_stock_data,
        op_args=[stock_data_path, tickers],
    )
    
    upload_stock_data_to_hdfs = PythonOperator(
        task_id="upload_json_to_hdfs_task",
        python_callable=upload_json_to_hdfs,
        op_args=[stock_data_path, tickers],
    )
    
    download_stock_data_task >> upload_stock_data_to_hdfs
