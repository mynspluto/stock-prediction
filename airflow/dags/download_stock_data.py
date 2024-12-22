import os
import yfinance as yf
import pandas as pd
import json

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

local_path = '/home/mynspluto/Project/stock-prediction/airflow/data/stock-history'
tickers = ['^IXIC']

def fetch_stock_data(local_path, tickers):
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    
    for ticker in tickers:
        # 가장 최근 데이터 날짜 확인
        ticker_directory = f"{local_path}/{ticker}"
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


def split_json_by_month(local_path, tickers):
    for ticker in tickers:
        json_file_path = f"{local_path}/{ticker}.json"
        ticker_directory = f"{local_path}/{ticker}"

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

with DAG(
    "download_stock_data",
    default_args={
        "owner": "airflow",
        "depend_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="주가 데이터를 수집하여 월별로 분할하여 파일 저장",
    schedule=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:
    
    fetch_stock_data_task = PythonOperator(
        task_id="fetch_stock_data_task",
        python_callable=fetch_stock_data,
        op_args=[local_path, tickers],
    )
    
    split_stock_data_by_month = PythonOperator(
        task_id="split_stock_data_by_month_task",
        python_callable=split_json_by_month,
        op_args=[local_path, tickers],
    )
    
    fetch_stock_data_task >> split_stock_data_by_month
