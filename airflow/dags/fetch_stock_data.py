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
        stock = yf.Ticker(ticker)
        df = stock.history(period="max")
        if df.empty:
            print(f"No data found for {ticker}.")
            continue

        df.reset_index(inplace=True)

        # Convert the Date column to YYYY-MM-DD format
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

        # Drop Dividends and Stock Splits columns if they exist
        columns_to_drop = ['Dividends', 'Stock Splits']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # Round numerical columns to integers
        # numerical_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        # for col in numerical_columns:
        #     if col in df.columns:
        #         df[col] = df[col].round(0).astype(int)

        # Save the DataFrame as JSON
        df.to_json(f"{local_path}/{ticker}.json", orient='records', date_format='iso')
        print(f"Data for {ticker} saved to {local_path}/{ticker}.json")

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
    "fetch_stock_data",
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
