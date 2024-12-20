import os
import yfinance as yf
import json
import requests
from datetime import datetime, timedelta
import pandas as pd

local_path = '/home/mynspluto/Project/stock-prediction/airflow/data'
#webhdfs_url = 'http://localhost:9870/webhdfs/v1'
webhdfs_url = 'http://localhost:9870/webhdfs/v1'

# 주가 데이터를 수집할 종목 리스트
tickers = ['^IXIC']

# 주가 데이터를 저장할 함수
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
        numerical_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numerical_columns:
            if col in df.columns:
                df[col] = df[col].round(0).astype(int)

        # Save the DataFrame as JSON
        df.to_json(f"{local_path}/{ticker}.json", orient='records', date_format='iso')
        print(f"Data for {ticker} saved to {local_path}/{ticker}.json")
        
def print_json_records(local_path, ticker):
    json_file_path = f"{local_path}/{ticker}.json"

    # Check if the JSON file exists
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as file:
            data = json.load(file)  # Load the JSON content as a list of dictionaries

        # Loop through each record (row) and print it
        for record in data:
            print(record)
    else:
        print(f"JSON file not found: {json_file_path}")

def split_json_by_month(local_path, ticker):
    json_file_path = f"{local_path}/{ticker}.json"

    # Check if the JSON file exists
    if os.path.exists(json_file_path):
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
            month_file_path = f"{local_path}/{ticker}_{year_month}.json"
            group.to_json(month_file_path, orient='records', date_format='iso')
            print(f"Data for {ticker} for {year_month} saved to {month_file_path}")
    else:
        print(f"JSON file not found: {json_file_path}")
        
fetch_stock_data(local_path, tickers)

# 예시 호출
ticker = '^IXIC'
print_json_records(local_path, ticker)

split_json_by_month(local_path, ticker)
