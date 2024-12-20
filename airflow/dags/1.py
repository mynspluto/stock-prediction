import os
import yfinance as yf
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
from hdfs import InsecureClient
import subprocess

local_path = '/home/mynspluto/Project/stock-prediction/airflow/data/stock-history'
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
        
def upload_json_to_hdfs(local_path, ticker):
    # local_path에서 모든 JSON 파일 찾기
    for root, dirs, files in os.walk(f"{local_path}/{ticker}"):
        for file in files:
            if file.endswith('.json'):
                # 각 JSON 파일 경로
                local_file_path = os.path.join(root, file)

                # HDFS 경로 설정 (디렉토리 구조를 유지하면서 업로드)
                hdfs_path = f"/stock-history/{ticker}/{file}"

                # JSON 파일을 HDFS에 업로드
                client.upload(hdfs_path, local_file_path)
                print(f"Uploaded {local_file_path} to HDFS path {hdfs_path}")
                

def combine_stock_files(client, hdfs_input_path, hdfs_output_path):
    """
    HDFS에 저장된 월별 주식 데이터 파일들을 하나로 합치는 함수
    
    Args:
        client: HDFS client 객체
        hdfs_input_path: 입력 디렉토리 경로 (예: '/stock-history/^IXIC')
        hdfs_output_path: 출력 파일 경로 (예: '/stock-history/^IXIC/combined.json')
    """
    # Map 단계: 모든 파일에서 데이터 수집
    all_records = []
    try:
        # 입력 디렉토리의 모든 JSON 파일 리스트 가져오기
        json_files = [f for f in client.list(hdfs_input_path) if f.endswith('.json')]
        
        # 각 파일의 데이터 읽기
        for json_file in json_files:
            input_file_path = f"{hdfs_input_path}/{json_file}"
            with client.read(input_file_path) as reader:
                content = reader.read()
                records = json.loads(content.decode('utf-8'))
                all_records.extend(records)
        
        # Reduce 단계: 날짜별로 데이터 정리
        # 1. 날짜를 키로 사용하여 딕셔너리 생성
        records_dict = {}
        for record in all_records:
            date = record['Date']
            if date not in records_dict:
                records_dict[date] = record
        
        # 2. 날짜순으로 정렬
        sorted_records = sorted(records_dict.values(), key=lambda x: x['Date'])
        
        # 결과를 HDFS에 저장
        output_data = json.dumps(sorted_records, indent=2)
        with client.write(hdfs_output_path) as writer:
            writer.write(output_data.encode('utf-8'))
            
        print(f"Successfully combined {len(json_files)} files into {hdfs_output_path}")
        print(f"Total records: {len(sorted_records)}")
        
    except Exception as e:
        print(f"Error combining files: {str(e)}")
        raise
    
def run_hadoop_mapreduce(input_path, output_path):
    """
    HDFS 클라이언트를 통해 하둡 스트리밍 맵리듀스 작업 실행
    """
    # Hadoop 홈 디렉토리 설정
    hadoop_home = os.path.expanduser('~/hadoop-3.4.1')
    hadoop_bin = os.path.join(hadoop_home, 'bin/hadoop')
    
    # Hadoop 스트리밍 JAR 파일 경로
    streaming_jar = os.path.join(hadoop_home, 'share/hadoop/tools/lib/hadoop-streaming-*.jar')
    streaming_jar = os.popen(f'ls {streaming_jar}').read().strip()
    
    # MapReduce 작업 실행 명령어
    hadoop_command = [
        hadoop_bin, 'jar', streaming_jar,
        '-files', './airflow/dags/stock_mapper.py,./airflow/dags/stock_reducer.py',
        '-mapper', 'python3 stock_mapper.py',
        '-reducer', 'python3 stock_reducer.py',
        '-input', input_path,
        '-output', output_path
    ]
    
    try:
        # MapReduce 작업 실행
        process = subprocess.Popen(
            hadoop_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 실행 결과 출력
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print("MapReduce job completed successfully")
            print(stdout.decode())
        else:
            print("MapReduce job failed")
            print(stderr.decode())
            
    except Exception as e:
        print(f"Error running MapReduce job: {str(e)}")
        raise


        
#ticker = '^IXIC'
# fetch_stock_data(local_path, tickers)
# print_json_records(local_path, tickers[0])
# split_json_by_month(local_path, tickers[0])

client = InsecureClient('http://localhost:9870', user='hadoop')
#upload_json_to_hdfs(local_path, tickers[0])

# input_path = '/stock-history/^IXIC'
# output_path = '/stock-history/^IXIC/combined.json'
# combine_stock_files(client, input_path, output_path)

input_path = '/stock-history/^IXIC'
output_path = '/stock-history/^IXIC/combined_mapreduce'
run_hadoop_mapreduce(input_path, output_path)
#./hdfs dfs -cat /stock-history/^IXIC/combined_mapreduce/part-00000