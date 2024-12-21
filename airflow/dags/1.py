import os
import yfinance as yf
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
from hdfs import InsecureClient
import subprocess
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

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
        # numerical_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        # for col in numerical_columns:
        #     if col in df.columns:
        #         df[col] = df[col].round(0).astype(int)

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
        
def upload_json_to_hdfs(client, local_path, ticker):
    # Ensure the HDFS directory exists
    hdfs_base_path = f"/stock-history/{ticker}"
    try:
        if not client.status(hdfs_base_path, strict=False):
            client.makedirs(hdfs_base_path)
            print(f"Created HDFS directory: {hdfs_base_path}")
    except Exception as e:
        print(f"Creating directory {hdfs_base_path}: {str(e)}")
        client.makedirs(hdfs_base_path)

    # Upload files
    for root, dirs, files in os.walk(f"{local_path}/{ticker}"):
        for file in files:
            if file.endswith('.json'):
                local_file_path = os.path.join(root, file)
                hdfs_path = f"{hdfs_base_path}/{file}"
                
                try:
                    # Check if file exists
                    if client.status(hdfs_path, strict=False):
                        print(f"File {hdfs_path} already exists, overwriting...")
                        client.delete(hdfs_path)
                    
                    # Upload the file
                    client.upload(hdfs_path, local_file_path)
                    print(f"Uploaded {local_file_path} to HDFS path {hdfs_path}")
                except Exception as e:
                    print(f"Error uploading {local_file_path}: {str(e)}")
                

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
    try:
        # Hadoop 홈 디렉토리 설정
        hadoop_home = os.path.expanduser('~/hadoop-3.4.1')
        hadoop_bin = os.path.join(hadoop_home, 'bin/hadoop')
        
        # Delete output directory if it exists
        delete_command = [hadoop_bin, 'fs', '-rm', '-r', output_path]
        subprocess.run(delete_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Cleaned up existing output directory: {output_path}")
        
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
    
def create_ml_features(df):
    """
    주가 데이터프레임에서 머신러닝용 특징 생성
    """
    # 기존 데이터 복사
    df_ml = df.copy()
    
    # 이동평균선
    df_ml['MA5'] = df_ml['Close'].rolling(window=5).mean()
    df_ml['MA20'] = df_ml['Close'].rolling(window=20).mean()
    df_ml['MA60'] = df_ml['Close'].rolling(window=60).mean()
    
    # 거래량 이동평균
    df_ml['Volume_MA5'] = df_ml['Volume'].rolling(window=5).mean()
    
    # 변동성 지표
    df_ml['Price_Range'] = df_ml['High'] - df_ml['Low']
    df_ml['Price_Change'] = df_ml['Close'] - df_ml['Open']
    
    # RSI (Relative Strength Index)
    delta = df_ml['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_ml['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df_ml['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_ml['Close'].ewm(span=26, adjust=False).mean()
    df_ml['MACD'] = exp1 - exp2
    df_ml['Signal_Line'] = df_ml['MACD'].ewm(span=9, adjust=False).mean()
    
    # 결측값 처리
    df_ml = df_ml.fillna(method='bfill')
    
    return df_ml

def prepare_ml_data(df_ml, prediction_days=1):
    """
    머신러닝 모델을 위한 데이터 준비 - 기본 가격 데이터만 사용
    """
    # 기본 가격 데이터만 사용
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    target_columns = ['Open', 'High', 'Low', 'Close']
    
    # 데이터 시프트하여 다음날 가격을 예측하도록 설정
    X = df_ml[feature_columns].values[:-prediction_days]
    y = df_ml[target_columns].shift(-prediction_days).values[:-prediction_days]
    
    # NaN 값 제거
    valid_idx = ~np.isnan(y).any(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    
    # 학습/테스트 분할 (80:20)
    split_point = int(len(X) * 0.8)
    X_train = X[:split_point]
    X_test = X[split_point:]
    y_train = y[:split_point]
    y_test = y[split_point:]
    
    # 데이터 정규화
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    
    return (X_train, X_test, y_train, y_test), (scaler_X, scaler_y)

def train_stock_model(X_train, y_train):
    """
    주가 예측 모델 학습
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # 모든 CPU 코어 사용
    )
    model.fit(X_train, y_train)
    return model

def predict_stock_prices(df, model, scalers):
    """
    학습된 모델을 사용하여 주가 예측 - 기본 가격 데이터만 사용
    """
    scaler_X, scaler_y = scalers
    
    # 예측을 위한 마지막 데이터 준비
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    X_pred = df[feature_columns].values[-1:]
    X_pred_scaled = scaler_X.transform(X_pred)
    
    # 예측
    predictions_scaled = model.predict(X_pred_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    
    # 마지막 날짜 가져오기
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    next_date = last_date + pd.Timedelta(days=1)
    
    # 예측 결과를 데이터프레임으로 변환
    pred_df = pd.DataFrame(predictions, 
                          columns=['Pred_Open', 'Pred_High', 'Pred_Low', 'Pred_Close'],
                          index=[next_date])
    
    return pred_df

def evaluate_predictions(y_true, y_pred, scalers):
    """
    예측 결과 평가
    """
    _, scaler_y = scalers
    
    # 스케일링된 데이터를 원래 스케일로 변환
    y_true = scaler_y.inverse_transform(y_true)
    y_pred = scaler_y.inverse_transform(y_pred)
    
    # 각 가격 유형별 RMSE 계산
    price_types = ['Open', 'High', 'Low', 'Close']
    metrics = {}
    
    for i, price_type in enumerate(price_types):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i])) * 100
        
        metrics[price_type] = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    return metrics

def run_stock_prediction(df):
    """
    전체 주가 예측 프로세스 실행
    """
    try:
        # 1. 특징 생성
        print("특징 생성 중...")
        df_ml = create_ml_features(df)
        
        # 2. 데이터 준비
        print("데이터 준비 중...")
        (X_train, X_test, y_train, y_test), scalers = prepare_ml_data(df_ml)
        
        # 3. 모델 학습
        print("모델 학습 중...")
        model = train_stock_model(X_train, y_train)
        
        # 4. 예측 수행
        print("예측 수행 중...")
        y_pred = model.predict(X_test)
        
        # 5. 성능 평가
        print("성능 평가 중...")
        metrics = evaluate_predictions(y_test, y_pred, scalers)
        
        # 6. 다음 거래일 예측
        next_day_pred = predict_stock_prices(df, model, scalers)
        
        return {
            'model': model,
            'metrics': metrics,
            'next_day_prediction': next_day_pred,
            'feature_importance': dict(zip(df_ml.columns, model.feature_importances_))
        }
        
    except Exception as e:
        print(f"예측 중 오류 발생: {str(e)}")
        raise

#ticker = '^IXIC'
# fetch_stock_data(local_path, tickers)
# print_json_records(local_path, tickers[0])
# split_json_by_month(local_path, tickers[0])

# client = InsecureClient('http://localhost:9870', user='hadoop')
# upload_json_to_hdfs(client, local_path, tickers[0])

# input_path = '/stock-history/^IXIC'
# output_path = '/stock-history/^IXIC/combined_mapreduce'
# run_hadoop_mapreduce(input_path, output_path)
#~/hadoop-3.4.1/bin/hdfs dfs -cat /stock-history/^IXIC/combined_mapreduce/part-00000



# 주가 데이터를 데이터프레임으로 변환
df = pd.read_json(f"{local_path}/{tickers[0]}.json")

# 머신러닝 예측 실행
prediction_results = run_stock_prediction(df)

# 결과 출력
print("\n=== 예측 성능 평가 ===")
for price_type, metrics in prediction_results['metrics'].items():
    print(f"\n{price_type}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.2f}")

print("\n=== 다음 거래일 예측 ===")
print(prediction_results['next_day_prediction'])

print("\n=== 특징 중요도 (상위 5개) ===")
importance = sorted(prediction_results['feature_importance'].items(), 
                   key=lambda x: x[1], reverse=True)[:5]
for feature, score in importance:
    print(f"{feature}: {score:.4f}")