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


from airflow import DAG
from airflow.operators.python import PythonOperator
from io import BytesIO

stock_data_path = '/home/mynspluto/Project/stock-prediction/airflow/data/stock-history'
hadoop_url = 'http://localhost:9870'
hdfs_path = "/stock-history"
hadoop_home = "/home/mynspluto/hadoop-3.4.1"
tickers = ['^IXIC']

def run_hadoop_mapreduce(tickers):
    """
    HDFS 클라이언트를 통해 하둡 스트리밍 맵리듀스 작업 실행
    """
    for ticker in tickers:
      try:
        input_path = f'{hdfs_path}/{ticker}/monthly/*'
        output_path = f'{hdfs_path}/{ticker}/combined_mapreduce'
      
        # Hadoop 홈 디렉토리 설정
        hadoop_bin = os.path.join(hadoop_home, 'bin/hadoop')
        
        # 현재 DAG 파일의 디렉토리 경로 가져오기
        dag_folder = os.path.dirname(os.path.abspath(__file__))
        mapper_path = os.path.join(dag_folder, 'mapreduce/stock_mapper.py')
        reducer_path = os.path.join(dag_folder, 'mapreduce/stock_reducer.py')
        
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
            '-files', f'{mapper_path},{reducer_path}',
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
    
    return 1
      
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


def read_combined_data_from_hdfs(ticker):
    """
    MapReduce 결과로 생성된 통합 데이터를 HDFS에서 읽어오기
    """
    try:
        client = InsecureClient(hadoop_url)
        mapreduce_output_path = f'{hdfs_path}/{ticker}/combined_mapreduce/part-00000'
        
        with client.read(mapreduce_output_path, encoding='utf-8') as reader:
            # MapReduce 출력을 DataFrame으로 변환
            lines = reader.readlines()
            data = []
            
            for line in lines:
                if line.strip():  # Skip empty lines
                    try:
                        row = json.loads(line.strip())
                        data.append(row)
                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON line: {line.strip()}, Error: {str(e)}")
            
            if not data:
                raise ValueError("No valid data found in the HDFS file")
            
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            print(f"Successfully loaded {len(df)} rows of data")
            return df
            
    except Exception as e:
        print(f"HDFS에서 데이터 읽기 실패: {str(e)}")
        raise
            
    except Exception as e:
        print(f"HDFS에서 데이터 읽기 실패: {str(e)}")
        raise

def save_model_results_to_hdfs(ticker, results):
    """모델 결과를 HDFS에 저장"""
    try:
        client = InsecureClient(hadoop_url)
        model_path = f'{hdfs_path}/{ticker}/model_results'
        
        # next_day_prediction DataFrame을 JSON 직렬화
        next_day_pred_dict = {}
        for col in results['next_day_prediction'].columns:
            next_day_pred_dict[col] = results['next_day_prediction'][col].iloc[0]
        
        # 결과 데이터 준비
        results_dict = {
            'metrics': results['metrics'],
            'next_day_prediction': next_day_pred_dict,
            'feature_importance': results['feature_importance'],
            'prediction_date': results['next_day_prediction'].index[0].strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat()
        }
        
        # JSON으로 변환
        results_json = json.dumps(results_dict)
        
        # 임시 파일 생성
        temp_path = '/tmp/latest_results.json'
        with open(temp_path, 'w') as f:
            f.write(results_json)
        
        # HDFS에 업로드
        client.upload(f'{model_path}/latest_results.json', 
                     temp_path,
                     overwrite=True)
        
        # 임시 파일 삭제
        os.remove(temp_path)
            
        print(f"모델 결과가 성공적으로 저장됨: {model_path}/latest_results.json")
        print(f"저장된 예측 결과: {next_day_pred_dict}")
        
    except Exception as e:
        print(f"모델 결과 저장 실패: {str(e)}")
        raise

def process_stock_predictions(**context):
    """
    주가 예측 프로세스를 실행하는 Airflow 태스크
    """
    try:
        for ticker in tickers:
            # 1. MapReduce 결과 데이터 읽기
            print(f"{ticker} 데이터 읽는 중...")
            df = read_combined_data_from_hdfs(ticker)
            
            # 2. 예측 모델 실행
            print(f"{ticker} 예측 모델 실행 중...")
            results = run_stock_prediction(df)
            
            # 3. 결과 저장
            print(f"{ticker} 결과 저장 중...")
            save_model_results_to_hdfs(ticker, results)
            
            print(f"{ticker} 처리 완료")
            
    except Exception as e:
        print(f"주가 예측 처리 중 오류 발생: {str(e)}")
        raise

with DAG(
    "update_stock_prediction_model_2",
    default_args={
        "owner": "airflow",
        "depend_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="월 별(YYYY-MM)로 저장된 주가 데이터를 합친후 주가 예측 모델 져장",
    schedule=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:

    #~/hadoop-3.4.1/bin/hdfs dfs -cat /stock-history/^IXIC/combined_mapreduce/part-00000
    merge_stock_data = PythonOperator(
        task_id="merge_stock_data_mapreduce",
        python_callable=run_hadoop_mapreduce,
        op_args=[tickers],
    )
    
    run_prediction = PythonOperator(
        task_id="run_stock_prediction",
        python_callable=process_stock_predictions,
    )
    
    merge_stock_data >> run_prediction
