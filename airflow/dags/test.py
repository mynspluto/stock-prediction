import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import os
import pickle
import numpy as np
import pandas as pd
import json
from pathlib import Path
import time
from datetime import datetime, timedelta
import requests

from airflow import DAG
from airflow.operators.python import PythonOperator

# 시스템에 설치된 폰트 설정
plt.rc('font', family='NanumGothic')

feature_sets = {
    "기본 특성": ['Close', 'Open', 'High', 'Low', 'Volume'],
    "기본 특성 + 기술 특성": [
        'Close', 'Open', 'High', 'Low', 'Volume',
        'SMA_5', 'SMA_20', 'SMA_60', 'RSI', 'MACD', 'HL_Ratio'
    ],
    # "기본 특성(1차 차분)": [
    #     'Close_Diff', 'Open_Diff', 'High_Diff', 'Low_Diff', 'Volume_Diff'
    # ],
    # "기본 특성(2차 차분)": [
    #     'Close_Diff2', 'Open_Diff2', 'High_Diff2', 'Low_Diff2', 'Volume_Diff2'
    # ],
    # "기본 특성(변화율)": [
    #     'Close_Change', 'Open_Change', 'High_Change', 'Low_Change', 'Volume_Change'
    # ],
    # "기본 특성(1차 차분) + 기술 특성": [
    #     'Close_Diff', 'Open_Diff', 'High_Diff', 'Low_Diff', 'Volume_Diff',
    #     'SMA_5', 'SMA_20', 'SMA_60', 'RSI', 'MACD', 'HL_Ratio'
    # ],
    # "기본 특성(2차 차분) + 기술 특성": [
    #     'Close_Diff2', 'Open_Diff2', 'High_Diff2', 'Low_Diff2', 'Volume_Diff2',
    #     'SMA_5', 'SMA_20', 'SMA_60', 'RSI', 'MACD', 'HL_Ratio'
    # ],
    # "기본 특성(변화율) + 기술 특성": [
    #     'Close_Change', 'Open_Change', 'High_Change', 'Low_Change', 'Volume_Change',
    #     'SMA_5', 'SMA_20', 'SMA_60', 'RSI', 'MACD', 'HL_Ratio'
    # ]
}

# 데이터 가져오기 함수
def fetch_stock_data(ticker, start=None, end=None, **kwargs):
    """
    Yahoo Finance API를 통해 주가 데이터를 가져옵니다.
    직접 데이터 파일을 생성하고 경로를 반환합니다.
    """
    if start is None:
        start = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    
    ti = kwargs['ti']
    
    # 데이터 디렉토리 생성
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # 파일 경로 설정
    file_path = f"{data_dir}/{ticker}_{start.replace('-', '')}_{end.replace('-', '')}.csv"
    
    print(f"{ticker} stock data from {start} to {end}...")
    
    # 데이터 가져오기
    stock = yf.Ticker(ticker)
    data = stock.history(start=start, end=end)
    
    # 빈 데이터 체크
    if data.empty:
        raise ValueError("Failed to get valid data from API")
    
    # 중요: 컬럼 레벨 처리 (원본 코드와 동일하게)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # 저장
    data.to_csv(file_path)
    print(f"Data saved: {file_path} ({len(data)} rows)")
    
    
    # XCom을 통해 파일 경로 전달
    ti.xcom_push(key='stock_data_file', value=file_path)
    
    return True

# 기술적 지표 추가 함수
def add_technical_indicators(**kwargs):
    """
    다양한 기술적 지표를 데이터에 추가합니다.
    """
    ti = kwargs['ti']
    data_file = ti.xcom_pull(key='stock_data_file', task_ids='fetch_data')
    
    print(f"Loading data from '{data_file}'...")
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    print("Adding technical indicators...")
    
    # SMA 추가
    for period in [5, 20, 60]:
        data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
    
    # MACD 추가
    fast, slow, signal = 12, 26, 9
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    data['MACD'] = ema_fast - ema_slow
    data['Signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
    data['Histogram'] = data['MACD'] - data['Signal']
    
    # RSI 추가
    window = 14
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Diff 추가
    data['Close_Diff'] = data['Close'].diff()
    data['Open_Diff'] = data['Open'].diff()
    data['Volume_Diff'] = data['Volume'].diff()
    data['Low_Diff'] = data['Low'].diff()
    data['High_Diff'] = data['High'].diff()
    
    # Diff2 추가
    data['Close_Diff2'] = data['Close_Diff'].diff()
    data['Open_Diff2'] = data['Open_Diff'].diff()
    data['Volume_Diff2'] = data['Volume_Diff'].diff()
    data['Low_Diff2'] = data['Low_Diff'].diff()
    data['High_Diff2'] = data['High_Diff'].diff()
    
    # 변화율 추가
    data['Close_Change'] = data['Close'].pct_change() * 100
    data['Open_Change'] = data['Open'].pct_change() * 100
    data['High_Change'] = data['High'].pct_change() * 100
    data['Low_Change'] = data['Low'].pct_change() * 100
    data['Volume_Change'] = data['Volume'].pct_change() * 100
    
    # HL_Ratio 추가
    data['HL_Ratio'] = (data['High'] - data['Low']) / data['Close'] * 100
    
    # 결측치 제거
    data.dropna(inplace=True)
    
    print(f"Technical indicators added: {len(data)} rows")
    
    # 처리된 데이터 저장
    processed_file = data_file.replace('.csv', '_processed.csv')
    data.to_csv(processed_file)
    print(f"Processed data saved: {processed_file}")
    
    # XCom을 통해 파일 경로 전달
    ti.xcom_push(key='processed_data_file', value=processed_file)
    
    return True

# 데이터 시각화 함수
def visualize_data(ticker, **kwargs):
    """
    주가 데이터와 지표를 시각화합니다.
    """
    ti = kwargs['ti']
    processed_file = ti.xcom_pull(key='processed_data_file', task_ids='add_indicators')
    
    print(f"Loading data from '{processed_file}'...")
    data = pd.read_csv(processed_file, index_col=0, parse_dates=True)
    
    # 결과 디렉토리 생성
    results_dir = f"results/{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("Visualizing data...")
    
    # 주가 및 이동평균 차트
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Close')
    plt.plot(data.index, data['SMA_20'], label='SMA_20')
    plt.plot(data.index, data['SMA_60'], label='SMA_60')
    plt.title(f'{ticker} Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/price_and_sma.png")
    plt.close()
    
    # MACD 차트
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['MACD'], label='MACD')
    plt.plot(data.index, data['Signal'], label='Signal Line')
    plt.bar(data.index, data['Histogram'], label='Histogram')
    plt.title(f'{ticker} MACD Indicator')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/macd.png")
    plt.close()
    
    # RSI 차트
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['RSI'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    plt.title(f'{ticker} RSI Indicator')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/rsi.png")
    plt.close()
    
    # 가격 차분 차트
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close_Diff'], label='Close_Diff')
    plt.plot(data.index, data['Close_Diff2'], label='Close_Diff2')
    plt.title(f'{ticker} Price Differentials')
    plt.xlabel('Date')
    plt.ylabel('Difference')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/price_diff.png")
    plt.close()
    
    # 가격 변화율 차트
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close_Change'], label='Close_Change')
    plt.title(f'{ticker} Price Change Rate (%)')
    plt.xlabel('Date')
    plt.ylabel('Change (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/price_change.png")
    plt.close()
    
    print(f"Data visualization completed: {results_dir}")
    
    # XCom을 통해 결과 디렉토리 전달
    ti.xcom_push(key='visualization_dir', value=results_dir)
    
    return True

# 다변량 시퀀스 생성 함수
def create_multivariate_sequences(data, feature_cols, target_col, seq_length=120):
    """
    다변량 시계열 데이터를 시퀀스 형태로 변환합니다.
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # 다변량 특성 시퀀스
        features_seq = data[feature_cols].iloc[i:i + seq_length].values
        
        # 타겟 값 (다음 종가)
        target = data[target_col].iloc[i + seq_length]
        
        X.append(features_seq)
        y.append(target)
    
    return np.array(X), np.array(y)

# LSTM 모델 생성 함수
def build_lstm_model(seq_length, n_features=1, layers=[50, 50], dropouts=[0.2, 0.2]):
    """
    LSTM 모델을 생성합니다.
    """
    model = Sequential()
    
    # 첫 번째 레이어 (다변량 입력 지원)
    model.add(LSTM(units=layers[0], 
                  return_sequences=(len(layers) > 1),
                  input_shape=(seq_length, n_features)))
    model.add(Dropout(dropouts[0]))
    
    # 중간 레이어
    for i in range(1, len(layers)-1):
        model.add(LSTM(units=layers[i], return_sequences=True))
        model.add(Dropout(dropouts[i]))
    
    # 마지막 레이어
    if len(layers) > 1:
        model.add(LSTM(units=layers[-1], return_sequences=False))
        model.add(Dropout(dropouts[-1]))
    
    # 출력 레이어
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error',
                 metrics=['mean_absolute_error'])
    
    return model

# 모델 저장 및 로드 함수 추가
def save_model(model, scaler_dict, feature_set, results_dir, ticker):
    """
    학습된 모델과 스케일러를 저장합니다.
    
    Parameters:
    model: 학습된 LSTM 모델
    scaler_dict: 데이터 정규화에 사용된 스케일러 사전
    feature_set: 사용된 특성 집합 이름
    results_dir: 결과 저장 디렉토리
    ticker: 주식 티커
    
    Returns:
    str: 모델 저장 경로
    """
    # 모델 디렉토리 생성
    model_dir = f"{results_dir}/models"
    os.makedirs(model_dir, exist_ok=True)
    
    # 안전한 파일명 생성 (특수문자 제거)
    safe_ticker = ticker.replace('^', '').replace('/', '_')
    safe_feature_set = feature_set.replace(' ', '_')
    
    # 모델 저장 경로 (확장자 .keras 추가)
    model_path = f"{model_dir}/{safe_ticker}_{safe_feature_set}_model.keras"
    
    # 모델 저장
    model.save(model_path)
    print(f"모델 저장 완료: {model_path}")
    
    # 스케일러 저장
    scaler_path = f"{model_dir}/{safe_ticker}_{safe_feature_set}_scalers.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_dict, f)
    print(f"스케일러 저장 완료: {scaler_path}")
    
    # 모델 메타데이터 저장
    meta_path = f"{model_dir}/{safe_ticker}_{safe_feature_set}_meta.json"
    meta_data = {
        'ticker': ticker,
        'feature_set': feature_set,
        'features': list(scaler_dict.keys()),
        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': model_path,
        'scaler_path': scaler_path
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)
    print(f"메타데이터 저장 완료: {meta_path}")
    
    return model_path

def load_model(model_path, scaler_path):
    """
    저장된 모델과 스케일러를 로드합니다.
    
    Parameters:
    model_path: 모델 파일 경로(.keras 확장자 포함)
    scaler_path: 스케일러 파일 경로
    
    Returns:
    tuple: (모델, 스케일러 사전)
    """
    # 파일 존재 확인
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"스케일러 파일이 존재하지 않습니다: {scaler_path}")
    
    # 모델 로드
    model = tf.keras.models.load_model(model_path)
    print(f"모델 로드 완료: {model_path}")
    
    # 스케일러 로드
    with open(scaler_path, 'rb') as f:
        scaler_dict = pickle.load(f)
    print(f"스케일러 로드 완료: {scaler_path}")
    
    return model, scaler_dict

# 모델 목록 조회 및 관리 함수
def list_saved_models(**kwargs):
    """
    저장된 모델 목록을 조회합니다.
    """
    models_dir = "results"
    model_files = []
    
    try:
        for root, dirs, files in os.walk(models_dir):
            for dir_name in dirs:
                if dir_name == "models":
                    model_dir = os.path.join(root, dir_name)
                    for model_root, _, model_files in os.walk(model_dir):
                        meta_files = [f for f in model_files if f.endswith("_meta.json")]
                        for meta_file in meta_files:
                            try:
                                meta_path = os.path.join(model_root, meta_file)
                                with open(meta_path, 'r', encoding='utf-8') as f:
                                    meta_data = json.load(f)
                                    model_files.append({
                                        'path': os.path.join(model_root, meta_file.replace("_meta.json", "_model")),
                                        'ticker': meta_data.get('ticker', 'Unknown'),
                                        'feature_set': meta_data.get('feature_set', 'Unknown'),
                                        'created_date': meta_data.get('created_date', 'Unknown')
                                    })
                            except Exception as e:
                                print(f"메타 파일 처리 중 오류: {meta_path}, 오류: {str(e)}")
    except Exception as e:
        print(f"모델 목록 로드 중 오류: {str(e)}")
    
    # 모델 정보 출력
    if model_files:
        print("\n=== 저장된 모델 목록 ===")
        for i, model_info in enumerate(model_files, 1):
            # 타입 검증 추가
            if isinstance(model_info, dict):
                print(f"{i}. 티커: {model_info.get('ticker', 'Unknown')}, 특성 집합: {model_info.get('feature_set', 'Unknown')}, 생성일: {model_info.get('created_date', 'Unknown')}")
            else:
                print(f"{i}. 모델 정보 형식 오류: {type(model_info)}, 내용: {model_info}")
    else:
        print("저장된 모델이 없습니다.")
    
    # XCom을 통해 모델 목록 전달
    kwargs['ti'].xcom_push(key='model_list', value=model_files)
    
    return True

# 모델 학습 및 평가 함수
def train_and_evaluate_model(ticker, **kwargs):
    """
    LSTM 모델을 학습하고 특성 조합별 성능을 평가합니다.
    """
    ti = kwargs['ti']
    processed_file = ti.xcom_pull(key='processed_data_file', task_ids='add_indicators')
    
    print(f"Loading data from '{processed_file}'...")
    data = pd.read_csv(processed_file, index_col=0, parse_dates=True)
    
    # 결과 디렉토리 가져오기
    results_dir = ti.xcom_pull(key='visualization_dir', task_ids='visualize_data')
    if not results_dir:
        results_dir = f"results/{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(results_dir, exist_ok=True)
    
    print("Training and evaluating model...")
    
    target_col = 'Close'
    seq_length = 120
    epochs = 10
    results = {}
    
    # 각 특성 조합 평가
    for name, feature_set in feature_sets.items():
        # 유효한 특성만 선택
        valid_features = [f for f in feature_set if f in data.columns]
        
        print(f"\nEvaluating {name}...")
        print(f"Features: {valid_features}")
        
        # 5번 반복 평가 (원본 코드와 동일)
        mapes = []
        for i in range(3):
            try:
                # 학습/테스트 데이터 분할 - 원본 코드와 동일하게
                end_idx = len(data) - 30  # 마지막 30일은 테스트용
                start_idx = end_idx - 300 - seq_length  # 300일 학습 + 시퀀스 길이
                
                if start_idx < 0:
                    start_idx = 0
                
                train_data = data.iloc[start_idx:end_idx]
                test_data = data.iloc[end_idx:min(end_idx+30, len(data))]
                
                # 데이터 정규화
                scaler_dict = {}
                scaled_train = pd.DataFrame(index=train_data.index)
                scaled_test = pd.DataFrame(index=test_data.index)
                
                for col in valid_features + [target_col]:
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_train[col] = scaler.fit_transform(train_data[[col]])
                    scaled_test[col] = scaler.transform(test_data[[col]])
                    scaler_dict[col] = scaler
                
                # 시퀀스 생성 - 원본 함수 사용
                X_train, y_train = create_multivariate_sequences(
                    scaled_train, valid_features, target_col, seq_length
                )
                
                # 모델 생성 및 학습 - 원본 함수 사용
                model = build_lstm_model(seq_length, n_features=len(valid_features))
                model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    verbose=0
                )
                
                # 테스트 예측
                test_seq = scaled_train[valid_features].iloc[-seq_length:].values
                test_seq = test_seq.reshape(1, seq_length, len(valid_features))
                
                preds = []
                actual = scaled_test[target_col].values
                
                for j in range(len(actual)):
                    pred = model.predict(test_seq, verbose=0)[0, 0]
                    preds.append(pred)
                    
                    if j < len(actual) - 1:
                        test_seq = np.roll(test_seq, -1, axis=1)
                        next_features = scaled_test[valid_features].iloc[j+1].values
                        test_seq[0, -1, :] = next_features
                
                # MAPE 계산
                y_true = scaler_dict[target_col].inverse_transform(actual.reshape(-1, 1))
                y_pred = scaler_dict[target_col].inverse_transform(np.array(preds).reshape(-1, 1))
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                mapes.append(mape)
                
                print(f"  Run {i+1}/5: MAPE = {mape:.2f}%")
                
            except Exception as e:
                print(f"  Error in run {i+1}: {str(e)}")
                continue
        
        # 평균 MAPE 계산
        if mapes:
            avg_mape = np.mean(mapes)
            results[name] = avg_mape
            print(f"{name} average MAPE: {avg_mape:.2f}%")
        else:
            print(f"{name} evaluation failed!")
    
    # 결과 요약 및 시각화
    print("\n=== 특성 조합별 성능 비교 ===")
    for name, mape in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name}: MAPE {mape:.2f}%")

    # 결과 데이터프레임 생성 - 성능 좋은 순(MAPE 낮은 순)으로 정렬
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    result_df = pd.DataFrame({
        'MAPE': [mape for _, mape in sorted_results]
    }, index=[name for name, _ in sorted_results])

    # 가로 막대 차트로 시각화
    plt.figure(figsize=(10, 6))
    plt.barh(result_df.index, result_df['MAPE'])
    plt.xlabel('MAPE (%)')
    plt.ylabel('특성 조합')
    plt.title('특성 조합별 성능 비교')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/feature_evaluation.png")
    plt.close()

    # 최고 성능 특성 조합 선택 (MAPE 값이 가장 낮은 것)
    #best_feature_set = sorted_results[0][0]
    best_feature_set = '기본 특성'
    print(f"\n최고 성능 특성 조합: {best_feature_set}")

    # 결과 저장 - 정렬된 결과와 UTF-8 인코딩 사용
    with open(f"{results_dir}/evaluation_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            'results': {k: float(v) for k, v in sorted_results},
            'best_feature_set': best_feature_set
        }, f, indent=4, ensure_ascii=False)
    
    print(f"Model evaluation results saved: {results_dir}/evaluation_results.json")
    
    print(f"\n최고 성능 특성 조합 '{best_feature_set}'으로 최종 모델 훈련 중...")
    
    # 유효한 특성 추출
    valid_features = [f for f in feature_sets[best_feature_set] if f in data.columns]
    
    # 전체 데이터를 학습에 사용
    scaler_dict = {}
    scaled_data = pd.DataFrame(index=data.index)
    
    for col in valid_features + [target_col]:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[col] = scaler.fit_transform(data[[col]])
        scaler_dict[col] = scaler
    
    # 시퀀스 생성
    X_all, y_all = create_multivariate_sequences(
        scaled_data, valid_features, target_col, seq_length
    )
    
    # 최종 모델 훈련
    final_model = build_lstm_model(seq_length, n_features=len(valid_features))
    final_model.fit(X_all, y_all, epochs=epochs, batch_size=32, verbose=0)
    print("최종 모델 훈련 완료!")
    
    # 모델 저장
    model_path = save_model(final_model, scaler_dict, best_feature_set, results_dir, ticker)
    
    # XCom을 통해 평가 결과 전달
    ti.xcom_push(key='evaluation_results', value=results)
    ti.xcom_push(key='best_feature_set', value=best_feature_set)
    ti.xcom_push(key='model_path', value=model_path)
    
    return True

# 예측 결과 시각화 함수
def visualize_predictions(ticker, **kwargs):
    """
    최적 모델의 예측 결과를 시각화합니다.
    """
    ti = kwargs['ti']
    processed_file = ti.xcom_pull(key='processed_data_file', task_ids='add_indicators')
    best_feature_set = ti.xcom_pull(key='best_feature_set', task_ids='train_model')
    results_dir = ti.xcom_pull(key='visualization_dir', task_ids='visualize_data')
    model_path = ti.xcom_pull(key='model_path', task_ids='train_model')
    
    print(f"Loading data from '{processed_file}'...")
    data = pd.read_csv(processed_file, index_col=0, parse_dates=True)
    
    print(f"Visualizing predictions with best feature set: '{best_feature_set}'")
    
    # 모델 경로 생성
    model_dir = f"{results_dir}/models"
    safe_ticker = ticker.replace('^', '').replace('/', '_')
    safe_feature_set = best_feature_set.replace(' ', '_')
    model_path = f"{model_dir}/{safe_ticker}_{safe_feature_set}_model.keras"
    scaler_path = f"{model_dir}/{safe_ticker}_{safe_feature_set}_scalers.pkl"

    # 모델이 있으면 로드, 없으면 새로 훈련
    try:
        model, scaler_dict = load_model(model_path, scaler_path)
        print("저장된 모델을 로드했습니다.")
    except Exception as e:
        print(f"저장된 모델을 로드할 수 없습니다: {str(e)}")
        print("새로운 모델을 훈련합니다.")
        
        # 모델 새로 훈련 (기존 코드와 동일)
        # 선택된 특성 집합 가져오기
        valid_features = [f for f in feature_sets[best_feature_set] if f in data.columns]
        target_col = 'Close'
        seq_length = 120
        
        # 학습/테스트 데이터 분할
        end_idx = len(data) - 30  # 마지막 30일은 테스트용
        start_idx = end_idx - 300 - seq_length  # 300일 학습 + 시퀀스 길이
        
        if start_idx < 0:
            start_idx = 0
        
        train_data = data.iloc[start_idx:end_idx]
        test_data = data.iloc[end_idx:min(end_idx+30, len(data))]
        
        # 데이터 정규화
        scaler_dict = {}
        scaled_train = pd.DataFrame(index=train_data.index)
        scaled_test = pd.DataFrame(index=test_data.index)
        
        for col in valid_features + [target_col]:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_train[col] = scaler.fit_transform(train_data[[col]])
            scaled_test[col] = scaler.transform(test_data[[col]])
            scaler_dict[col] = scaler
        
        # 시퀀스 생성
        X_train, y_train = create_multivariate_sequences(
            scaled_train, valid_features, target_col, seq_length
        )
        
        # 모델 생성 및 학습
        model = build_lstm_model(seq_length, n_features=len(valid_features))
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # 여기서부터는 모델과 스케일러를 사용한 예측 코드
    # 선택된 특성 집합 가져오기
    valid_features = [f for f in feature_sets[best_feature_set] if f in data.columns]
    target_col = 'Close'
    seq_length = 120
    
    # 테스트 데이터 분할
    end_idx = len(data) - 30
    test_data = data.iloc[end_idx:min(end_idx+30, len(data))]
    
    # 예측을 위한 시퀀스 준비
    input_sequence = data.iloc[end_idx-seq_length:end_idx][valid_features]
    input_scaled = np.zeros((1, seq_length, len(valid_features)))
    
    # 입력 시퀀스 스케일링
    for i, col in enumerate(valid_features):
        input_scaled[0, :, i] = scaler_dict[col].transform(input_sequence[[col]]).flatten()
    
    # 예측 수행
    preds = []
    actual = test_data[target_col].values
    current_seq = input_scaled.copy()
    
    for i in range(len(actual)):
        # 예측
        pred = model.predict(current_seq, verbose=0)[0, 0]
        preds.append(pred)
        
        # 다음 시퀀스 준비 (예측이 더 필요한 경우)
        if i < len(actual) - 1:
            # 새로운 데이터 얻기
            new_point = np.zeros(len(valid_features))
            for j, feat in enumerate(valid_features):
                if feat == target_col:
                    new_point[j] = pred
                else:
                    # 실제 다른 특성값 사용
                    new_point[j] = scaler_dict[feat].transform(test_data.iloc[i:i+1][[feat]])[0, 0]
            
            # 시퀀스 업데이트
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, :] = new_point
    
    # 원래 스케일로 변환
    y_true = actual
    y_pred = scaler_dict[target_col].inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    
    # MAPE, MAE, RMSE 계산
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"Model performance:")
    print(f"MAPE: {mape:.2f}%")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    # 예측 결과 시각화
    plt.figure(figsize=(14, 7))
    plt.plot(test_data.index, y_true, label='Actual Price')
    plt.plot(test_data.index, y_pred, label='Predicted Price', linestyle='--')
    plt.title(f'{ticker} Stock Price Prediction (MAPE: {mape:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/prediction_results.png")
    plt.close()
    
    # 예측 결과 저장
    pred_df = pd.DataFrame({
    'Actual': y_true,
    'Predicted': y_pred,
    'Error': y_true - y_pred,
    'Error_Pct': (y_true - y_pred) / y_true * 100
}, index=test_data.index)

    # 파일로 저장할 때 인덱스 포함
    pred_df.to_csv(f"{results_dir}/prediction_results.csv", date_format='%Y-%m-%d')  # date_format 지정
    
    print(f"Prediction visualization and results saved: {results_dir}/prediction_results.png")
    
    return True

# DAG 정의
with DAG(
    'test',
    default_args={
        'owner': 'airflow',
        'depends_on_past': False,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='Stock data collection, technical indicator addition, and LSTM prediction',
    schedule_interval='@daily',  # 매일 실행
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['stock', 'prediction', 'lstm'],
) as dag:
    
    # 티커 및 날짜 설정
    ticker = '^IXIC'  # NASDAQ 지수
    start_date = '2010-01-01'
    end_date = '2020-01-01'
    
    # start_date = '1971-02-05'
    # end_date = '2025-05-01'
    
    # 태스크 정의
    fetch_data_task = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_stock_data,
        op_kwargs={'ticker': ticker, 'start': start_date, 'end': end_date},
    )
    
    add_indicators_task = PythonOperator(
        task_id='add_indicators',
        python_callable=add_technical_indicators,
    )
    
    visualize_data_task = PythonOperator(
        task_id='visualize_data',
        python_callable=visualize_data,
        op_kwargs={'ticker': ticker},
    )
    
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_and_evaluate_model,
        op_kwargs={'ticker': ticker},
    )
    
    visualize_predictions_task = PythonOperator(
        task_id='visualize_predictions',
        python_callable=visualize_predictions,
        op_kwargs={'ticker': ticker},
    )

    list_models_task = PythonOperator(
        task_id='list_models',
        python_callable=list_saved_models,
    )

    # 태스크 의존성 설정 수정
    fetch_data_task >> add_indicators_task >> visualize_data_task >> train_model_task >> visualize_predictions_task >> list_models_task