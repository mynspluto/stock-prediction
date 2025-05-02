import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import os
import pickle
from datetime import datetime, timedelta

# 1. 데이터 가져오기 함수
def fetchData(ticker, start=None, end=None):
    """
    Yahoo Finance API를 통해 주가 데이터를 가져옵니다.
    
    Parameters:
    ticker (str): 주식 티커 (예: 'AAPL', 'QQQ', '^IXIC')
    start (str, optional): 시작 날짜 (예: '2010-01-01')
    end (str, optional): 종료 날짜 (예: '2023-01-01')
    
    Returns:
    pd.DataFrame: 가져온 주가 데이터
    """
    if start is None:
        start = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    
    data = yf.download(ticker, start=start, end=end)
    return data

# 2. 기술적 지표 추가 함수
def addSMA(data, periods=[5, 20, 60]):
    """
    단순 이동평균(SMA) 지표를 추가합니다.
    
    Parameters:
    data (pd.DataFrame): 주가 데이터
    periods (list): SMA 기간 리스트
    
    Returns:
    pd.DataFrame: SMA 지표가 추가된 데이터
    """
    for period in periods:
        data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
    return data

def addMACD(data, fast=12, slow=26, signal=9):
    """
    MACD(Moving Average Convergence Divergence) 지표를 추가합니다.
    
    Parameters:
    data (pd.DataFrame): 주가 데이터
    fast (int): 단기 EMA 기간
    slow (int): 장기 EMA 기간
    signal (int): 신호선 기간
    
    Returns:
    pd.DataFrame: MACD 지표가 추가된 데이터
    """
    # EMA 계산
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    
    # MACD 라인 계산
    data['MACD'] = ema_fast - ema_slow
    
    # 신호선 계산
    data['Signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
    
    # MACD 히스토그램 계산
    data['Histogram'] = data['MACD'] - data['Signal']
    
    return data

def plotChart(data, columns=['Close', 'SMA_20', 'SMA_60'], title=None):
    """
    주가 데이터와 지표를 시각화합니다.
    
    Parameters:
    data (pd.DataFrame): 주가 데이터
    columns (list): 시각화할 열 이름 리스트
    title (str, optional): 차트 제목
    """
    plt.figure(figsize=(14, 7))
    for col in columns:
        if col in data.columns:
            plt.plot(data.index, data[col], label=col)
    
    plt.title(title or 'Stock Price and Indicators')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # MACD 차트 (MACD가 있는 경우)
    if 'MACD' in data.columns and 'Signal' in data.columns:
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data['MACD'], label='MACD')
        plt.plot(data.index, data['Signal'], label='Signal Line')
        plt.bar(data.index, data['Histogram'], label='Histogram')
        plt.title('MACD')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

# 3. 데이터 저장 및 로드 함수
def saveData(data, filename):
    """
    전처리된 데이터를 파일로 저장합니다.
    
    Parameters:
    data (pd.DataFrame): 저장할 데이터
    filename (str): 파일 이름 (확장자 제외)
    """
    if not os.path.exists('data'):
        os.makedirs('data')
    
    data.to_csv(f'data/{filename}.csv')
    print(f"데이터 저장 완료: data/{filename}.csv")

def loadData(filename):
    """
    저장된 데이터를 로드합니다.
    
    Parameters:
    filename (str): 파일 이름 (확장자 제외)
    
    Returns:
    pd.DataFrame: 로드된 데이터
    """
    data = pd.read_csv(f'data/{filename}.csv', index_col=0, parse_dates=True)
    print(f"데이터 로드 완료: data/{filename}.csv")
    return data

# 4. LSTM 모델 생성 함수
def build_lstm_model(seq_length, layers=[50, 50], dropouts=[0.2, 0.2]):
    """
    LSTM 모델을 생성합니다.
    
    Parameters:
    seq_length (int): 시퀀스 길이
    layers (list): LSTM 레이어별 유닛 수
    dropouts (list): LSTM 레이어별 드롭아웃 비율
    
    Returns:
    tensorflow.keras.models.Sequential: LSTM 모델
    """
    model = Sequential()
    
    # 첫 번째 레이어
    model.add(LSTM(units=layers[0], 
                  return_sequences=(len(layers) > 1),
                  input_shape=(seq_length, 1)))
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

# 5. 워크포워드 검증 함수 (수정된 버전)
def walk_forward_validation(data, target_col='Close', seq_length=60, 
                           train_days=252*3, pred_days=21, step_days=21,
                           start_idx=None, end_idx=None, epochs=20, batch_size=32):
    """
    워크포워드 방식으로 주가 예측을 수행합니다.
    
    Parameters:
    data (pd.DataFrame): 전체 시계열 데이터
    target_col (str): 대상 열 이름
    seq_length (int): 시퀀스 길이
    train_days (int): 학습에 사용할 일수
    pred_days (int): 예측할 일수
    step_days (int): 워크포워드 단계 이동 간격
    start_idx (int, optional): 시작 인덱스
    end_idx (int, optional): 종료 인덱스
    epochs (int): 학습 에포크 수
    batch_size (int): 배치 크기
    
    Returns:
    tuple: (모델 리스트, 예측 결과, 실제 값, 평가 결과)
    """
    # 데이터 정규화
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[[target_col]])
    
    # 인덱스 설정
    if start_idx is None:
        start_idx = seq_length + train_days  # 최소한 시퀀스 길이 + 학습 기간이 필요
    if end_idx is None:
        end_idx = len(data) - pred_days
    
    # 결과 저장용 리스트
    models = []
    predictions = []
    actuals = []
    evaluation = []
    
    # 워크포워드 반복
    for i in range(start_idx, end_idx, step_days):
        # 학습 구간 설정
        train_start = max(0, i - train_days)
        train_end = i
        
        # 예측 구간 설정
        test_start = train_end
        test_end = min(test_start + pred_days, len(data))
        
        if test_end - test_start < 1:
            continue  # 예측 구간이 너무 짧은 경우 건너뜀
        
        print(f"\n워크포워드 반복 - 학습: {data.index[train_start].strftime('%Y-%m-%d')} ~ {data.index[train_end-1].strftime('%Y-%m-%d')}")
        print(f"예측: {data.index[test_start].strftime('%Y-%m-%d')} ~ {data.index[min(test_end-1, len(data)-1)].strftime('%Y-%m-%d')}")
        
        # 학습 데이터 준비
        train_data = scaled_data[train_start:train_end]
        
        # 시퀀스 생성
        X_train, y_train = [], []
        for j in range(len(train_data) - seq_length):
            X_train.append(train_data[j:j + seq_length])
            y_train.append(train_data[j + seq_length])
        
        # 충분한 데이터가 없는 경우 건너뜀
        if len(X_train) < 10:  # 최소 10개의 시퀀스가 필요하다고 가정
            print(f"경고: 충분한 학습 데이터가 없습니다. 시퀀스 수: {len(X_train)}")
            continue
        
        # 배열로 변환
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # 테스트 데이터 생성 (마지막 시퀀스 + 실제 타겟값)
        X_test = scaled_data[train_end - seq_length:train_end]
        X_test = X_test.reshape(1, seq_length, 1)
        y_test = scaled_data[test_start:test_end]
        
        # 모델 생성 및 학습
        model = build_lstm_model(seq_length)
        
        # 학습 데이터 형태 조정
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        # 학습
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=min(batch_size, len(X_train)),  # 배치 크기가 데이터보다 크지 않도록
            verbose=1
        )
        
        # 예측 수행
        test_predictions = []
        current_batch = X_test.copy()
        
        for j in range(len(y_test)):
            # 다음 값 예측
            current_pred = model.predict(current_batch, verbose=0)[0]
            test_predictions.append(current_pred[0])
            
            # 입력 시퀀스 업데이트 (가장 오래된 값 제거하고 예측값 추가)
            if j < len(y_test) - 1:  # 마지막 예측이 아니면 시퀀스 업데이트
                current_batch = np.roll(current_batch, -1, axis=1)
                current_batch[0, -1, 0] = current_pred[0]
        
        # 결과 저장
        models.append(model)
        predictions.append(np.array(test_predictions).reshape(-1, 1))
        actuals.append(y_test)
        
        # 평가
        mse = mean_squared_error(y_test, test_predictions)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, test_predictions)
        
        evaluation.append({
            'train_start': data.index[train_start],
            'train_end': data.index[train_end-1],
            'test_start': data.index[test_start],
            'test_end': data.index[min(test_end-1, len(data)-1)],
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        })
        
        print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")
    
    # 예측 결과가 없는 경우
    if not predictions:
        print("경고: 유효한 예측 결과가 없습니다. 파라미터를 조정해 보세요.")
        return [], [], [], [], scaler
    
    # 전체 예측 결과 시각화
    plt.figure(figsize=(15, 7))
    
    # 원래 스케일로 복원
    all_pred = []
    all_actual = []
    pred_dates = []
    
    for i, (pred, actual, eval_info) in enumerate(zip(predictions, actuals, evaluation)):
        pred_inv = scaler.inverse_transform(pred)
        actual_inv = scaler.inverse_transform(actual)
        
        all_pred.extend(pred_inv.flatten())
        all_actual.extend(actual_inv.flatten())
        
        test_dates = pd.date_range(start=eval_info['test_start'], 
                                end=eval_info['test_end'], 
                                freq='B')[:len(pred_inv)]
        pred_dates.extend(test_dates)
        
        plt.plot(test_dates, pred_inv, 'r-', alpha=0.7)
        plt.plot(test_dates, actual_inv, 'b-', alpha=0.7)
    
    # 원래 데이터 전체 표시
    plt.plot(data.index, data[target_col], 'k-', alpha=0.3, label='Original Data')
    plt.plot([], [], 'r-', label='Predictions')
    plt.plot([], [], 'b-', label='Actual')
    
    plt.title('Walk-Forward Validation Results')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 평가 지표 평균 계산
    if evaluation:
        avg_metrics = {
            'mse': np.mean([eval_info['mse'] for eval_info in evaluation]),
            'rmse': np.mean([eval_info['rmse'] for eval_info in evaluation]),
            'mae': np.mean([eval_info['mae'] for eval_info in evaluation])
        }
        print("\n평균 평가 지표:")
        for metric, value in avg_metrics.items():
            print(f"{metric.upper()}: {value:.6f}")
    
    return models, predictions, actuals, evaluation, scaler

# 6. 예측 결과 시각화 함수
def plot_prediction_results(data, target_col, predictions, actuals, evaluation, scaler):
    """
    워크포워드 방식의 예측 결과를 개별적으로 시각화합니다.
    
    Parameters:
    data (pd.DataFrame): 원본 데이터
    target_col (str): 대상 열 이름
    predictions (list): 예측 결과 리스트
    actuals (list): 실제 값 리스트
    evaluation (list): 평가 정보 리스트
    scaler (sklearn.preprocessing.MinMaxScaler): 정규화에 사용한 스케일러
    """
    if not predictions:
        print("시각화할 예측 결과가 없습니다.")
        return
        
    # 각 예측 기간별 결과 시각화
    for i, (pred, actual, eval_info) in enumerate(zip(predictions, actuals, evaluation)):
        # 원래 스케일로 복원
        pred_inv = scaler.inverse_transform(pred)
        actual_inv = scaler.inverse_transform(actual)
        
        # 날짜 범위 생성
        test_dates = pd.date_range(start=eval_info['test_start'], 
                                  end=eval_info['test_end'], 
                                  freq='B')[:len(pred_inv)]
        
        # 시각화
        plt.figure(figsize=(12, 6))
        
        # 학습 데이터 표시
        train_dates = pd.date_range(start=eval_info['train_start'], 
                                   end=eval_info['train_end'], 
                                   freq='B')
        train_data = data.loc[eval_info['train_start']:eval_info['train_end'], target_col]
        plt.plot(train_data.index, train_data, 'k-', alpha=0.3, label='Training Data')
        
        # 테스트 기간의 원본 데이터
        test_data = data.loc[eval_info['test_start']:eval_info['test_end'], target_col][:len(pred_inv)]
        
        # 예측 및 실제 값 표시
        plt.plot(test_dates, pred_inv, 'r-', label='Predicted')
        plt.plot(test_dates, actual_inv, 'b-', label='Actual')
        
        # 전체 기간 표시를 위한 날짜 범위 설정
        try:
            date_range = pd.date_range(start=eval_info['train_start'] - timedelta(days=10), 
                                    end=eval_info['test_end'] + timedelta(days=10), 
                                    freq='B')
            data_range = data.loc[date_range[0]:date_range[-1], target_col]
        except:
            pass  # 날짜 범위 오류 시 무시
        
        plt.title(f'Prediction Period {i+1}: {eval_info["test_start"].strftime("%Y-%m-%d")} to {eval_info["test_end"].strftime("%Y-%m-%d")}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # 평가 지표 표시
        plt.figtext(0.15, 0.15, f'MSE: {eval_info["mse"]:.6f}\nRMSE: {eval_info["rmse"]:.6f}\nMAE: {eval_info["mae"]:.6f}', 
                  bbox=dict(facecolor='white', alpha=0.8))
        
        plt.show()

# 7. 미래 예측 함수
def predict_future(model, last_sequence, days=30, scaler=None):
    """
    학습된 모델을 사용하여 미래 가격을 예측합니다.
    
    Parameters:
    model (tensorflow.keras.models.Sequential): 학습된 모델
    last_sequence (numpy.ndarray): 마지막 시퀀스 데이터
    days (int): 예측할 날짜 수
    scaler (sklearn.preprocessing.MinMaxScaler): 정규화에 사용한 스케일러
    
    Returns:
    numpy.ndarray: 예측 결과
    """
    if model is None:
        print("모델이 없습니다.")
        return None
        
    # 예측 결과 저장용 리스트
    future_predictions = []
    
    # 초기 시퀀스
    current_sequence = last_sequence.reshape(1, len(last_sequence), 1)
    
    # 날짜별 예측
    for _ in range(days):
        # 다음 값 예측
        next_pred = model.predict(current_sequence, verbose=0)[0, 0]
        future_predictions.append(next_pred)
        
        # 시퀀스 업데이트
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred
    
    # 스케일러가 제공된 경우 원래 스케일로 복원
    if scaler:
        future_predictions = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1))
    
    return future_predictions

# 8. 메인 로직
def main():
    # 설정
    ticker = 'QQQ'
    start_date = '2021-01-01'
    end_date = '2024-01-01'
    target_col = 'Close'
    seq_length = 60 # 길이만큼의 과거 데이터를 보고 다음 날의 값 예측
    
    # 학습 및 예측 기간 설정 (일 단위)
    train_days = 120  # 3년 (약 252 거래일/년)
    pred_days = 21  # 1개월 (약 21 거래일/월)
    step_days = 21  # 1개월 단위로 이동
    
    # 1. 데이터 가져오기
    data = fetchData(ticker, start_date, end_date)
    print('데이터 가져오기 완료:\n', data.head())
    
    # 2. 특성 추가
    data = addSMA(data)
    data = addMACD(data)
    print('\n특성 추가 완료:')
    print(data[['Close', 'SMA_5', 'SMA_20', 'SMA_60', 'MACD', 'Signal']].tail())
    
    # 3. 데이터 시각화
    plotChart(data)
    
    # 4. 워크포워드 검증 수행
    # 첫 번째 학습 기간에 최소한 seq_length + 충분한 학습 데이터가 있어야 함
    # 시작 인덱스를 조정하여 충분한 데이터가 있도록 함
    min_start_idx = seq_length + 500  # 충분한 학습 데이터 확보
    
    models, predictions, actuals, evaluation, scaler = walk_forward_validation(
        data, target_col, seq_length, train_days, pred_days, step_days,
        start_idx=min_start_idx, epochs=20, batch_size=32
    )
    
    # 5. 예측 결과 개별 시각화
    plot_prediction_results(data, target_col, predictions, actuals, evaluation, scaler)
    
    # 6. 최신 모델로 미래 예측
    if models:
        latest_model = models[-1]
        last_sequence = data[target_col].values[-seq_length:].reshape(-1, 1)
        last_sequence = scaler.transform(last_sequence)
        
        future_days = 30
        future_predictions = predict_future(
            latest_model, last_sequence.flatten(), days=future_days, scaler=scaler)
        
        if future_predictions is not None:
            # 미래 날짜 생성
            last_date = data.index[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=future_days,
                freq='B'  # 영업일 기준
            )
            
            # 미래 예측 시각화
            plt.figure(figsize=(14, 7))
            plt.plot(data[target_col][-90:].index, data[target_col][-90:], 'b-', label='Historical Data')
            plt.plot(future_dates, future_predictions, 'r--', label='Future Predictions')
            plt.title(f'{ticker} Future Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            print("\n미래 예측 결과:")
            for date, price in zip(future_dates[:10], future_predictions[:10]):
                print(f"{date.strftime('%Y-%m-%d')}: {price[0]:.2f}")
    else:
        print("미래 예측을 위한 모델이 없습니다.")

# 메인 함수 실행
if __name__ == "__main__":
    main()