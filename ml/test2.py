import pandas as pd
import numpy as np
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

def addRSI(data, window=14, column='Close'):
    """
    상대강도지수(RSI)를 계산합니다.
    
    Parameters:
    data (pd.DataFrame): 주가 데이터
    window (int): RSI 계산 기간
    column (str): RSI를 계산할 열 이름
    
    Returns:
    pd.DataFrame: RSI 지표가 추가된 데이터
    """
    # 가격 변화 계산
    delta = data[column].diff()
    
    # 상승/하락 분리
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 평균 상승/하락 계산
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # RS 및 RSI 계산
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

def addDiff(data):
    data['Close_diff'] = data['Close'].diff()
    data['Open_diff'] = data['Open'].diff()
    data['Volume_diff'] = data['Volume'].diff()
    data['Low_diff'] = data['Low'].diff()
    data['High_diff'] = data['High'].diff()
    
    return data

def addDiff2(data):
    data['Close_diff2'] = data['Close_diff'].diff()
    data['Open_diff2'] = data['Open_diff'].diff()
    data['Volume_diff2'] = data['Volume_diff'].diff()
    data['Low_diff2'] = data['Low_diff'].diff()
    data['High_diff2'] = data['High_diff'].diff()
    
    return data

def addChange(data):
    data['Close_Change'] = data['Close'].pct_change() * 100
    data['Open_Change'] = data['Open'].pct_change() * 100
    data['High_Change'] = data['High'].pct_change() * 100
    data['Low_Change'] = data['Low'].pct_change() * 100
    data['Volume_Change'] = data['Volume'].pct_change() * 100
    
    return data

def plotChart(data, columns=['Close', 'SMA_20', 'SMA_60'], title=None, save_path=None):
    """
    주가 데이터와 지표를 시각화합니다.
    
    Parameters:
    data (pd.DataFrame): 주가 데이터
    columns (list): 시각화할 열 이름 리스트
    title (str, optional): 차트 제목
    save_path (str, optional): 저장 경로
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
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    # else:
    #     plt.show()
    
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
        
        if save_path:
            macd_path = save_path.replace('.png', '_macd.png')
            plt.savefig(macd_path)
            plt.close()
        # else:
        #     plt.show()

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

# 4. 다변량 시퀀스 생성 함수
def create_multivariate_sequences(data, feature_cols, target_col, seq_length=60):
    """
    다변량 시계열 데이터를 시퀀스 형태로 변환합니다.
    
    Parameters:
    data (pd.DataFrame): 시퀀스로 변환할 데이터
    feature_cols (list): 입력 특성 열 이름 리스트
    target_col (str): 대상 열 이름
    seq_length (int): 시퀀스 길이
    
    Returns:
    tuple: (X 시퀀스, y 타겟값)
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

# 5. LSTM 모델 생성 함수
def build_lstm_model(seq_length, n_features=1, layers=[50, 50], dropouts=[0.2, 0.2]):
    """
    LSTM 모델을 생성합니다.
    
    Parameters:
    seq_length (int): 시퀀스 길이
    n_features (int): 입력 특성 수
    layers (list): LSTM 레이어별 유닛 수
    dropouts (list): LSTM 레이어별 드롭아웃 비율
    
    Returns:
    tensorflow.keras.models.Sequential: LSTM 모델
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

# 6. 워크포워드 검증 함수 (다변량 버전)
def walk_forward_validation(data, feature_cols, target_col='Close', seq_length=60, 
                           train_days=252, pred_days=21, step_days=21,
                           start_idx=None, end_idx=None, epochs=20, batch_size=32, 
                           save_path=None):
    """
    워크포워드 방식으로 주가 예측을 수행합니다 (다변량 입력 지원).
    
    Parameters:
    data (pd.DataFrame): 전체 시계열 데이터
    feature_cols (list): 입력 특성 열 이름 리스트
    target_col (str): 대상 열 이름
    seq_length (int): 시퀀스 길이
    train_days (int): 학습에 사용할 일수
    pred_days (int): 예측할 일수
    step_days (int): 워크포워드 단계 이동 간격
    start_idx (int, optional): 시작 인덱스
    end_idx (int, optional): 종료 인덱스
    epochs (int): 학습 에포크 수
    batch_size (int): 배치 크기
    save_path (str, optional): 그래프 저장 경로
    
    Returns:
    tuple: (모델 리스트, 예측 결과, 실제 값, 평가 결과, 스케일러 딕셔너리)
    """
    # 데이터 정규화
    scaler_dict = {}
    scaled_data = pd.DataFrame(index=data.index)
    
    # 각 특성별로 개별 스케일러 사용
    for col in feature_cols + [target_col]:
        if col not in data.columns:
            continue
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(data[[col]])
        scaled_data[col] = scaled_values
        scaler_dict[col] = scaler
    
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
        train_data = scaled_data.iloc[train_start:train_end]
        
        # 다변량 시퀀스 생성
        X_train, y_train = create_multivariate_sequences(train_data, feature_cols, target_col, seq_length)
        
        # 충분한 데이터가 없는 경우 건너뜀
        if len(X_train) < 10:  # 최소 10개의 시퀀스가 필요하다고 가정
            print(f"경고: 충분한 학습 데이터가 없습니다. 시퀀스 수: {len(X_train)}")
            continue
        
        # 모델 생성 및 학습
        model = build_lstm_model(seq_length, n_features=len(feature_cols))
        
        # 학습
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=min(batch_size, len(X_train)),  # 배치 크기가 데이터보다 크지 않도록
            verbose=1
        )
        
        # 예측 수행 (롤링 방식)
        test_sequence = scaled_data.iloc[train_end - seq_length:train_end][feature_cols].values
        test_sequence = test_sequence.reshape(1, seq_length, len(feature_cols))
        
        test_predictions = []
        actual_values = scaled_data.iloc[test_start:test_end][target_col].values
        
        for j in range(len(actual_values)):
            # 다음 값 예측
            current_pred = model.predict(test_sequence, verbose=0)[0, 0]
            test_predictions.append(current_pred)
            
            # 예측 다음 기간의 실제 다변량 데이터가 있는 경우에만 업데이트
            if j < len(actual_values) - 1:
                # 시퀀스를 한 칸씩 밀고 새 값으로 업데이트
                test_sequence = np.roll(test_sequence, -1, axis=1)
                
                # 다음 기간의 실제 다변량 데이터
                next_features = scaled_data.iloc[test_start + j + 1][feature_cols].values
                
                # 마지막 위치에 새 데이터 업데이트
                test_sequence[0, -1, :] = next_features
        
        # 결과 저장
        models.append(model)
        predictions.append(np.array(test_predictions).reshape(-1, 1))
        actuals.append(actual_values.reshape(-1, 1))
        
        # 평가
        mse = mean_squared_error(actual_values, test_predictions)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(actual_values, test_predictions)
        
        # 원래 스케일로 복원한 값으로 MAPE 계산
        y_test_inv = scaler_dict[target_col].inverse_transform(actual_values.reshape(-1, 1))
        y_pred_inv = scaler_dict[target_col].inverse_transform(np.array(test_predictions).reshape(-1, 1))
        mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
        
        evaluation.append({
            'train_start': data.index[train_start],
            'train_end': data.index[train_end-1],
            'test_start': data.index[test_start],
            'test_end': data.index[min(test_end-1, len(data)-1)],
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        })
        
        print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.2f}%")
    
    # 예측 결과가 없는 경우
    if not predictions:
        print("경고: 유효한 예측 결과가 없습니다. 파라미터를 조정해 보세요.")
        return [], [], [], [], scaler_dict
    
    # 전체 예측 결과 시각화
    plt.figure(figsize=(15, 7))
    
    # 원래 스케일로 복원
    all_pred = []
    all_actual = []
    pred_dates = []
    
    for i, (pred, actual, eval_info) in enumerate(zip(predictions, actuals, evaluation)):
        # 원래 스케일로 복원
        pred_inv = scaler_dict[target_col].inverse_transform(pred)
        actual_inv = scaler_dict[target_col].inverse_transform(actual)
        
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
    
    if save_path:
        plt.savefig(f"{save_path}/walkforward_results.png")
        plt.close()
    # else:
    #     plt.show()
    
    # 평가 지표 평균 계산
    if evaluation:
        avg_metrics = {
            'mse': np.mean([eval_info['mse'] for eval_info in evaluation]),
            'rmse': np.mean([eval_info['rmse'] for eval_info in evaluation]),
            'mae': np.mean([eval_info['mae'] for eval_info in evaluation]),
            'mape': np.mean([eval_info['mape'] for eval_info in evaluation])
        }
        print("\n평균 평가 지표:")
        for metric, value in avg_metrics.items():
            if metric == 'mape':
                print(f"{metric.upper()}: {value:.2f}%")
            else:
                print(f"{metric.upper()}: {value:.6f}")
    
    return models, predictions, actuals, evaluation, scaler_dict

# 7. 예측 결과 시각화 함수
def plot_prediction_results(data, target_col, predictions, actuals, evaluation, scaler_dict, save_path=None):
    """
    워크포워드 방식의 예측 결과를 개별적으로 시각화합니다.
    
    Parameters:
    data (pd.DataFrame): 원본 데이터
    target_col (str): 대상 열 이름
    predictions (list): 예측 결과 리스트
    actuals (list): 실제 값 리스트
    evaluation (list): 평가 정보 리스트
    scaler_dict (dict): 스케일러 딕셔너리
    save_path (str, optional): 그래프 저장 경로
    """
    if not predictions:
        print("시각화할 예측 결과가 없습니다.")
        return
        
    # 각 예측 기간별 결과 시각화
    for i, (pred, actual, eval_info) in enumerate(zip(predictions, actuals, evaluation)):
        # 원래 스케일로 복원
        pred_inv = scaler_dict[target_col].inverse_transform(pred)
        actual_inv = scaler_dict[target_col].inverse_transform(actual)
        
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
        
        try:
            # 전체 기간 표시를 위한 날짜 범위 설정
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
        plt.figtext(0.15, 0.15, f'MSE: {eval_info["mse"]:.6f}\nRMSE: {eval_info["rmse"]:.6f}\nMAE: {eval_info["mae"]:.6f}\nMAPE: {eval_info["mape"]:.2f}%', 
                  bbox=dict(facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(f"{save_path}/prediction_{i+1}.png")
            plt.close()
        # else:
        #     plt.show()

# 8. 미래 예측 함수
def predict_future(model, last_data, feature_cols, scaler_dict, days=30, freq='B'):
    """
    학습된 모델을 사용하여 미래 가격을 예측합니다 (다변량 입력 지원).
    
    Parameters:
    model (tensorflow.keras.models.Sequential): 학습된 모델
    last_data (pd.DataFrame): 마지막 시퀀스 데이터
    feature_cols (list): 입력 특성 열 이름 리스트
    scaler_dict (dict): 스케일러 딕셔너리
    days (int): 예측할 날짜 수
    freq (str): 날짜 빈도 ('B'=영업일, 'D'=달력일)
    
    Returns:
    numpy.ndarray: 예측 결과
    """
    if model is None:
        print("모델이 없습니다.")
        return None
    
    # 시퀀스 길이 확인
    seq_length = model.input_shape[1]
    
    # 마지막 시퀀스 준비
    last_sequence = last_data[feature_cols].values[-seq_length:]
    last_sequence = last_sequence.reshape(1, seq_length, len(feature_cols))
    
    # 예측 결과 저장용 리스트
    future_predictions = []
    
    # 마지막 예측에 사용할 시퀀스
    current_sequence = last_sequence.copy()
    
    # 날짜별 예측
    for _ in range(days):
        # 다음 값 예측
        next_pred = model.predict(current_sequence, verbose=0)[0, 0]
        future_predictions.append(next_pred)
        
        # 시퀀스 업데이트
        current_sequence = np.roll(current_sequence, -1, axis=1)
        
        # 마지막 행 업데이트
        # 여기서는 단순화를 위해 예측된 종가만 업데이트하고 나머지 특성은 마지막 값 복사
        new_features = current_sequence[0, -2, :].copy()
        
        # 종가 인덱스 찾기
        close_idx = feature_cols.index('Close') if 'Close' in feature_cols else 0
        new_features[close_idx] = next_pred
        
        # 업데이트된 특성으로 마지막 시퀀스 설정
        current_sequence[0, -1, :] = new_features
    
    # 예측값 역정규화 (종가만)
    future_pred_scaled = scaler_dict['Close'].inverse_transform(
        np.array(future_predictions).reshape(-1, 1))
    
    return future_pred_scaled

def evaluate_feature_sets(data, target_col='Close', seq_length=60, epochs=10):
    """
    여러 특성 조합의 성능을 비교합니다.
    """
    # 기본 특성 세트
    base_features = [
        'Close', 
        'Open', 
        'High', 
        'Low', 
        'Volume'
    ]
    
    # 기술적 지표
    technical_features = [
        'Close_diff',
        'Open_diff', 
        'High_diff',
        'Low_diff',
        'Volume_diff',
        'SMA_5', 
        'SMA_20', 
        'SMA_60', 
        'RSI', 
        'MACD', 
        'HL_Ratio'
    ]
    
    technical_features2 = [
        'Close_diff2',
        'Open_diff2', 
        'High_diff2',
        'Low_diff2',
        'Volume_diff2',
        'SMA_5', 
        'SMA_20', 
        'SMA_60', 
        'RSI', 
        'MACD', 
        'HL_Ratio'
    ]
    
    # 변화 지표
    change_features = [
        'Close_Change',
        'Open_Change', 
        'High_Change',
        'Low_Change',
        'Volume_Change',
        'SMA_5', 
        'SMA_20', 
        'SMA_60', 
        'RSI', 
        'MACD', 
        'HL_Ratio'
    ]
    
    # 특성 조합 정의
    feature_sets = {
        "기본 특성만": base_features,
        "기술적 특성만": technical_features,
        "기술적2 특성만": technical_features2,
        "변화 특성만": change_features,
        #"기술적2 + 계절": change_features + seasonality_features,
    }
    
    results = {}
    
    # 각 특성 조합 평가
    for name, feature_set in feature_sets.items():
        # 해당 특성이 데이터에 있는지 확인
        valid_features = [f for f in feature_set if f in data.columns]
        
        # if len(valid_features) < 2:  # 최소 2개 이상의 유효한 특성 필요
        #     print(f"{name}: 유효한 특성이 부족합니다.")
        #     continue
            
        print(f"\n{name} 평가 중...")
        print(f"사용 특성: {valid_features}")
        
        # 5번 반복 평가
        mapes = []
        for i in range(5):
            # 간소화된 워크포워드 검증
            end_idx = len(data) - 30  # 마지막 30일은 테스트용
            start_idx = end_idx - 300 - seq_length  # 300일 학습 + 시퀀스 길이
            
            # 학습/테스트 데이터 분할
            train_data = data.iloc[start_idx:end_idx]
            test_data = data.iloc[end_idx:end_idx+30]
            
            # 데이터 정규화 및 시퀀스 생성
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
        
        # 평균 MAPE 계산
        avg_mape = np.mean(mapes)
        results[name] = avg_mape
        print(f"{name} 평균 MAPE: {avg_mape:.2f}%")
    
    # 결과 요약
    print("\n=== 특성 조합별 성능 비교 ===")
    for name, mape in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name}: MAPE {mape:.2f}%")
    
    # 시각화
    plt.figure(figsize=(10, 6))
    names = []
    mapes = []
    for name, mape in sorted(results.items(), key=lambda x: x[1]):
        names.append(name)
        mapes.append(mape)
    
    plt.barh(names, mapes)
    plt.xlabel('MAPE (%)')
    plt.title('특성 조합별 성능 비교')
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()
        
    return results

# 9. 메인 로직
def main():
    # 설정
    ticker = '^IXIC'
    start_date = '2010-01-01'
    end_date = '2020-01-01'
    target_col = 'Close'
    seq_length = 120
    
    # 학습 및 예측 기간 설정 (일 단위)
    train_days = 300  # 학습 기간 (최소 seq_length의 2배 이상)
    pred_days = 60    # 예측 기간 (약 1개월)
    step_days = 60    # 워크포워드 이동 간격
    
    # 그래프 저장 설정
    save_graphs = False
    if save_graphs:
        save_path = f"results/{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = None
    
    # 1. 데이터 가져오기
    data = fetchData(ticker, start_date, end_date)
    print('데이터 가져오기 완료:\n', data.head())
    
    # 2. 지표 추가
    data = addSMA(data)
    data = addMACD(data)
    data = addRSI(data)
    data = addDiff(data)
    data = addDiff2(data)
    data = addChange(data)
    data['HL_Ratio'] = (data['High'] - data['Low']) / data['Close'] * 100
    
    print('\n특성 추가 완료:')
    print(data[['Close', 'SMA_5', 'SMA_20', 'SMA_60', 'MACD', 'RSI']].tail())
    
    # 결측치 제거
    data.dropna(inplace=True)
    
    # 3. 데이터 시각화
    plotChart(data, ['Close', 'SMA_20', 'SMA_60'], f'{ticker} Stock Price', save_path) 
        
    # 4. 특성 평가 수행
    print("\n다양한 특성 조합에 대한 평가를 시작합니다...")
    feature_evaluation = evaluate_feature_sets(data, target_col=target_col, seq_length=seq_length, epochs=5)
    
if __name__ == "__main__":
  main()
  
# cd ml
# python -m venv venv
# .\venv\Scripts\Activate.ps1
# pip install -r requirements.txt
# python test2.py