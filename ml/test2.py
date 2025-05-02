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

def calculateRSI(data, window=14, column='Close'):
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

def add_seasonality_features(data):
    """
    데이터프레임에 표준 달력 기반 계절성 특성을 추가합니다.
    
    Parameters:
    data (pd.DataFrame): 날짜 인덱스가 있는 DataFrame
    
    Returns:
    pd.DataFrame: 계절성 특성이 추가된 DataFrame
    """
    # 경고를 피하기 위해 복사
    data = data.copy()
    
    # 날짜 구성요소 추출
    data['day_of_week'] = data.index.dayofweek  # 월요일=0, 일요일=6
    data['month'] = data.index.month
    data['quarter'] = data.index.quarter
    data['year'] = data.index.year
    
    # 사이클릭 인코딩 적용
    # 요일에 대한 사이클릭 인코딩
    data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    
    # 월에 대한 사이클릭 인코딩
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    
    # 분기에 대한 사이클릭 인코딩
    data['quarter_sin'] = np.sin(2 * np.pi * data['quarter'] / 4)
    data['quarter_cos'] = np.cos(2 * np.pi * data['quarter'] / 4)
    
    # 연도 특성 (연도 간 변화 캡처)
    # 기준 연도 대비 상대적 연도 (최근 연도일수록 값이 큼)
    min_year = data['year'].min()
    data['rel_year'] = data['year'] - min_year
    
    # 원래 범주형 열 삭제
    data = data.drop(['day_of_week', 'month', 'quarter', 'year'], axis=1)
    
    return data

def add_trading_calendar_features(data):
    """
    거래일 기반 계절성 특성을 추가합니다.
    휴장일을 고려하여 더 정확한 계절성 특성을 생성합니다.
    
    Parameters:
    data (pd.DataFrame): 날짜 인덱스가 있는 DataFrame
    
    Returns:
    pd.DataFrame: 거래일 기반 계절성 특성이 추가된 DataFrame
    """
    # 경고를 피하기 위해 복사
    data = data.copy()
    
    try:
        # 거래일 관련 특성 추가
        # 각 날짜의 연중 거래일 번호 계산
        data['trading_day_of_year'] = data.groupby(data.index.year).cumcount() + 1
        
        # 연간 총 거래일 수 계산
        yearly_trading_days = data.groupby(data.index.year).size()
        # 각 연도의 총 거래일 수를 데이터에 매핑
        data['total_trading_days_in_year'] = data.index.year.map(yearly_trading_days)
        
        # 정규화된 연중 거래일 위치 계산 (0~1 사이 값)
        data['norm_trading_day_of_year'] = data['trading_day_of_year'] / data['total_trading_days_in_year']
        
        # 거래일 기반 사이클릭 인코딩
        data['trading_day_sin'] = np.sin(2 * np.pi * data['norm_trading_day_of_year'])
        data['trading_day_cos'] = np.cos(2 * np.pi * data['norm_trading_day_of_year'])
        
        # 간단한 거래일 특성만 유지
        data = data.drop(['trading_day_of_year', 'total_trading_days_in_year', 'norm_trading_day_of_year'], axis=1)
        
    except Exception as e:
        print(f"거래일 특성 생성 중 오류 발생: {e}")
        print("일부 거래일 특성을 건너뛰고 계속 진행합니다.")
    
    return data

# Add this function to analyze and detect seasonality
def analyze_seasonality(data, column='Close', plot=True, save_path=None):
    """
    시계열 데이터의 계절성을 분석합니다.
    
    Parameters:
    data (pd.DataFrame): 날짜 인덱스가 있는 시계열 데이터
    column (str): 계절성을 분석할 열
    plot (bool): 계절성 분해를 시각화할지 여부
    save_path (str, optional): 그래프 저장 경로
    
    Returns:
    dict: 계절성 분석 결과
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    import pandas as pd
    
    # 데이터 빈도 가져오기
    if data.index.inferred_freq is None:
        # 빈도 추론 시도
        data = data.asfreq('B')  # 영업일 빈도
    
    # 계절성 분해 수행
    try:
        # 연간 계절성 (252 영업일)
        result_annual = seasonal_decompose(data[column], model='additive', period=252)  
        
        # 분기 계절성 (63 영업일)
        result_quarterly = seasonal_decompose(data[column], model='additive', period=63)
        
        # 월간 계절성 (21 영업일)
        result_monthly = seasonal_decompose(data[column], model='additive', period=21)
        
        # 주간 계절성 (5 영업일)
        result_weekly = seasonal_decompose(data[column], model='additive', period=5)
        
        if plot:
            # 연간 계절성 분해 시각화
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 10))
            result_annual.observed.plot(ax=ax1)
            ax1.set_title('관측값')
            result_annual.trend.plot(ax=ax2)
            ax2.set_title('추세')
            result_annual.seasonal.plot(ax=ax3)
            ax3.set_title('계절성 (연간)')
            result_annual.resid.plot(ax=ax4)
            ax4.set_title('잔차')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}/annual_seasonal_decomposition.png")
                plt.close()
            
            # 각 계절성 구성요소 비교
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
            
            # 연간/분기/월간 계절성 패턴 비교
            result_annual.seasonal.iloc[-252:].plot(ax=axes[0], label='연간 계절성')
            result_quarterly.seasonal.iloc[-252:].plot(ax=axes[0], label='분기 계절성')
            result_monthly.seasonal.iloc[-252:].plot(ax=axes[0], label='월간 계절성')
            axes[0].set_title('다양한 계절성 패턴 비교')
            axes[0].legend()
            
            # 월간 평균 수익률 분석 (월별 효과)
            monthly_returns = data[column].pct_change().groupby(data.index.month).mean() * 100
            monthly_returns.index = ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']
            monthly_returns.plot(kind='bar', ax=axes[1])
            axes[1].set_title('월별 평균 수익률')
            axes[1].set_ylabel('평균 일일 수익률 (%)')
            
            # 요일별 평균 수익률 분석 (요일 효과)
            daily_returns = data[column].pct_change().groupby(data.index.dayofweek).mean() * 100
            daily_returns.index = ['월', '화', '수', '목', '금']
            daily_returns.plot(kind='bar', ax=axes[2])
            axes[2].set_title('요일별 평균 수익률')
            axes[2].set_ylabel('평균 일일 수익률 (%)')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}/seasonal_patterns_analysis.png")
                plt.close()
        
        # 계절성 강도 계산
        seasonal_annual_var = np.var(result_annual.seasonal)
        seasonal_quarterly_var = np.var(result_quarterly.seasonal)
        seasonal_monthly_var = np.var(result_monthly.seasonal)
        seasonal_weekly_var = np.var(result_weekly.seasonal)
        
        residual_var = np.var(result_annual.resid.dropna())
        trend_var = np.var(result_annual.trend.dropna())
        
        # 각 주기별 계절성 강도
        annual_strength = seasonal_annual_var / (seasonal_annual_var + residual_var)
        quarterly_strength = seasonal_quarterly_var / (seasonal_quarterly_var + residual_var)
        monthly_strength = seasonal_monthly_var / (seasonal_monthly_var + residual_var)
        weekly_strength = seasonal_weekly_var / (seasonal_weekly_var + residual_var)
        trend_strength = trend_var / (trend_var + residual_var)
        
        # 월별, 요일별 효과 분석
        monthly_effect = data[column].pct_change().groupby(data.index.month).mean()
        day_of_week_effect = data[column].pct_change().groupby(data.index.dayofweek).mean()
        
        # 분석 결과
        analysis = {
            'trend_strength': trend_strength,
            'annual_seasonality_strength': annual_strength,
            'quarterly_seasonality_strength': quarterly_strength,
            'monthly_seasonality_strength': monthly_strength,
            'weekly_seasonality_strength': weekly_strength,
            'monthly_effect': monthly_effect.to_dict(),
            'day_of_week_effect': day_of_week_effect.to_dict()
        }
        
        print("\n====== 계절성 분석 결과 ======")
        print(f"추세 강도: {trend_strength:.4f}")
        print(f"연간 계절성 강도: {annual_strength:.4f}")
        print(f"분기 계절성 강도: {quarterly_strength:.4f}")
        print(f"월간 계절성 강도: {monthly_strength:.4f}")
        print(f"주간 계절성 강도: {weekly_strength:.4f}")
        
        print("\n----- 월별 수익률 효과 -----")
        for month, effect in monthly_effect.items():
            month_name = ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월'][month-1]
            print(f"{month_name}: {effect*100:.4f}%")
        
        print("\n----- 요일별 수익률 효과 -----")
        for day, effect in day_of_week_effect.items():
            day_name = ['월요일', '화요일', '수요일', '목요일', '금요일'][day]
            print(f"{day_name}: {effect*100:.4f}%")
        
        return analysis
    
    except Exception as e:
        print(f"계절성 분해 중 오류 발생: {e}")
        return None
    
# evaluate_feature_sets 함수 추가
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
        'SMA_5', 
        'SMA_20', 
        'SMA_60', 
        'RSI', 
        'MACD', 
        'Price_Change', 
        'HL_Ratio'
    ]
    
    # 계절성 특성
    seasonality_features = [
        # add_seasonality_features 함수에서 추가되는 특성
        'day_of_week_sin',   # 요일 사인 변환
        'day_of_week_cos',   # 요일 코사인 변환
        'month_sin',         # 월 사인 변환
        'month_cos',         # 월 코사인 변환
        'quarter_sin',       # 분기 사인 변환
        'quarter_cos',       # 분기 코사인 변환
        'rel_year',          # 상대적 연도
        
        # add_trading_calendar_features 함수에서 추가되는 특성
        'trading_day_sin',               # 연중 거래일 사인 변환
        'trading_day_cos',               # 연중 거래일 코사인 변환
    ]
    
    # 특성 조합 정의
    feature_sets = {
        "기본 특성만": base_features,
        "기본 + 기술적 지표": base_features + technical_features,
        "기본 + 계절성": base_features + seasonality_features,
        "모든 특성": base_features + technical_features + seasonality_features
    }
    
    results = {}
    
    # 각 특성 조합 평가
    for name, feature_set in feature_sets.items():
        # 해당 특성이 데이터에 있는지 확인
        valid_features = [f for f in feature_set if f in data.columns]
        
        if len(valid_features) < 2:  # 최소 2개 이상의 유효한 특성 필요
            print(f"{name}: 유효한 특성이 부족합니다.")
            continue
            
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
    
    # 2. 기술적 지표 추가
    data = addSMA(data)
    data = addMACD(data)
    data = calculateRSI(data)
    
    # 거래량 로그 변환
    data['log_Volume'] = np.log(data['Volume'] + 1)  # 0 값 처리를 위해 +1
    
    # 가격 변화율 추가
    data['Price_Change'] = data['Close'].pct_change() * 100
    
    # 추가 특성: 고가-저가 비율
    data['HL_Ratio'] = (data['High'] - data['Low']) / data['Close'] * 100
    
    # 결측치 제거 - 계절성 분석 전에 수행
    data.dropna(inplace=True)
    
    # 계절성 분석 시도 - 실패해도 계속 진행
    try:
        seasonality_analysis = analyze_seasonality(data, column='Close', plot=True, save_path=save_path)
    except Exception as e:
        print(f"계절성 분석 중 오류 발생: {e}")
        print("계절성 분석을 건너뛰고 계속 진행합니다.")
        seasonality_analysis = None
    
    # 계절성 특성 추가 - 실패해도 계속 진행
    data_with_seasonality = None
    try:
        # 표준 계절성 특성 추가
        data_with_seasonality = add_seasonality_features(data.copy())
        print("표준 계절성 특성 추가 완료")
    except Exception as e:
        print(f"표준 계절성 특성 추가 중 오류 발생: {e}")
        print("표준 계절성 특성을 건너뛰고 계속 진행합니다.")
        data_with_seasonality = data.copy()  # 원본 데이터 유지
    
    # 거래일 기반 계절성 특성 추가 - 실패해도 계속 진행
    try:
        if data_with_seasonality is not None:
            data_with_seasonality = add_trading_calendar_features(data_with_seasonality)
            data = data_with_seasonality  # 성공하면 결과를 원본 데이터에 반영
            print("거래일 기반 계절성 특성 추가 완료")
    except Exception as e:
        print(f"거래일 기반 계절성 특성 추가 중 오류 발생: {e}")
        print("거래일 기반 계절성 특성을 건너뛰고 계속 진행합니다.")
    
    print('\n특성 추가 완료:')
    print(data[['Close', 'SMA_5', 'SMA_20', 'SMA_60', 'MACD', 'RSI']].tail())
    
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