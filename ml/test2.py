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

plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 경우 '맑은 고딕' 폰트

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
    data['Close_Diff'] = data['Close'].diff()
    data['Open_Diff'] = data['Open'].diff()
    data['Volume_Diff'] = data['Volume'].diff()
    data['Low_Diff'] = data['Low'].diff()
    data['High_Diff'] = data['High'].diff()
    
    return data

def addDiff2(data):
    data['Close_Diff2'] = data['Close_Diff'].diff()
    data['Open_Diff2'] = data['Open_Diff'].diff()
    data['Volume_Diff2'] = data['Volume_Diff'].diff()
    data['Low_Diff2'] = data['Low_Diff'].diff()
    data['High_Diff2'] = data['High_Diff'].diff()
    
    return data

def addChange(data):
    data['Close_Change'] = data['Close'].pct_change() * 100
    data['Open_Change'] = data['Open'].pct_change() * 100
    data['High_Change'] = data['High'].pct_change() * 100
    data['Low_Change'] = data['Low'].pct_change() * 100
    data['Volume_Change'] = data['Volume'].pct_change() * 100
    
    return data

def plotChart(data, columns=['Close', 'SMA_20', 'SMA_60'], title=None, save_path=None, show_macd=False):
    """
    주가 데이터와 지표를 시각화합니다.
    
    Parameters:
    data (pd.DataFrame): 주가 데이터
    columns (list): 시각화할 열 이름 리스트
    title (str, optional): 차트 제목
    save_path (str, optional): 저장 경로
    show_macd (bool, optional): MACD 차트를 함께 표시할지 여부
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
    
    # MACD 차트 (MACD가 있는 경우 & show_macd가 True인 경우에만)
    if show_macd and 'MACD' in data.columns and 'Signal' in data.columns:
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

def saveData(data, filename):
    """
    전처리된 데이터를 파일로 저장합니다.
    
    Parameters:
    data (pd.DataFrame): 저장할 데이터
    filename (str): 파일 이름 (확장자 제외)
    """
    # 데이터 디렉토리 생성
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 멀티 인덱스가 있는지 확인하고 리셋
    if isinstance(data.index, pd.MultiIndex) or data.index.name == 'Ticker':
        data = data.reset_index()
    
    # 날짜 인덱스가 있는 데이터프레임 저장
    data.to_csv(f'data/{filename}.csv', index=True)
    print(f"데이터 저장 완료: data/{filename}.csv")

def loadData(filename):
    """
    저장된 데이터를 로드합니다.
    
    Parameters:
    filename (str): 파일 이름 (확장자 제외)
    
    Returns:
    pd.DataFrame: 로드된 데이터
    """
    # 파일 존재 여부 확인
    file_path = f'data/{filename}.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    # 데이터프레임 로드
    data = pd.read_csv(file_path)
    
    # 'Date' 열이 있으면 이를 인덱스로 설정
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
    
    print(f"데이터 로드 완료: {file_path}")
    print(f"데이터 형태: {data.shape}")
    print(f"인덱스 타입: {data.index.dtype}")
    
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

def plotBarChart(data, x_col, y_col, title=None, xlabel=None, ylabel=None, horizontal=True, save_path=None):
    """
    막대 차트를 그립니다.
    
    Parameters:
    data (pd.DataFrame): 차트 데이터
    x_col (str): x축 열 이름
    y_col (str): y축 열 이름
    title (str, optional): 차트 제목
    xlabel (str, optional): x축 레이블
    ylabel (str, optional): y축 레이블
    horizontal (bool): 가로 막대 차트 여부
    save_path (str, optional): 저장 경로
    """
    plt.figure(figsize=(10, 6))
    
    if horizontal:
        plt.barh(data.index, data[y_col])
        plt.xlabel(xlabel or y_col)
        plt.ylabel(ylabel or x_col)
    else:
        plt.bar(data.index, data[y_col])
        plt.xlabel(xlabel or x_col)
        plt.ylabel(ylabel or y_col)
    
    plt.title(title or f'{y_col} by {x_col}')
    plt.grid(True, axis='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

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
        'Close_Diff',
        'Open_Diff', 
        'High_Diff',
        'Low_Diff',
        'Volume_Diff',
        'SMA_5', 
        'SMA_20', 
        'SMA_60', 
        'RSI', 
        'MACD', 
        'HL_Ratio'
    ]
    
    technical_features2 = [
        'Close_Diff2',
        'Open_Diff2', 
        'High_Diff2',
        'Low_Diff2',
        'Volume_Diff2',
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
    
    # 결과 요약 및 시각화
    print("\n=== 특성 조합별 성능 비교 ===")
    for name, mape in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name}: MAPE {mape:.2f}%")

    # 결과 데이터프레임 생성
    result_df = pd.DataFrame({
        'MAPE': [mape for _, mape in sorted(results.items(), key=lambda x: x[1])]
    }, index=[name for name, _ in sorted(results.items(), key=lambda x: x[1])])

    # 가로 막대 차트로 시각화
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"results/feature_evaluation_{timestamp}.png"
    plotBarChart(result_df, '', 'MAPE', '특성 조합별 성능 비교', 'MAPE (%)', '특성 조합', True, save_path)
        
    return results

# 9. 메인 로직
def main():
    # 설정
    ticker = '^IXIC'
    start_date = '2010-01-01'
    end_date = '2020-01-01'
    target_col = 'Close'
    seq_length = 120
    
    # 파일 이름 설정
    data_filename = f"{ticker}_{start_date.replace('-', '')}_{end_date.replace('-', '')}"
    
    # 1. 데이터 가져오기
    rawData = fetchData(ticker, start_date, end_date)
    rawData.columns = rawData.columns.get_level_values(0)
    print('데이터 가져오기 완료:\n', rawData.head())
    
    # 2. 지표 추가
    print("\n기술적 지표 추가 중...")
    processedData = rawData.copy()
    processedData = addSMA(processedData)
    processedData = addMACD(processedData)
    processedData = addRSI(processedData)
    processedData = addDiff(processedData)
    processedData = addDiff2(processedData)
    processedData = addChange(processedData)
    processedData['HL_Ratio'] = (processedData['High'] - processedData['Low']) / processedData['Close'] * 100
    
    # 결측치 제거
    processedData.dropna(inplace=True)
    
    print("기술적 지표 추가 완료!")
    
    # 3. 전처리된 데이터 저장
    saveData(processedData, data_filename)
    
    # 저장된 데이터 로드
    print(f"\n저장된 데이터 파일 '{data_filename}' 로드 중...")
    loadedData = loadData(data_filename)
    print("저장된 데이터 파일 로드 완료!")
    
    # 데이터 확인
    print('\n데이터 특성 확인:')
    print(loadedData[['Close', 'SMA_5', 'SMA_20', 'SMA_60', 'MACD', 'RSI', 'Close_Diff', 'Close_Diff2', 'Close_Change']].tail())
    
    loadedData.to_csv(f'data/loaded.csv', index=True)
    # 그래프 저장 설정
    save_graphs = True
    if save_graphs:
        save_path = f"results/{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = None
    
    # 4. 데이터 시각화
    print("\n데이터 시각화 중...")
    plotChart(loadedData, ['Close', 'SMA_20', 'SMA_60'], f'{ticker} 주가 및 이동평균', f'{save_path}/price_and_sma') 
    plotChart(loadedData, ['MACD', 'Signal'], f'{ticker} MACD 지표', f'{save_path}/macd', show_macd=True)
    plotChart(loadedData, ['RSI'], f'{ticker} RSI 지표', f'{save_path}/rsi')
    plotChart(loadedData, ['Close_Diff', 'Close_Diff2'], f'{ticker} 가격 차분', f'{save_path}/price_diff')
    plotChart(loadedData, ['Close_Change'], f'{ticker} 가격 변화율(%)', f'{save_path}/price_change')
    print("데이터 시각화 완료!")
    
    # 5. 특성 평가 수행
    print("\n다양한 특성 조합에 대한 평가를 시작합니다...")
    feature_evaluation = evaluate_feature_sets(loadedData, target_col=target_col, seq_length=seq_length, epochs=5)
    print("특성 평가 완료!")
    
    # 6. 평가 결과 요약
    print("\n=== 최종 특성 조합별 성능 요약 ===")
    for name, mape in sorted(feature_evaluation.items(), key=lambda x: x[1]):
        print(f"{name}: MAPE {mape:.2f}%")
    
    # 최고 성능 특성 조합 선택
    best_feature_set = min(feature_evaluation.items(), key=lambda x: x[1])[0]
    print(f"\n최고 성능 특성 조합: {best_feature_set}")
    
if __name__ == "__main__":
    main()
  
# cd ml
# python -m venv venv
# .\venv\Scripts\Activate.ps1
# pip install -r requirements.txt
# python test2.py

# cd ml
# python3 -m venv venv
# source .venv/bin/activate
# pip install -r requirements.txt
# python3 test2.py
