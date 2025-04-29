# cd ml
# python -m venv venv
# .\venv\Scripts\Activate.ps1
# ./venv/bin/activate
# pip install -r requirements.txt

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
import math


# yf.download는 pandas.Dataframe 자료구조로 return
data = yf.download('AAPL', start='2022-01-01', end='2023-01-01')
data.head()

# 결측치 검사 0이 나온다면, 각 열에 결측치(NaN)가 하나도 없다
data.isnull().sum()

# 데이터프레임의 열 이름(키) 확인
print("원본 데이터 컬럼:", data.columns)

# 데이터프레임 정보 요약 확인
data.info()

# 멀티인덱스 컬럼('AAPL', 'Close')에서 단일 레벨 컬럼('Close')으로
data.columns = data.columns.droplevel(1)
print("컬럼 변환 후:", data.columns)

# 1. 데이터 준비 및 전처리
# 종가 데이터만 사용
close_data = data[['Close']]  # 'Close' 컬럼만 선택
print("종가 데이터 형태:", close_data.shape)

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)
print("스케일링된 데이터 형태:", scaled_data.shape)

# 2. 학습 및 테스트 데이터셋 생성
print(f"전체 데이터 크기: {len(scaled_data)}")
train_size = int(len(scaled_data) * 0.6)  # 80%에서 60%로 조정
test_size = len(scaled_data) - train_size
train_data = scaled_data[0:train_size, :]
test_data = scaled_data[train_size:, :]
print(f"학습 데이터 크기: {len(train_data)}")
print(f"테스트 데이터 크기: {len(test_data)}")

# 시퀀스 데이터 생성 함수
def create_sequences(data, time_steps):
    X, y = [], []
    if len(data) <= time_steps:
        print(f"경고: 데이터 길이({len(data)})가 time_steps({time_steps})보다 작거나 같습니다.")
        return np.array([]), np.array([])
        
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    
    X_array = np.array(X)
    y_array = np.array(y)
    print(f"생성된 시퀀스 - X 형태: {X_array.shape}, y 형태: {y_array.shape}")
    
    return X_array, y_array

# 적절한 time_steps 확인
if len(test_data) <= 60:
    print(f"경고: 테스트 데이터({len(test_data)})가 기본 time_steps(60)보다 작거나 같습니다.")
    time_steps = min(30, len(test_data) // 2)  # 안전하게 조정
    print(f"time_steps를 {time_steps}로 조정합니다.")
else:
    time_steps = 60
    print(f"기본 time_steps {time_steps}를 사용합니다.")

# 시퀀스 생성
X_train, y_train = create_sequences(train_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

# 빈 시퀀스 확인
if len(X_train) == 0 or len(X_test) == 0:
    print("경고: 생성된 시퀀스가 비어 있습니다. 더 작은 time_steps로 다시 시도합니다.")
    time_steps = min(15, len(test_data) // 3)  # 훨씬 작은 값으로 설정
    print(f"time_steps를 {time_steps}로 재설정합니다.")
    X_train, y_train = create_sequences(train_data, time_steps)
    X_test, y_test = create_sequences(test_data, time_steps)
    
    # 여전히 빈 배열이면 오류 발생
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("시퀀스 생성 실패. 데이터셋이 너무 작거나 분할이 적절하지 않습니다.")

print("학습 시퀀스 형태:", X_train.shape)
print("테스트 시퀀스 형태:", X_test.shape)

# LSTM 입력을 위한 reshape: [samples, time_steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print("LSTM 입력 형태 (학습):", X_train.shape)
print("LSTM 입력 형태 (테스트):", X_test.shape)

# 3. LSTM 모델 구축
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# 모델 요약
model.summary()

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 4. 모델 학습
history = model.fit(
    X_train, 
    y_train, 
    epochs=50, 
    batch_size=32, 
    validation_split=0.1,
    verbose=1
)

# 5. 예측 및 성능 평가
# 테스트 데이터에 대한 예측
predictions = model.predict(X_test)
# 스케일 원복
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 평균 제곱근 오차(RMSE) 계산
rmse = math.sqrt(mean_squared_error(y_test_actual, predictions))
print(f'테스트 RMSE: {rmse}')

# 6. 결과 시각화
# 학습 손실 그래프
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('모델 학습 손실')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 예측 결과 시각화
plt.figure(figsize=(12, 6))
# 실제 주가
plt.plot(y_test_actual, color='blue', label='실제 주가')
# 예측 주가
plt.plot(predictions, color='red', label='예측 주가')
plt.title('주가 예측 결과')
plt.xlabel('시간')
plt.ylabel('주가')
plt.legend()
plt.show()

# 7. 미래 주가 예측 (선택적)
# 마지막 time_steps일 데이터로 다음날 예측
last_sequence = scaled_data[-time_steps:, :]
X_future = np.array([last_sequence[:, 0]])
X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))
future_price = model.predict(X_future)
future_price = scaler.inverse_transform(future_price)
print(f'다음 거래일 예상 주가: {future_price[0][0]}')