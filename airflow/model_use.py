import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# 모델과 스케일러 로드
model_path = "IXIC_기본_특성_model.keras"  # 모델 경로
scaler_path = "IXIC_기본_특성_scalers.pkl"  # 스케일러 경로

# 모델 로드
model = tf.keras.models.load_model(model_path)

# 스케일러 로드
with open(scaler_path, 'rb') as f:
    scaler_dict = pickle.load(f)

# 예측 함수
def predict_stock(ticker='^IXIC', days=7):
    # 더 많은 데이터 가져오기 (충분한 거래일을 확보하기 위해)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1년치 데이터 (충분한 거래일 확보)
    
    print(f"{ticker} 데이터 가져오는 중...")
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    print(f"총 {len(data)}개 데이터 로드됨")
    
    # 사용할 특성 (모델 학습에 사용된 특성과 일치해야 함)
    features = ['Close', 'Open', 'High', 'Low', 'Volume']
    
    # 모델 입력 준비 (정확히 마지막 120일 데이터)
    seq_length = 120
    
    if len(data) < seq_length:
        raise ValueError(f"데이터가 부족합니다. 필요: {seq_length}, 실제: {len(data)}")
    
    # 정확히 seq_length개 행만 선택
    input_data = data[features].iloc[-seq_length:].values
    print(f"입력 데이터 형태: {input_data.shape}")
    
    # 정규화
    input_scaled = np.zeros((1, seq_length, len(features)))
    for i, col in enumerate(features):
        try:
            scaled_values = scaler_dict[col].transform(input_data[:, i].reshape(-1, 1)).flatten()
            input_scaled[0, :, i] = scaled_values
        except Exception as e:
            print(f"특성 '{col}' 정규화 중 오류: {str(e)}")
            raise
    
    # 예측
    pred = model.predict(input_scaled, verbose=0)[0, 0]
    
    # 역정규화
    predicted_price = scaler_dict['Close'].inverse_transform(np.array([[pred]]))[0, 0]
    
    # 마지막 종가와 함께 출력
    last_close = data['Close'].iloc[-1]
    print(f"\n마지막 종가 ({data.index[-1].strftime('%Y-%m-%d')}): {last_close:.2f}")
    print(f"다음 거래일 예상 종가: {predicted_price:.2f}")
    print(f"예상 변화: {(predicted_price - last_close):.2f} ({(predicted_price - last_close) / last_close * 100:.2f}%)")
    
    return predicted_price

# 실행
if __name__ == "__main__":
    try:
        predict_stock()
    except Exception as e:
        print(f"예측 중 오류 발생: {str(e)}")