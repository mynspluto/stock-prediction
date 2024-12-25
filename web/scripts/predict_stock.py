# scripts/predict_stock.py
import sys
import json
from hdfs import InsecureClient
import joblib
from io import BytesIO
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import traceback

def load_model_from_hdfs():
    """HDFS에서 모델과 스케일러 로드"""
    try:
        print("HDFS 연결 시도...", file=sys.stderr)
        client = InsecureClient('http://127.0.0.1:9870', user='hadoop')
        model_path = '/stock-history/^IXIC/model'
        
        print("모델 파일 읽기 시도...", file=sys.stderr)
        with client.read(f'{model_path}/model.joblib') as reader:
            model_bytes = reader.read()
        model = joblib.load(BytesIO(model_bytes))
        print("모델 로드 성공", file=sys.stderr)
        
        print("스케일러 파일 읽기 시도...", file=sys.stderr)
        with client.read(f'{model_path}/scalers.joblib') as reader:
            scalers_bytes = reader.read()
        scalers = joblib.load(BytesIO(scalers_bytes))
        print("스케일러 로드 성공", file=sys.stderr)
        
        return model, scalers
    except Exception as e:
        print(f"모델 로드 중 에러 발생: {str(e)}", file=sys.stderr)
        print("상세 에러:", traceback.format_exc(), file=sys.stderr)
        raise

def create_ml_features(df):
    """주가 데이터프레임에서 머신러닝용 특징 생성"""
    df_ml = df.copy()
    
    # 이동평균선
    df_ml['MA5'] = df_ml['Close'].rolling(window=5, min_periods=1).mean()
    df_ml['MA20'] = df_ml['Close'].rolling(window=20, min_periods=1).mean()
    df_ml['MA60'] = df_ml['Close'].rolling(window=60, min_periods=1).mean()
    
    # 거래량 이동평균
    df_ml['Volume_MA5'] = df_ml['Volume'].rolling(window=5, min_periods=1).mean()
    
    # 변동성 지표
    df_ml['Price_Range'] = df_ml['High'] - df_ml['Low']
    df_ml['Price_Change'] = df_ml['Close'] - df_ml['Open']
    
    # RSI
    delta = df_ml['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df_ml['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df_ml['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_ml['Close'].ewm(span=26, adjust=False).mean()
    df_ml['MACD'] = exp1 - exp2
    df_ml['Signal_Line'] = df_ml['MACD'].ewm(span=9, adjust=False).mean()
    
    return df_ml.fillna(method='ffill').fillna(method='bfill')

def get_latest_stock_data(ticker='^IXIC'):
    """Yahoo Finance에서 최신 주가 데이터 가져오기"""
    try:
        print("주가 데이터 다운로드 시도...", file=sys.stderr)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 90일치 데이터
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        print(f"다운로드된 데이터 행 수: {len(df)}", file=sys.stderr)
        
        return df
    except Exception as e:
        print(f"주가 데이터 가져오기 실패: {str(e)}", file=sys.stderr)
        raise

def predict_stock_prices(df, model, scalers):
    """학습된 모델을 사용하여 주가 예측"""
    try:
        print("예측 준비 중...", file=sys.stderr)
        scaler_X, scaler_y = scalers
        
        # 예측을 위한 마지막 데이터 준비
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        last_data = df[feature_columns].iloc[-1:]
        print("마지막 데이터:", last_data.to_dict('records'), file=sys.stderr)
        
        # 스케일링 적용
        X_pred = scaler_X.transform(last_data)
        print("스케일링된 입력 shape:", X_pred.shape, file=sys.stderr)
        
        # 예측 수행
        predictions_scaled = model.predict(X_pred)
        predictions = scaler_y.inverse_transform(predictions_scaled)
        print("예측 결과:", predictions, file=sys.stderr)
        
        # 다음 날짜
        last_date = df.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        
        # 결과 반환
        current_values = {
            "current_price": float(df['Close'].iloc[-1]),
            "predicted_price": float(predictions[0][3]),  # Close price
            "price_change": float((predictions[0][3] - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100),
            "last_update": last_date.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print("최종 결과:", current_values, file=sys.stderr)
        print(json.dumps(current_values))
        sys.exit(0)
        
    except Exception as e:
        print(f"예측 중 에러 발생: {str(e)}", file=sys.stderr)
        print("상세 에러:", traceback.format_exc(), file=sys.stderr)
        raise

def predict_next_day():
    try:
        print("예측 프로세스 시작...", file=sys.stderr)
        
        # 1. 모델과 스케일러 로드
        model, scalers = load_model_from_hdfs()
        
        # 2. 최신 주가 데이터 가져오기
        df = get_latest_stock_data()
        
        # 3. 특징 생성
        df_ml = create_ml_features(df)
        
        # 4. 예측 수행
        predict_stock_prices(df_ml, model, scalers)
        
    except Exception as e:
        print(f"예측 중 에러 발생: {str(e)}", file=sys.stderr)
        print("상세 에러:", traceback.format_exc(), file=sys.stderr)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    predict_next_day()