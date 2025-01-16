from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import yfinance as yf
from datetime import datetime, timedelta
from hdfs import InsecureClient
import pandas as pd
from io import BytesIO
import joblib
import numpy as np
import json
from confluent_kafka import Producer

app = FastAPI(title="Stock Prediction API")

producer = Producer({
    'bootstrap.servers': 'localhost:9092'
})

# 설정
hadoop_url = 'http://localhost:9870'
hdfs_path = "/stock-history"

# HDFS 클라이언트 초기화
client = InsecureClient(hadoop_url)

def create_ml_features(df):
    """주가 데이터프레임에서 머신러닝용 특징 생성"""
    df_ml = df.copy()
    
    # 이동평균선
    df_ml['MA5'] = df_ml['Close'].rolling(window=5).mean()
    df_ml['MA20'] = df_ml['Close'].rolling(window=20).mean()
    df_ml['MA60'] = df_ml['Close'].rolling(window=60).mean()
    
    # 거래량 이동평균
    df_ml['Volume_MA5'] = df_ml['Volume'].rolling(window=5).mean()
    
    # MACD
    exp1 = df_ml['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_ml['Close'].ewm(span=26, adjust=False).mean()
    df_ml['MACD'] = exp1 - exp2
    df_ml['Signal_Line'] = df_ml['MACD'].ewm(span=9, adjust=False).mean()
    
    # 결측값 처리
    df_ml = df_ml.fillna(method='bfill')
    
    return df_ml

def get_latest_stock_data(ticker):
    """Yahoo Finance에서 최신 주가 데이터 가져오기"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        df = df.reset_index()
        df = df.rename(columns={'index': 'Date', 'Stock Splits': 'StockSplits'})
        
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock data: {str(e)}")

def load_model_from_hdfs(client, ticker):
    """HDFS에서 저장된 모델과 스케일러 로드"""
    try:
        model_path = f'{hdfs_path}/{ticker}/model'
        
        with client.read(f'{model_path}/model.joblib') as reader:
            model_bytes = reader.read()
        model = joblib.load(BytesIO(model_bytes))
        
        with client.read(f'{model_path}/scalers.joblib') as reader:
            scalers_bytes = reader.read()
        scalers = joblib.load(BytesIO(scalers_bytes))
        
        return model, scalers
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

def predict_stock_prices(df, model, scalers):
    """학습된 모델을 사용하여 주가 예측"""
    try:
        scaler_X, scaler_y = scalers
        feature_columns = ['Close', 'Volume', 'MACD', 'Signal_Line']
        
        X_pred = df[feature_columns].values[-1:]
        X_pred_scaled = scaler_X.transform(X_pred)
        
        predictions_scaled = model.predict(X_pred_scaled)
        predictions_scaled = predictions_scaled.reshape(-1, 1)
        predictions = scaler_y.inverse_transform(predictions_scaled)
        
        last_date = pd.to_datetime(df['Date'].iloc[-1])
        next_date = last_date + pd.Timedelta(days=1)
        
        pred_df = pd.DataFrame(
            predictions, 
            columns=['Pred_Close'],
            index=[next_date]
        )
        
        return pred_df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

class PredictionResponse(BaseModel):
    ticker: str
    prediction_date: str
    predicted_close: float
    current_close: float
    prediction_timestamp: str

@app.get("/")
def read_root():
    return {"message": "Stock Prediction API"}

@app.get("/predict/{ticker}", response_model=PredictionResponse)
async def predict(ticker: str):
    try:
        # 1. 모델 및 스케일러 로드
        model, scalers = load_model_from_hdfs(client, ticker)
        
        # 2. 최신 주가 데이터 가져오기
        df = get_latest_stock_data(ticker)
        
        # 3. 특징 생성
        df_ml = create_ml_features(df)
        
        # 4. 예측 수행
        next_day_pred = predict_stock_prices(df_ml, model, scalers)
        
        # 5. 응답 생성
        response = PredictionResponse(
            ticker=ticker,
            prediction_date=next_day_pred.index[0].strftime('%Y-%m-%d'),
            predicted_close=float(next_day_pred['Pred_Close'].iloc[0]),
            current_close=float(df['Close'].iloc[-1]),
            prediction_timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/produce/{message}")
async def produce_message(message: str):
    try:
        # 메시지 준비
        kafka_message = {
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        # Kafka로 메시지 전송
        producer.produce(
            'test_1',
            key=str(datetime.now().timestamp()),
            value=json.dumps(kafka_message)
        )
        producer.flush()
        
        return {
            "status": "success",
            "message": f"Message sent to Kafka: {message}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to send message: {str(e)}"
        }

if __name__ == "__main__":
    uvicorn.run("stock_prediction_api:app", host="0.0.0.0", port=8000, reload=True)