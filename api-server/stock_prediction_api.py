import os
import joblib
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
from datetime import datetime, timedelta
from hdfs import InsecureClient
from io import BytesIO
from confluent_kafka import Producer
from typing import List, Dict, Any
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Stock Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영환경에서는 특정 도메인만 허용하는 것이 좋습니다
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 설정
ENVIRONMENT = os.getenv('EXEC_ENV', 'local')  # 기본값은 local

# 환경별 설정
ENV_CONFIG = {
    'local': {
        'HADOOP_URL': 'http://host.minikube.internal:9870',
        'KAFKA_URL':'http://host.minikube.internal:9092'
    },
    'ec2-kubernetes': {
        'HADOOP_URL': 'http://18.190.148.99:9870',
        'KAFKA_URL':'http://18.190.148.99:9092'
    }
}

current_config = ENV_CONFIG.get(ENVIRONMENT, ENV_CONFIG['local'])

# kubernates 환경인 경우 minikube.host로
producer = Producer({
    'bootstrap.servers': os.getenv('KAFKA_URL', current_config['KAFKA_URL'])
})

# 설정
hadoop_url = os.getenv('HADOOP_URL', current_config['HADOOP_URL'])
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
    is_intraday: bool
    historical_data: List[Dict[str, Any]]

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
        
        predicted_close = float(next_day_pred['Pred_Close'].iloc[0])
        current_close = float(df['Close'].iloc[-1])
        
        # 5. 장중 데이터인지 확인
        last_data_date = pd.to_datetime(df['Date'].iloc[-1]).date()
        prediction_date = next_day_pred.index[0].date()
        
        is_intraday = (last_data_date == prediction_date)
        
        kafka_message = {
            "message": "test message",
            "timestamp": datetime.now().isoformat()
        }
        
        # Kafka로 메시지 전송
        producer.produce(
            'test_1',
            key=str(datetime.now().timestamp()),
            value=json.dumps(kafka_message)
        )
        producer.flush()
        
        # 장중인 경우에만 알림 발생
        if is_intraday:
            price_diff_percent = abs(predicted_close - current_close) / current_close * 100
            
            if price_diff_percent >= 1.0:
                alert_message = {
                    "ticker": ticker,
                    "current_price": current_close,
                    "predicted_price": predicted_close,
                    "price_diff_percent": price_diff_percent,
                    "prediction_date": prediction_date.strftime('%Y-%m-%d'),
                    "last_update": df['Date'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    "alert_timestamp": datetime.now().isoformat(),
                    "alert_type": "intraday_price_difference",
                    "direction": "up" if predicted_close > current_close else "down",
                    "hi": "hello"
                }
                
                producer.produce(
                    'test_1',
                    key=ticker,
                    value=json.dumps(alert_message)
                )
                producer.flush()
        
        response = PredictionResponse(
            ticker=ticker,
            prediction_date=prediction_date.strftime('%Y-%m-%d'),
            predicted_close=predicted_close,
            current_close=current_close,
            prediction_timestamp=datetime.now().isoformat(),
            is_intraday=is_intraday,
            historical_data = df.to_dict('records')
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