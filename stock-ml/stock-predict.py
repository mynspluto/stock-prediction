
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
    
def create_ml_features(df):
    """
    주가 데이터프레임에서 머신러닝용 특징 생성
    """
    # 기존 데이터 복사
    df_ml = df.copy()
    
    # 이동평균선
    df_ml['MA5'] = df_ml['Close'].rolling(window=5).mean()
    df_ml['MA20'] = df_ml['Close'].rolling(window=20).mean()
    df_ml['MA60'] = df_ml['Close'].rolling(window=60).mean()
    
    # 거래량 이동평균
    df_ml['Volume_MA5'] = df_ml['Volume'].rolling(window=5).mean()
    
    # 변동성 지표
    df_ml['Price_Range'] = df_ml['High'] - df_ml['Low']
    df_ml['Price_Change'] = df_ml['Close'] - df_ml['Open']
    
    # RSI (Relative Strength Index)
    delta = df_ml['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_ml['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df_ml['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_ml['Close'].ewm(span=26, adjust=False).mean()
    df_ml['MACD'] = exp1 - exp2
    df_ml['Signal_Line'] = df_ml['MACD'].ewm(span=9, adjust=False).mean()
    
    # 결측값 처리
    df_ml = df_ml.fillna(method='bfill')
    
    return df_ml

def prepare_ml_data(df_ml, window_size=252):  # 1년치 거래일 기준
    """
    시간에 따른 추세를 고려한 데이터 준비
    """
    # 기본 특징 설정
    feature_columns = ['Close', 'Volume']
    target_columns = ['Close']

    # 퍼센트 변화율 계산
    df_pct = df_ml[feature_columns].pct_change()
    df_pct = df_pct.fillna(0)
    
    # 입력 데이터와 타겟 데이터 준비
    X = df_pct[feature_columns].values[:-1]  # 마지막 날 제외
    print('X', X)
    
    # 다음날 가격 변화율을 타겟으로
    y_pct = df_pct[target_columns].shift(-1).values[:-1]
    
    # 학습/테스트 분할 (80:20)
    # split_point = int(len(X) * 0.8)
    split_point = int(len(X))
    X_train = X[:split_point]
    X_test = X[split_point:]
    y_train = y_pct[:split_point]
    y_test = y_pct[split_point:]
    
    return (X_train, X_test, y_train, y_test), df_ml

def train_stock_model(X_train, y_train):
    """
    변화율 예측을 위한 모델 학습
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def predict_stock_prices(df, model, last_data):
    """
    변화율 기반 예측
    """
    # 최근 데이터의 변화율 계산
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    last_changes = df[feature_columns].pct_change().iloc[-1:].values
    
    # 다음날 변화율 예측
    pred_changes = model.predict(last_changes)
    
    # 실제 가격으로 변환
    last_prices = df[['Open', 'High', 'Low', 'Close']].iloc[-1]
    predicted_prices = last_prices * (1 + pred_changes[0])
    
    # 결과 데이터프레임 생성
    next_date = pd.to_datetime(df['Date'].iloc[-1]) + pd.Timedelta(days=1)
    pred_df = pd.DataFrame([predicted_prices], 
                          columns=['Pred_Open', 'Pred_High', 'Pred_Low', 'Pred_Close'],
                          index=[next_date])
    
    return pred_df
  
def predict_next_day(df, model):
    """
    다음 거래일 주가 예측
    """
    # 마지막 거래일의 데이터
    last_day = df.iloc[-1]
    
    # 전일 대비 변화율 계산
    prev_day = df.iloc[-2]
    changes = [(last_day[col] - prev_day[col])/prev_day[col] for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
    changes = np.array(changes).reshape(1, -1)
    
    # 다음날 변화율 예측
    pred_changes = model.predict(changes)[0]
    
    # 예측 가격 계산
    next_prices = {
        'Pred_Open': last_day['Open'] * (1 + pred_changes[0]),
        'Pred_High': last_day['High'] * (1 + pred_changes[1]),
        'Pred_Low': last_day['Low'] * (1 + pred_changes[2]),
        'Pred_Close': last_day['Close'] * (1 + pred_changes[3])
    }
    
    # 다음 거래일 날짜
    next_date = pd.to_datetime(last_day['Date']) + pd.Timedelta(days=1)
    
    # 결과를 데이터프레임으로 변환
    pred_df = pd.DataFrame([next_prices], index=[next_date])
    
    return pred_df

def evaluate_predictions(y_true, y_pred, scalers):
    """
    예측 결과 평가
    """
    _, scaler_y = scalers
    
    # 스케일링된 데이터를 원래 스케일로 변환
    y_true = scaler_y.inverse_transform(y_true)
    y_pred = scaler_y.inverse_transform(y_pred)
    
    # 각 가격 유형별 RMSE 계산
    price_types = ['Open', 'High', 'Low', 'Close']
    metrics = {}
    
    for i, price_type in enumerate(price_types):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i])) * 100
        
        metrics[price_type] = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    return metrics

def run_stock_prediction(df):
    try:
        print("데이터 준비 중...")
        (X_train, X_test, y_train, y_test), last_data = prepare_ml_data(df)
        
        print("모델 학습 중...")
        model = train_stock_model(X_train, y_train)
        
        print("예측 수행 중...")
        y_pred = model.predict(X_test)
        
        # 성능 평가
        metrics = {}
        price_types = ['Open', 'High', 'Low', 'Close']
        for i, price_type in enumerate(price_types):
            last_prices = last_data[price_type].values[:-1]
            y_true_prices = last_prices[-len(y_test):] * (1 + y_test[:, i])
            y_pred_prices = last_prices[-len(y_test):] * (1 + y_pred[:, i])
            
            rmse = np.sqrt(mean_squared_error(y_true_prices, y_pred_prices))
            mae = mean_absolute_error(y_true_prices, y_pred_prices)
            mape = np.mean(np.abs((y_true_prices - y_pred_prices) / y_true_prices)) * 100
            
            metrics[price_type] = {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            }
        
        # 다음 거래일 예측
        next_day_pred = predict_next_day(df, model)
        
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        
        # 예측 신뢰도 계산
        confidence = 1 - np.mean([m['MAPE']/100 for m in metrics.values()])
        
        print(f"\n예측 신뢰도: {confidence:.2%}")
        print("\n=== 최근 실제 가격 ===")
        print(df.tail(1)[['Date', 'Open', 'High', 'Low', 'Close']])
        
        return {
            'model': model,
            'metrics': metrics,
            'next_day_prediction': next_day_pred,
            'feature_importance': feature_importance,
            'confidence': confidence
        }
        
    except Exception as e:
        print(f"예측 중 오류 발생: {str(e)}")
        raise



local_path = '/home/mynspluto/Project/stock-prediction/stock-ml'

df = pd.read_json(f"{local_path}/^IXIC.json")
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] >= '2024-12-10'].copy()


print("====데이터=====")
print(df[['Date', 'Close', 'Volume']])


(X_train, X_test, y_train, y_test), last_data = prepare_ml_data(df)
print("====학습 X====")
print(X_train)
print("====학습 Y====")
print(y_train)

print("====테스트 X====")
print(X_test)
print("====테스트 Y====")
print(y_test)


print("모델 학습 중...")
model = train_stock_model(X_train, y_train)

print("예측 수행 중...")
y_pred = model.predict(X_test)

# 성능 평가
# metrics = {}
# price_types = ['Open', 'High', 'Low', 'Close']
# for i, price_type in enumerate(price_types):
#     last_prices = last_data[price_type].values[:-1]
#     y_true_prices = last_prices[-len(y_test):] * (1 + y_test[:, i])
#     y_pred_prices = last_prices[-len(y_test):] * (1 + y_pred[:, i])
    
#     rmse = np.sqrt(mean_squared_error(y_true_prices, y_pred_prices))
#     mae = mean_absolute_error(y_true_prices, y_pred_prices)
#     mape = np.mean(np.abs((y_true_prices - y_pred_prices) / y_true_prices)) * 100
    
#     metrics[price_type] = {
#         'RMSE': rmse,
#         'MAE': mae,
#         'MAPE': mape
#     }

# # 다음 거래일 예측
# next_day_pred = predict_next_day(df, model)

# feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
# feature_importance = dict(zip(feature_columns, model.feature_importances_))

# # 예측 신뢰도 계산
# confidence = 1 - np.mean([m['MAPE']/100 for m in metrics.values()])

# print(f"\n예측 신뢰도: {confidence:.2%}")
# print("\n=== 최근 실제 가격 ===")
# print(df.tail(1)[['Date', 'Open', 'High', 'Low', 'Close']])

# prediction_results =  {
#     'model': model,
#     'metrics': metrics,
#     'next_day_prediction': next_day_pred,
#     'feature_importance': feature_importance,
#     'confidence': confidence
# }

# # 결과 출력
# print("\n=== 예측 성능 평가 ===")
# for price_type, metrics in prediction_results['metrics'].items():
#     print(f"\n{price_type}:")
#     for metric_name, value in metrics.items():
#         print(f"{metric_name}: {value:.2f}")

# print("\n=== 다음 거래일 예측 ===")
# print(prediction_results['next_day_prediction'])

# print("\n=== 특징 중요도 (상위 5개) ===")
# importance = sorted(prediction_results['feature_importance'].items(), 
#                    key=lambda x: x[1], reverse=True)[:5]
# for feature, score in importance:
#     print(f"{feature}: {score:.4f}")