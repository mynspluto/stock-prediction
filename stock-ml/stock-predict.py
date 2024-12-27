
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json

# 데이터 로드 및 전처리
local_path = '/home/mynspluto/Project/stock-prediction/stock-ml'
df = pd.read_json(f"{local_path}/^IXIC.json")
df['Date'] = pd.to_datetime(df['Date'])

# 기본 특징 설정
feature_columns = ['Close', 'Volume']
target_columns = ['Close']

# 퍼센트 변화율 계산 및 이상치 처리
df_pct = df[feature_columns].pct_change()
df_pct = df_pct.replace([np.inf, -np.inf], np.nan)  # inf 값을 nan으로 변경
df_pct = df_pct.fillna(0)  # nan 값을 0으로 채움

# 극단값 제거 (예: 99 퍼센타일 이상의 값 제한)
for column in df_pct.columns:
    upper_limit = np.percentile(df_pct[column], 99)
    lower_limit = np.percentile(df_pct[column], 1)
    df_pct[column] = df_pct[column].clip(lower_limit, upper_limit)

# 입력 데이터와 타겟 데이터 준비
X = df_pct[feature_columns].values[:-1]  # 마지막 날 제외
y_pct = df_pct[target_columns].shift(-1).values[:-1]

# 데이터 품질 확인
print("X에 inf 있음:", np.any(np.isinf(X)))
print("X에 nan 있음:", np.any(np.isnan(X)))
print("y에 inf 있음:", np.any(np.isinf(y_pct)))
print("y에 nan 있음:", np.any(np.isnan(y_pct)))

# 학습/테스트 분할 (80:20 비율)
X_train, X_test, y_train, y_test = train_test_split(X, y_pct, test_size=0.2, random_state=42)

print("====학습 데이터 크기====")
print("X_train:", len(X_train))
print("y_train:", len(y_train))
print("X_test:", len(X_test))
print("y_test:", len(y_test))

# 모델 학습
print("\n모델 학습 중...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train.ravel())


print("예측 수행 중...")
y_pred = model.predict(X_test)

# 성능 평가
metrics = {}
price_types = ['Close']

for i, price_type in enumerate(price_types):
    # 마지막 가격 데이터 가져오기
    last_prices = df[price_type].values[:-1]
    
    # y_test와 y_pred의 퍼센트 변화율을 실제 가격으로 변환
    y_true_prices = last_prices[-len(y_test):] * (1 + y_test.ravel())  # y_test는 2D라면 ravel()로 1D 변환
    y_pred_prices = last_prices[-len(y_test):] * (1 + y_pred)  # y_pred는 이미 1D
    
    # RMSE, MAE, MAPE 계산
    rmse = np.sqrt(mean_squared_error(y_true_prices, y_pred_prices))
    mae = mean_absolute_error(y_true_prices, y_pred_prices)
    mape = np.mean(np.abs((y_true_prices - y_pred_prices) / y_true_prices)) * 100
    
    metrics[price_type] = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

# 결과 출력
print("==== 성능 평가 ====")
for price_type, metric_values in metrics.items():
    print(f"{price_type}:")
    for metric_name, value in metric_values.items():
        print(f"  {metric_name}: {value:.4f}")

# # 다음 거래일 예측
# 마지막 거래일 데이터
last_day = df.iloc[-1]

# 전일 대비 변화율 계산 (Close와 Volume만 포함)
prev_day = df.iloc[-2]
changes = [(last_day[col] - prev_day[col]) / prev_day[col] for col in ['Close', 'Volume']]
changes = np.array(changes).reshape(1, -1)  # 2D 배열로 변환 (모델 입력 요구사항 충족)

# 다음날 Close 가격 예측
pred_close_change = model.predict(changes)[0]  # 단일 값 반환
pred_close_price = last_day['Close'] * (1 + pred_close_change)

# 다음 거래일 날짜 계산
next_date = pd.to_datetime(last_day['Date']) + pd.Timedelta(days=1)

# 결과를 데이터프레임으로 변환
pred_df = pd.DataFrame({
    'Pred_Close': [pred_close_price]
}, index=[next_date])

# 예측 결과 출력
print("예측", pred_df)
