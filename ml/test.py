from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def fetchData(ticker, start=None, end=None):
    data = yf.download(ticker, start, end)
    return data

def addSMA(df):
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    return df

def addMACD(df):
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def plotChart(df):
    plt.figure(figsize=(14, 8))

    # 서브플롯 1: 가격 + 이동평균선
    plt.subplot(2, 1, 1)
    plt.plot(df['Close'], label='Close', color='black')
    plt.plot(df['SMA_5'], label='SMA 5', linestyle='--')
    plt.plot(df['SMA_20'], label='SMA 20', linestyle='--')
    plt.plot(df['SMA_60'], label='SMA 60', linestyle='--')
    plt.title('NASDAQ (^IXIC) Price and Moving Averages')
    plt.legend()
    plt.grid(True)

    # 서브플롯 2: MACD + Signal
    plt.subplot(2, 1, 2)
    plt.plot(df['MACD'], label='MACD', color='blue')
    plt.plot(df['Signal'], label='Signal', color='red', linestyle='--')
    plt.axhline(0, color='gray', linewidth=1, linestyle='--')
    plt.title('MACD')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

#ticker = '^IXIC'
ticker = 'QQQ'
#start='2022-01-01'
#end='2023-01-01'

#1. 데이터 가져오기
data = fetchData(ticker)
#data = fetchData(ticker, start, end)
print('fetchData\n', data)

#2. 특성 추가
data = addSMA(data)
data = addMACD(data)
print(data[['Close', 'SMA_5', 'SMA_20', 'SMA_60', 'MACD', 'Signal']].tail())
#plotChart(data)

#saveData(data)
#data = loadData('IXIC')
#data = adjust(data) 차분 등 계절성 데이터 조정, 정규화 등 모델에 맞게 조정

# 시계열 데이터를 학습할 수 있도록 아래와 같이 시퀀스화
# X = [
#     [Day 1, Day 2, ..., Day 60],  # 첫 번째 시퀀스
#     [Day 2, Day 3, ..., Day 61],  # 두 번째 시퀀스
#     ...,
#     [Day 940, Day 941, ..., Day 1000]  # 마지막 시퀀스
# ]

# y = [
#     Day 61,  # 첫 번째 시퀀스에 대한 목표값
#     Day 62,  # 두 번째 시퀀스에 대한 목표값
#     ...,
#     Day 1001  # 마지막 시퀀스에 대한 목표값
# ]

#[x_train, y_train, x_text, y_test] = split()
#model = train(data)
#evaluate(model, data)