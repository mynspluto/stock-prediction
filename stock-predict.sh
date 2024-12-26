#!/bin/zsh

# python3.12 -m venv ./stock-ml/stock_predict_env
source ./stock-ml/stock_predict_env/bin/activate
# pip install --upgrade pip
# #pip install -r ./airflow-local/requirements.txt
# pip install numpy pandas scikit-learn yfinance requests hdfs apache-airflow
python3.12 ./stock-ml/stock-predict.py