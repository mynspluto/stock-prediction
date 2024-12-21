#!/bin/zsh

python3.12 -m venv ./airflow/airflow_env
source ./airflow/airflow_env/bin/activate
pip install --upgrade pip
#pip install -r ./airflow-local/requirements.txt
pip install numpy pandas scikit-learn yfinance requests hdfs apache-airflow
python3.12 ./airflow/dags/1.py