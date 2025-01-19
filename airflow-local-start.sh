#!/usr/bin/env zsh

rm -rf ~/airflow/dags
rm -rf ~/airflow/logs
mkdir -p ~/airflow/dags
mkdir -p ~/airflow/dags/mapreduce

cp ./airflow/dags/download_upload_stock_data.py ~/airflow/dags/download_upload_stock_data.py
cp ./airflow/dags/update_stock_prediction_model.py ~/airflow/dags/update_stock_prediction_model.py
cp ./airflow/dags/mapreduce/stock_mapper.py ~/airflow/dags/mapreduce/stock_mapper.py
cp ./airflow/dags/mapreduce/stock_reducer.py ~/airflow/dags/mapreduce/stock_reducer.py

python3.12 -m venv ./airflow/airflow_env
source ./airflow/airflow_env/bin/activate
pip install -r ./airflow/requirements.txt
export AIRFLOW_HOME=~/airflow
export AIRFLOW_ENV=local
airflow standalone
