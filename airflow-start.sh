#!/usr/bin/env zsh

rm -rf ~/airflow/dags
rm -rf ~/airflow/logs
mkdir -p ~/airflow/dags

cp ./airflow/dags/fetch_stock_data.py ~/airflow/dags/fetch_stock_data.py

python3.12 -m venv ./airflow/airflow_env
source ./airflow/airflow_env/bin/activate
pip install -r ./airflow/requirements.txt
airflow standalone
