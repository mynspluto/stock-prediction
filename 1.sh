#!/bin/zsh

python3.12 -m venv ./airflow/airflow_env
source ./airflow/airflow_env/bin/activate
python3.12 ./airflow/dags/1.py