# 설치

sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv

export AIRFLOW_HOME=~/airflow

python3 -m venv airflow_env
source airflow_env/bin/activate

pip install apache-airflow
pip install 'apache-airflow[postgres,celery]'

airflow db init

airflow users create \
 --username admin \
 --firstname Admin \
 --lastname User \
 --role Admin \
 --email admin@example.com

## 실행

source airflow_env/bin/activate
airflow standalone
~/airflow/dags에 dag추가
http://localhost:8080

## db 지우기

cd ~/airflow
rm airflow.db
