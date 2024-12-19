# rm -rf ~/airflow/dags
# rm -rf ~/airflow/logs
# mkdir -p ~/airflow/dags

# #cp -r ./airflow-local/dags/* ~/airflow/dags
# cp ./airflow-local/dags/3.py ~/airflow/dags/3.py

# pwd

# python3 -m venv ./airflow-local/airflow_env
# source ./airflow-local/airflow_env/bin/activate
# pip install -r ./airflow-local/requirements.txt
# airflow standalone

./airflow-local-stop.sh
./airflow-local-start.sh