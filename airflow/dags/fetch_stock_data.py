import yfinance as yf
import pandas as pd

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.hdfs.operators.hdfs import HdfsOperator

with DAG(
    "stock-predict",
    default_args={
        "owner": "airflow",
        "depend_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="Examples of Kafka Operators with HDFS",
    schedule=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:
    
    fetch_stock_task = PythonOperator(
        task_id="fetch_stock_data_task",
        python_callable=fetch_stock_data,
        op_args=[local_path, tickers],  # 로컬 경로와 티커 리스트를 인자로 전달
    )
    
    upload_to_hadoop_task = HdfsCreateFileOperator(
        task_id="upload_to_hdfs_task",
        file_path="/stock_data/^IXIC.csv",  # HDFS 경로
        data=open(f"{local_path}/^IXIC.csv", "r").read(),  # 파일 내용
        hdfs_conn_id="hdfs_default",  # Airflow HDFS Connection 설정
    )


    # 작업 의존성 설정
    fetch_stock_task >> upload_to_hadoop_task
