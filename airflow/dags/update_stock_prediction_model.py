import os
import yfinance as yf
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
from hdfs import InsecureClient
import subprocess
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


from airflow import DAG
from airflow.operators.python import PythonOperator

stock_data_path = '/home/mynspluto/Project/stock-prediction/airflow/data/stock-history'
hadoop_url = 'http://localhost:9870'
hdfs_path = "/stock-history"
hadoop_home = "/home/mynspluto/hadoop-3.4.1"
tickers = ['^IXIC']

def run_hadoop_mapreduce(tickers):
    """
    HDFS 클라이언트를 통해 하둡 스트리밍 맵리듀스 작업 실행
    """
    for ticker in tickers:

      input_path = f'{hdfs_path}/{ticker}'
      output_path = f'{hdfs_path}/{ticker}/combined_mapreduce'
    
      try:
        # Hadoop 홈 디렉토리 설정
        hadoop_bin = os.path.join(hadoop_home, 'bin/hadoop')
        
        # 현재 DAG 파일의 디렉토리 경로 가져오기
        dag_folder = os.path.dirname(os.path.abspath(__file__))
        mapper_path = os.path.join(dag_folder, 'mapreduce/stock_mapper.py')
        reducer_path = os.path.join(dag_folder, 'mapreduce/stock_reducer.py')
        
        # Delete output directory if it exists
        delete_command = [hadoop_bin, 'fs', '-rm', '-r', output_path]
        subprocess.run(delete_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Cleaned up existing output directory: {output_path}")
        
        # Hadoop 스트리밍 JAR 파일 경로
        streaming_jar = os.path.join(hadoop_home, 'share/hadoop/tools/lib/hadoop-streaming-*.jar')
        streaming_jar = os.popen(f'ls {streaming_jar}').read().strip()
        
        # MapReduce 작업 실행 명령어
        hadoop_command = [
            hadoop_bin, 'jar', streaming_jar,
            '-files', f'{mapper_path},{reducer_path}',
            '-mapper', 'python3 stock_mapper.py',
            '-reducer', 'python3 stock_reducer.py',
            '-input', input_path,
            '-output', output_path
        ]
        
        # MapReduce 작업 실행
        process = subprocess.Popen(
            hadoop_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 실행 결과 출력
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print("MapReduce job completed successfully")
            print(stdout.decode())
        else:
            print("MapReduce job failed")
            print(stderr.decode())
            
      except Exception as e:
          print(f"Error running MapReduce job: {str(e)}")
          raise

with DAG(
    "update_stock_prediction_model",
    default_args={
        "owner": "airflow",
        "depend_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="주가 데이터를 수집하여 월별로 분할하여 파일 저장",
    schedule=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:

    merge_stock_data_by_month_mapreduce = PythonOperator(
        task_id="merge_stock_data_by_month_mapreduce_task",
        python_callable=run_hadoop_mapreduce,
        op_args=[tickers],
    )
  
    
    merge_stock_data_by_month_mapreduce
