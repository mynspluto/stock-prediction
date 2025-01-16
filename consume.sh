#!/usr/bin/env zsh

# venv 생성
python3.8 -m venv ./kafka/kafka_env

# kafka_env 활성화 (airflow_env가 아님)
source ./kafka/kafka_env/bin/activate

# 패키지 설치
pip install kafka-python

# 스크립트 실행
python3.8 ./kafka/consume.py

# 테스트
# ./consume.sh
# curl http://localhost:8000/produce/hello
# {"status":"success","message":"Message sent to Kafka: hello"}%