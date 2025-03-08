# 주식 가격 예측 파이프라인

- 개요

  - 과거 주식 가격을 바탕으로 미래의 주식 가격을 예측하는 파이프라인을 구축했습니다

- 목적
  - ETL 파이프라인 구축과 모델 학습을 구현하여 데이터 엔지니어링의 핵심 요소를 경험하기 위함입니다

## 주요 기능

- 데이터 수집
  - Airflow Dag로 등록된 수집 로직이 주기적으로 실행됩니다
  - 야후 파이낸스 API로 데이터를 수집합니다
- 데이터 전처리, 저장
  - 가격의 변화로 변화율을 추출합니다
  - 전체 데이터를 YYYY-MM(년-월) 단위로 파싱하여 파일을 분할하여 저장합니다
- 모델 학습, 저장
  - 분할된 파일을 하둡의 맵리듀스를 통해 로드합니다
  - 가격의 변화율을 기반으로 모델을 학습시킵니다
  - 이를 하둡을 통해 저장합니다
- 모델 로드, 예측
  - 저장된 모델을 로드합니다
  - 로드된 모델로 가격 예측 API를 구현합니다
  - API 리스폰스에 예측된 가격을 포함합니다
  - 예측된 가격과 실제 가격의 오차가 크면 카프카 메시지를 발송합니다

## 시스템 구성도

![시스템 구성도](./system_diagram.svg)

## 🛠️ 기술 스택

Airflow, Hadoop, Kafka, Kubernetes(Minikube), Docker, Next, FastAPI
