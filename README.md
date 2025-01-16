## airflow 설치

sudo apt update
sudo apt install python3 python3-pip

sudo apt install libkrb5-dev krb5-config

https://airflow.apache.org/docs/apache-airflow/stable/start.html

## hadoop 설치

https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html
ssh 설정
설정 ~/hadoop-3.4.1/etc/hadoop/core-site.xml, hdfs-site.xml 등 수정 필요
hadoop.env.sh 자바 11버전 이전으로

## minikube 설치

도커 설치, 권한 부여

kubectl 설치
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

minikube 설치

## helm 설치

curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

## fast api

기본 8000번 포트
http://localhost:8000/
http://localhost:8000/predict/^IXIC

## todo

250116 목
kafka + minikube
fast api에서 요청 받았을 때 kafka produce
airflow에서 이걸 consume

250117 금
fast api 요청 받았을 때 특정일 예측 결과와
특정일 분봉 데이터 비교
예측 결과 보다 1프로 이상 차이나면 produce
현재
1 장중인지
장중이라면 분봉데이터와 오늘 날자 예측 결과 비교
2 아닌지
return
차트 렌더

250118 토
airflow minikube + dag git sync

250119 일
hadoop 도커화

250123 ~250126 목금토일
배포
minikube
kafka

minikube
airflow

docker
hadoop

fastapi + ci/cd
next + ci/cd
