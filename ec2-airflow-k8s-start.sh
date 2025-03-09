#!/usr/bin/env bash

# 네임스페이스 생성 및 설정
kubectl create namespace airflow 2>/dev/null || true
kubectl config set-context --current --namespace=airflow

# EC2 환경에서는 minikube 관련 코드 제거
# eval $(minikube -p minikube docker-env)

# 타임스탬프를 이용한 고유 태그 생성
TIMESTAMP=$(date +%Y%m%d%H%M%S)
IMAGE_TAG="mynspluto-airflow:${TIMESTAMP}"

# Docker 이미지 빌드
echo "Building Docker image: ${IMAGE_TAG}"
eval $(minikube -p minikube docker-env)
docker build -t ${IMAGE_TAG} -f ./airflow/ec2-Dockerfile ./airflow

# values.yml 파일에 이미지 태그 업데이트 (필요한 경우)
# sed -i "s|image: mynspluto-airflow:.*|image: ${IMAGE_TAG}|g" ./airflow/values.yml

# Helm 차트 설치 또는 업그레이드
helm repo add airflow-stable https://airflow-helm.github.io/charts
helm repo add apache-airflow https://airflow.apache.org
helm repo update

echo "Deploying Airflow with Helm..."
helm upgrade --install airflow apache-airflow/airflow -n airflow -f ./airflow/values.yml

# Airflow 웹서버 Pod가 Ready 상태가 될 때까지 대기
echo "Waiting for Airflow webserver to be ready..."
kubectl wait --for=condition=ready pod -l component=webserver -n airflow --timeout=300s

# 포트 포워딩 설정
PORT=8080

# 기존 포트 포워딩 프로세스 확인 및 종료
PF_PID=$(ps aux | grep "kubectl port-forward.*airflow-webserver" | grep -v grep | awk '{print $2}')
if [ -n "$PF_PID" ]; then
    echo "Stopping existing port-forward (PID: $PF_PID)"
    kill $PF_PID
fi

# 포트가 사용 중인지 확인하고 사용 중이면 해당 프로세스 종료
PID=$(lsof -t -i :$PORT)
if [ -n "$PID" ]; then
    echo "Port $PORT is already in use by PID $PID. Terminating the process..."
    kill $PID
fi


nohup kubectl port-forward --address 0.0.0.0 -n airflow svc/airflow-webserver 8080:8080 > airflow-port-forward.log 2>&1 &