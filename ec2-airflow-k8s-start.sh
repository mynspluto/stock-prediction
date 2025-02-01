#!/usr/bin/env bash

kubectl create namespace airflow
kubectl config set-context --current --namespace=airflow

eval $(minikube -p minikube docker-env)
# unset DOCKER_HOST
docker build -t mynspluto-airflow:latest -f ./airflow/ec2-Dockerfile ./airflow

helm repo add airflow-stable https://airflow-helm.github.io/charts
helm repo add apache-airflow https://airflow.apache.org
helm repo update
helm upgrade --install airflow apache-airflow/airflow -n airflow -f ./airflow/values.yml

sleep 5

kubectl patch svc airflow-webserver -n airflow -p '{"spec": {"type": "NodePort", "ports": [{"port": 8080, "targetPort": 8080, "nodePort": 31000}]}}'
#kubectl patch svc airflow-webserver -n airflow -p '{"spec": {"type": "ClusterIP", "ports": [{"port": 8080, "targetPort": 8080}]}}'

# PORT=8080

# # Check if the port is in use and get the PID
# PID=$(lsof -t -i :$PORT)

# # If the port is in use, kill the process
# if [ -n "$PID" ]; then
#     echo "Port $PORT is already in use by PID $PID. Terminating the process..."
#     kill $PID
#     sleep 2  # Wait for the process to terminate
# fi

# sleep 10

# nohup kubectl port-forward svc/airflow-webserver $PORT:$PORT -n airflow > port-forward.log 2>&1 &