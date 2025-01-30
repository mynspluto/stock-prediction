#!/bin/zsh

kubectl create namespace fastapi
kubectl config set-context --current --namespace=fastapi

eval $(minikube -p minikube docker-env)
# unset DOCKER_HOST
docker build -t api-server:latest -f ./api-server/Dockerfile ./api-server

kubectl apply -f ./api-server/dep.yml -n fastapi