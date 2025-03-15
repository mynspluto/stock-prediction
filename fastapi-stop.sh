#!/bin/zsh
kubectl config set-context --current --namespace=fastapi

kubectl delete all --all -n fastapi
kubectl delete -f ./api-server/dep.yml -n fastapi
kubectl delete -f ./api-server/svc.yml -n fastapi
kubectl delete -f ./api-server/ingress.yml -n fastapi