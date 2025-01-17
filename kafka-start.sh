#!/usr/bin/env zsh
minikube start
kubectl create namespace kafka
kubectl config set-context --current --namespace=kafka

kubectl apply -f ./kafka/dep.yml
sleep 10
kubectl wait --for=jsonpath='{.status.phase}'=Running --timeout=120s pod/kafka-0
sleep 3
kubectl apply -f ./kafka/svc.yml
sleep 2
kubectl port-forward svc/kafka-service 9092:9092 &
sleep 1

cd /home/mynspluto/kafka_2.12-3.9.0/bin
./kafka-topics.sh --create --topic test-topic --bootstrap-server localhost:9092
./kafka-topics.sh --bootstrap-server localhost:9092 --list
