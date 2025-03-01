#!/usr/bin/env bash

git pull && \
minikube delete && \
./ec2-minikube-start.sh && \
./ec2-hadoop-start.sh && \
./ec2-kafka-start.sh && \
./ec2-airflow-k8s-start.sh && \
./ec2-server-k8s-start.sh && \
./ec2-next-k8s-start.sh