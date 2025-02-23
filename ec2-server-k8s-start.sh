kubectl create namespace fastapi
kubectl config set-context --current --namespace=fastapi

eval $(minikube -p minikube docker-env)
# unset DOCKER_HOST
docker build -t api-server:latest -f ./api-server/ec2-Dockerfile ./api-server

# kubectl port-forward pod/stock-prediction-api 8000:8000
# nohup kubectl port-forward --address 0.0.0.0 -n fastapi pod/stock-prediction-api 8000:8000 > fastapi-portforward.log 2>&1 &

kubectl apply -f ./api-server/dep.yml -n fastapi

sleep 5

nohup kubectl port-forward --address 0.0.0.0 -n fastapi pod/stock-prediction-api 8000:8000 > fastapi-portforward.log 2>&1 &