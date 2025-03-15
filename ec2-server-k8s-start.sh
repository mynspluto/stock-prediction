kubectl create namespace fastapi
kubectl config set-context --current --namespace=fastapi

eval $(minikube -p minikube docker-env)
docker build -t api-server:latest -f ./api-server/ec2-Dockerfile ./api-server

kubectl apply -f api-server/dep.yml -f api-server/svc.yml -f api-server/ingress.yml
kubectl rollout restart deployment/stock-prediction-api
kubectl get deployments
kubectl get pods