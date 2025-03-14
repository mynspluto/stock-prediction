kubectl create namespace airflow
kubectl config set-context --current --namespace=airflow

eval $(minikube -p minikube docker-env)
# unset DOCKER_HOST
docker build --no-cache -t mynspluto-airflow:latest -f ./airflow/ec2-Dockerfile ./airflow

helm repo add airflow-stable https://airflow-helm.github.io/charts
helm repo add apache-airflow https://airflow.apache.org
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm install nginx-ingress ingress-nginx/ingress-nginx
helm upgrade --install airflow apache-airflow/airflow -n airflow -f ./airflow/values.yml

sleep 5

#kubectl patch svc airflow-webserver -n airflow -p '{"spec": {"type": "ClusterIP", "ports": [{"port": 8080, "targetPort": 8080}]}}'
kubectl patch svc airflow-webserver -n airflow -p '{"spec": {"type": "NodePort", "ports": [{"port": 8080, "targetPort": 8080, "nodePort": 30080}]}}'
minikube addons enable ingress
minikube addons enable ingress-dns

kubectl apply -f ./airflow/ec2-ingress.yml

# pod내에서 dag 파일 변경 되어도 dag 실행 후 Code탭 들어가서 확인시 적용 안되서 rollout해야 됨
kubectl rollout restart deployment airflow-webserver
kubectl rollout restart statefulset airflow-worker
kubectl rollout restart deployment airflow-scheduler