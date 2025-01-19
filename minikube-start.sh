minikube start --cpus 6 --memory 30000 --driver=docker
minikube addons enable metrics-server
minikube docker-env
eval $(minikube -p minikube docker-env)
unset DOCKER_HOST