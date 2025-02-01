minikube start --driver=docker
minikube addons enable metrics-server
minikube addons enable ingress
minikube addons enable ingress-dns
minikube docker-env
eval $(minikube -p minikube docker-env)
unset DOCKER_HOST

minikube addons list | grep ingress