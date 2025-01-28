mkdir -p ~/.minikube/files/etc
#ip addr show | grep inet | grep -v inet6 | grep -v 127.0.0.1
        # inet 192.168.0.11/24 brd 192.168.0.255 scope global dynamic noprefixroute wlx705dccf17662
        # inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0
        # inet 192.168.49.1/24 brd 192.168.49.255 scope global br-759caf2dff41
echo 192.168.49.1    mynspluto-pc > ~/.minikube/files/etc/hosts

minikube start --cpus 6 --memory 30000 --driver=docker
minikube addons enable metrics-server
minikube addons enable ingress
minikube addons enable ingress-dns
minikube docker-env
eval $(minikube -p minikube docker-env)
unset DOCKER_HOST

minikube addons list | grep ingress