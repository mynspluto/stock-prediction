kubectl config set-context --current --namespace=web

kubectl delete all --all -n web
kubectl delete -f ./web/dep.yml -n web