#!/bin/zsh
#!/bin/zsh
kubectl config set-context --current --namespace=fastapi

kubectl delete all --all -n fastapi
kubectl delete -f ./api-server/dep.yml -n fastapi