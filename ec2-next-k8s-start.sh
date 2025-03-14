# 네임스페이스 생성
kubectl create namespace web

# 현재 컨텍스트의 네임스페이스 설정
kubectl config set-context --current --namespace=web

# minikube에서 로컬 Docker 환경 사용
eval $(minikube -p minikube docker-env) # unset DOCKER_HOST

# Docker 이미지 빌드
docker build -t nextjs:latest -f ./web/ec2-Dockerfile ./web

# 배포 적용
kubectl apply -f ./web/dep.yml -n web

kubectl rollout restart deployment nextjs -n web
kubectl rollout status deployment/nextjs -n web --timeout=300s

# Pod가 Running 상태가 될 때까지 대기
echo "Waiting for nextjs pod to be ready..."
kubectl wait --for=condition=Ready pod -l app=nextjs -n web --timeout=300s

kubectl apply -f ./web/svc.yml

minikube addons enable ingress
minikube addons enable ingress-dns

kubectl apply -f ./web/ec2-ingress.yml