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

# Pod가 Running 상태가 될 때까지 대기
echo "Waiting for nextjs pod to be ready..."
kubectl wait --for=condition=Ready pod -l app=nextjs -n web --timeout=300s

# 포트 포워딩 설정
nohup kubectl port-forward --address 0.0.0.0 -n web pod/nextjs 3000:3000 > nextjs-portforward.log 2>&1 &

echo "Port forwarding started for nextjs pod on port 3000"
