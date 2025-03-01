# 코드 업데이트 (GitHub Actions에서 이미 처리되었을 수 있음)
cd ~/stock-prediction
git pull
# 네임스페이스 생성
kubectl create namespace web || true

# 현재 컨텍스트의 네임스페이스 설정
kubectl config set-context --current --namespace=web

# minikube에서 로컬 Docker 환경 사용
eval $(minikube -p minikube docker-env)

# 새 이미지 태그 생성
IMAGE_TAG="nextjs:$(date +%Y%m%d%H%M%S)"

# Docker 이미지 빌드
docker build -t ${IMAGE_TAG} -f ./web/Dockerfile ./web

# 새 이미지 태그로 deployment.yaml 수정
sed -i "s|image: nextjs:.*|image: ${IMAGE_TAG}|g" ./k8s/deployment.yaml

# Kubernetes 리소스 적용
kubectl apply -f ./k8s/deployment.yaml -n web
kubectl apply -f ./k8s/service.yaml -n web

# 배포 완료 대기
kubectl rollout status deployment/nextjs -n web --timeout=300s

# NodePort 확인
NODE_PORT=$(kubectl get service nextjs-service -n web -o jsonpath='{.spec.ports[0].nodePort}')
MINIKUBE_IP=$(minikube ip)

echo "======================================"
echo "애플리케이션 접속 URL: http://${MINIKUBE_IP}:${NODE_PORT}"
echo "======================================"