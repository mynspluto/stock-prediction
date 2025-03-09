# 코드 업데이트 (GitHub Actions에서 이미 처리되었을 수 있음)
cd ~/stock-prediction
git pull

# 네임스페이스 존재 확인 및 생성
kubectl get namespace web > /dev/null 2>&1 || kubectl create namespace web

# 현재 컨텍스트의 네임스페이스 설정
kubectl config set-context --current --namespace=web

# minikube에서 로컬 Docker 환경 사용
eval $(minikube -p minikube docker-env) # unset DOCKER_HOST

# 이미지 태그에 타임스탬프 추가하여 고유한 빌드 생성
TIMESTAMP=$(date +%Y%m%d%H%M%S)
IMAGE_TAG="nextjs:${TIMESTAMP}"

# Docker 이미지 빌드
echo "Building Docker image: $IMAGE_TAG"
docker build -t $IMAGE_TAG -f ./web/ec2-Dockerfile ./web

# deployment.yaml 파일에 새 이미지 태그 적용
sed -i "s|image: nextjs:.*|image: $IMAGE_TAG|g" ./web/dep.yml

# 배포 적용 - 이미 존재하면 자동으로 롤링 업데이트 수행
kubectl apply -f ./web/dep.yml -n web

# Pod가 Running 상태가 될 때까지 대기
echo "Waiting for new pods to be ready..."
kubectl rollout status deployment/nextjs -n web --timeout=300s

# 현재 실행 중인 port-forward 프로세스 찾기
PF_PID=$(ps aux | grep "kubectl port-forward.*nextjs" | grep -v grep | awk '{print $2}')

# 포트 포워딩이 이미 실행 중이면 종료
if [ ! -z "$PF_PID" ]; then
  echo "Stopping existing port-forward (PID: $PF_PID)"
  kill $PF_PID
  sleep 2
fi

# 새로운 포트 포워딩 설정
echo "Starting port forwarding for nextjs pod on port 3000"
nohup kubectl port-forward --address 0.0.0.0 -n web pod/$(kubectl get pods -n web -l app=nextjs -o jsonpath='{.items[0].metadata.name}') 3000:3000 > nextjs-portforward.log 2>&1 &

echo "Deployment completed successfully"