#!/bin/bash
set -e

# 코드 업데이트
cd ~/stock-prediction
git stash
git pull
git stash pop || true

# 네임스페이스 확인 및 설정
kubectl get namespace web > /dev/null 2>&1 || kubectl create namespace web
kubectl config set-context --current --namespace=web

# minikube Docker 환경 설정
eval $(minikube -p minikube docker-env)

# 이미지 빌드
TIMESTAMP=$(date +%Y%m%d%H%M%S)
IMAGE_TAG="nextjs:${TIMESTAMP}"
echo "Building Docker image: $IMAGE_TAG"
docker build -t $IMAGE_TAG -f ./web/ec2-Dockerfile ./web

# deployment.yaml 파일 업데이트
sed -i "s|image: nextjs:.*|image: $IMAGE_TAG|g" ./web/dep.yml

# 현재 실행 중인 포트 포워딩 PID 찾기 (종료하지 않음)
OLD_PF_PID=$(pgrep -f "kubectl port-forward.*nextjs.*3000:3000" || echo "")
if [ -n "$OLD_PF_PID" ]; then
  echo "Current port-forwarding running with PID: $OLD_PF_PID"
else
  echo "No existing port-forwarding found"
fi

# 현재 사용 중인 포드 이름 저장
OLD_POD_NAME=$(kubectl get pods -n web -l app=nextjs -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
echo "Current active pod: $OLD_POD_NAME"

# 배포 적용
echo "Applying deployment..."
kubectl apply -f ./web/dep.yml -n web

# 새 포드가 준비될 때까지 대기
echo "Waiting for new pods to be ready..."
kubectl rollout status deployment/nextjs -n web --timeout=300s

# 새 포드 이름 가져오기
sleep 5 # 포드 스케줄링 및 이름 지정을 위한 짧은 대기
NEW_PODS=$(kubectl get pods -n web -l app=nextjs -o jsonpath='{.items[*].metadata.name}')
NEW_POD=""

# 이전 포드와 다른 새 포드 찾기
for pod in $NEW_PODS; do
  if [ "$pod" != "$OLD_POD_NAME" ]; then
    NEW_POD=$pod
    break
  fi
done

if [ -z "$NEW_POD" ]; then
  echo "Could not identify new pod, using first available"
  NEW_POD=$(echo $NEW_PODS | cut -d' ' -f1)
fi

echo "New pod for port-forwarding: $NEW_POD"

# 새 포드가 Running 상태인지 확인
POD_STATUS=$(kubectl get pod $NEW_POD -n web -o jsonpath='{.status.phase}')
echo "New pod status: $POD_STATUS"

if [ "$POD_STATUS" != "Running" ]; then
  echo "Waiting for pod to be in Running state..."
  kubectl wait --for=condition=ready pod/$NEW_POD -n web --timeout=60s
fi

mkdir -p ~/stock-prediction/log
# 새 포트 포워딩 시작 (다른 파일에 로그 남김)
echo "Starting new port forwarding for pod $NEW_POD"
nohup kubectl port-forward --address 0.0.0.0 -n web pod/$NEW_POD 3000:3000 > ./log/nextjs-portforward.log 2>&1 &
NEW_PF_PID=$!

# 새 포트 포워딩이 정상적으로 시작되었는지 확인
sleep 3
if ps -p $NEW_PF_PID > /dev/null; then
  echo "New port forwarding started successfully with PID: $NEW_PF_PID"
  
  # 이전 포트 포워딩이 있다면 종료
  if [ -n "$OLD_PF_PID" ]; then
    echo "Terminating old port forwarding (PID: $OLD_PF_PID) after successful transition"
    kill $OLD_PF_PID || true
  fi
  
  # 로그 파일 이름 변경
  mv nextjs-portforward-new.log nextjs-portforward.log
else
  echo "New port forwarding failed to start. Check log:"
  cat nextjs-portforward-new.log
  
  if [ -n "$OLD_PF_PID" ]; then
    echo "Keeping old port forwarding active"
  fi
  
  exit 1
fi

echo "Zero-downtime deployment completed successfully"