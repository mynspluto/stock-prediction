#!/usr/bin/env bash

kubectl delete -f ../kafka/svc.yml
kubectl delete -f ../kafka/dep.yml

# 모든 port-forward 프로세스 찾아서 종료
echo "Killing all kubectl port-forward processes..."
pkill -f "kubectl port-forward"

# 확실한 종료를 위해 추가 검증
for pid in $(pgrep -f "kubectl port-forward"); do
    kill -9 $pid 2>/dev/null || true
done

# 포트 사용 확인 및 정리
if lsof -i :9092 >/dev/null; then
    echo "Cleaning up port 9092..."
    sudo lsof -ti:9092 | xargs kill -9
fi

echo "Verifying no remaining processes..."
ps aux | grep "kubectl port-forward" | grep -v grep

#kubectl delete all --all -n kafka
#kubectl delete pvc --all -n kafka
#kubectl delete pv --all -n kafka
#docker system prune -a
