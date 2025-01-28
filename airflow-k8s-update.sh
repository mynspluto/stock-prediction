# 이미지 변경해서 업데이트 잘 안되는 경우 minikube ssh => docker images로 호스트에서 업데이트한 이미지 적용됐는지 확인
# 안된 경우 호스트에서 eval $(minikube -p minikube docker-env)

# 이미지 변경 없이 현재 버전으로 재시작
# 하는 이유 도커 이미지가 빌드되어 적용(helm install)이 되어 컨테이너의 파일상에는 코드 수정이 적용되었는데 dag실행시 로직에는 반영이 안되는 경우 있음 => scheduler 껐다 켜야 적용됨
./airflow-k8s-start.sh
sleep 2

kubectl rollout restart statefulset airflow-worker -n airflow
kubectl rollout restart deployment airflow-scheduler -n airflow
kubectl rollout restart deployment airflow-webserver -n airflow

# 이미지 변경하여 재시작, 헬름으로 생성한경우는 helm upgrade로 업데이트하는 게 더 일반적인듯
# kubectl set image statefulset/airflow-worker *=mynspluto-airflow:latest -n airflow
# kubectl set image deployment/airflow-scheduler *=mynspluto-airflow:latest -n airflow
# kubectl set image deployment/airflow-webserver *=mynspluto-airflow:latest -n airflow

