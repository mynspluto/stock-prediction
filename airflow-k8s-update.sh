# 이미지 변경해서 업데이트 잘 안되는 경우 minikube ssh => docker images로 호스트에서 업데이트한 이미지 적용됐는지 확인
# 안된 경우 호스트에서 eval $(minikube -p minikube docker-env)

eval $(minikube -p minikube docker-env)
#unset DOCKER_HOST

# docker build -t mynspluto-airflow:latest -f ./airflow/Dockerfile ./airflow
# helm upgrade --install airflow apache-airflow/airflow -n airflow -f ./airflow/values.yml

PORT=8080

# Check if the port is in use and get the PID
PID=$(lsof -t -i :$PORT)

# If the port is in use, kill the process
if [ -n "$PID" ]; then
    echo "Port $PORT is already in use by PID $PID. Terminating the process..."
    kill $PID
    sleep 2  # Wait for the process to terminate
fi

sleep 10

nohup kubectl port-forward svc/airflow-webserver $PORT:$PORT -n airflow > port-forward.log 2>&1 &

# 이미지 변경 없이 현재 버전으로 재시작
# kubectl rollout restart statefulset airflow-worker -n airflow
# kubectl rollout restart deployment airflow-scheduler -n airflow
# kubectl rollout restart deployment airflow-webserver -n airflow

# 이미지 변경하여 재시작, 헬름으로 생성한경우는 helm upgrade로 업데이트하는 게 더 일반적인듯
# kubectl set image statefulset/airflow-worker *=mynspluto-airflow:latest -n airflow
# kubectl set image deployment/airflow-scheduler *=mynspluto-airflow:latest -n airflow
# kubectl set image deployment/airflow-webserver *=mynspluto-airflow:latest -n airflow

