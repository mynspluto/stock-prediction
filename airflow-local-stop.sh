# helm uninstall airflow -n airflow
# kubectl delete all --all -n airflow
# kubectl delete namespace airflow

# Airflow 관련 프로세스 찾기
airflow_processes=$(ps aux | grep 'airflow' | grep -v 'grep' | awk '{print $2}')

if [ -z "$airflow_processes" ]; then
  echo "No Airflow processes found."
  exit 0
fi

# 프로세스 종료
echo "Found Airflow processes: $airflow_processes"
for pid in $airflow_processes; do
  echo "Killing process $pid..."
  kill -9 $pid
done