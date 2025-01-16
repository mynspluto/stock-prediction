# 네임스페이스 생성 및 변경

kubectl create namespace kafka
kubectl config set-context --current --namespace=kafka

# 기존 요소 제거

kubectl delete -f platform-kraft.yml
kubectl delete all --all -n kafka (pv, pvc는 안지워짐)
kubectl delete pvc --all -n kafka
kubectl delete pv --all -n kafka
docker system prune -a

# 요소 생성, 서비스 포워딩

helm repo add confluentinc https://packages.confluent.io/helm
helm repo update
helm upgrade --install \
 confluent-operator confluentinc/confluent-for-kubernetes
kubectl get pods

kubectl apply -f platform-kraft.yml

kubectl port-forward controlcenter-0 9021:9021
127.0.0.1:9021 접속(컨트롤센터 웹)

kubectl port-forward svc/kafkarestproxy 8082:8082
curl -X GET -H "Accept: application/vnd.kafka.v2+json" \
 http://localhost:8082/topics

image pull error 뜨는 경우
docker pull confluentinc/cp-kafka-rest:7.6.1.arm64 이런식으로 수동으로 받은후
minikube image load confluentinc/cp-kafka-rest:7.6.1.arm64 하여 이미지 로드

crashback error 뜨는 경우
minikube delete
minikube start --memory 15976 (15976은 도커 데스크탑앱 설정에서 제한되는듯)

replica 1로 바꾸면 에러남
metadata.namespace바꿔도 에러나는듯
=> https://github.com/confluentinc/confluent-kubernetes-examples/blob/master/quickstart-deploy/confluent-platform-singlenode.yaml
싱글 모드 예제로 해결

todo
airflow로 하루에 한번 주가 데이터 수집
수집한 데이터 하둡으로 저장
저장했다고 카프카 메시지로 알림
쿠버네티스에 deployment로 등록된 카프카 소비자가 이를 감지
하둡으로 저장했던 파일을 스파크를 통해 로드
전처리를 하여 보조지표 등 필요한 데이터 파싱
학습
학습된 모델 하둡으로 저장

카프카 정상적으로 생성
elastic-0 생성 실패함
=> namespace 1.yml, 2.yml 수정한거,
하둡 에어플로우 때매 하드웨어 스펙 부족으로 실패 추측
https://docs.confluent.io/operator/current/co-quickstart.html

생성된 카프카 서비스 확인후
kafka-rest로 producer, consumer test
https://docs.confluent.io/platform/current/kafka-rest/index.html

에어플로우와 연동하여
consumer, producer test
https://airflow.apache.org/docs/apache-airflow-providers-apache-kafka/stable/_modules/tests/system/providers/apache/kafka/example_dag_hello_kafka.html

https://docs.confluent.io/kafka-clients/python/current/overview.html

# 카프카 토픽 미생성시 에러

- 클러스터 id 확인
  curl -X GET "http://localhost:8082/v3/clusters"

- test_1 topic 생성
  curl -X GET "http://localhost:8082/v3/clusters/28e637f6-5449-4e11-a5w/topics/test_1/configs"

# 리소스 사용량

NAME CPU(cores) MEMORY(bytes)  
confluent-operator-6df55d9796-zjsnn 3m 50Mi  
connect-0 18m 3296Mi  
controlcenter-0 318m 475Mi  
kafka-0 139m 1328Mi  
kafka-1 142m 1381Mi  
kafka-2 147m 1307Mi  
kafkarestproxy-0 3m 351Mi  
kraftcontroller-0 22m 613Mi  
kraftcontroller-1 22m 547Mi  
kraftcontroller-2 41m 623Mi  
ksqldb-0 161m 464Mi  
schemaregistry-0 37m 375Mi  
schemaregistry-1 41m 413Mi  
schemaregistry-2 38m 376Mi

# 에어플로우 + 하둡 + 카프카 리소스 사용량

NAME CPU(cores) CPU% MEMORY(bytes) MEMORY%
minikube 1275m 15% 19422Mi 80%

# 테스트

kubectl port-forward svc/kafkarestproxy 8082:8082

curl -X GET -H "Accept: application/vnd.kafka.v2+json" http://localhost:8082/topics
=> bad hostname 나오면 줄바꿈 때문
