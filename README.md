## airflow + k8s(client), hadoop + host(server) 연결 이슈

host에서 hadoop server 실행중
airflow는 k8s에서 실행하여 hadoop client로 접속시 연결안됨

airflow 컨테이너 접속
cd /opt/airflow/dags/
hadoop jar /opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.4.1.jar \
 -files mapreduce/stock_mapper.py,mapreduce/stock_reducer.py \
 -mapper "python3 stock_mapper.py" \
 -reducer "python3 stock_reducer.py" \
-input /stock-history/^IXIC/monthly \
 -output /stock-history/^IXIC/combined_mapreduce

맨 처음에 response size가 크다고 안됐는데
hdfs dfs -ls /도 안되는 걸로 보아 연결 자체가 안된것으로 판단했으나

클라이언트 core-site
<property>
<name>fs.defaultFS</name>
<value>hdfs://host.minikube.internal:9000(원래 9870 이었는데 수정)</value>
</property>로 수정하니

Call From airflow-webserver-64d6c974fb-vz8w6/10.244.0.33 to host.minikube.internal:9000 failed on connection exception: java.net.ConnectException: Connection refused; For more details see: http://wiki.apache.org/hadoop/ConnectionRefused
커넥션이 안된다고 에러메시지 변경됨

즉 원래 연결이 되었던 것

연결이 되었으나 mapreduce 안된 원인 원인
9870은 웹포트임(dfs.namenode.http-address)
9000으로 연결해야함(fs.defaultFs 클라이언트가 접속시 사용할 주소)
근데 9000은 어떤 문제인지 호스트에서 lsof -i :9000으로 확인은 되나 연결이 안됨 방화벽이나 하둡 설정 문제로 추정
=>
서버측 fs.defaultFs는 localhost로 하면 외부에서 접근 불가 0.0.0.0:9000으로 교체 하니 연결은 되나 다른 에러 발생
Input path does not exist: hdfs://host.minikube.internal:9000/stock-history/^IXIC/monthly
Streaming Command Failed!
=> 실제로 파일이 없었음 파일 download dag 실행해서 채우고 다른 에러 발생 Permission denied: user=airflow, access=WRITE, inode="/stock-history/^IXIC":hadoop:supergroup:drwxr-xr-x
권한에러 해당 디렉토리에 777권한 주면 해결될것으로 예상

## airflow 설치

sudo apt update
sudo apt install python3 python3-pip

sudo apt install libkrb5-dev krb5-config

https://airflow.apache.org/docs/apache-airflow/stable/start.html

## hadoop 설치

https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html
ssh 설정
설정 ~/hadoop-3.4.1/etc/hadoop/core-site.xml, hdfs-site.xml 등 수정 필요
hadoop.env.sh 자바 11버전 이전으로

## minikube 설치

도커 설치, 권한 부여

kubectl 설치
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

minikube 설치

## airflow minikube 이슈

minikube ssh 안에서 mynspluto-pc 접근 가능
설정 =>
mkdir -p ~/.minikube/files/etc
#ip addr show | grep inet | grep -v inet6 | grep -v 127.0.0.1 # inet 192.168.0.11/24 brd 192.168.0.255 scope global dynamic noprefixroute wlx705dccf17662 # inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0 # inet 192.168.49.1/24 brd 192.168.49.255 scope global br-759caf2dff41
echo 192.168.49.1 mynspluto-pc > ~/.minikube/files/etc/hosts

airflow-webserver 컨테이너에서 mynspluto-pc 접근 불가능
client.upload 시 http://host.minikube.internal:9870는 접근이 되나
redirect 되는 http://mynspluto-pc:9870에 접근 불가능

해결 =>
def custom_getaddrinfo(host, port, *args, \*\*kwargs):
if host == 'mynspluto-pc' and ENVIRONMENT == 'kubernetes':
return socket.getaddrinfo('host.minikube.internal', port, *args, **kwargs)
return socket.\_getaddrinfo(host, port, \*args, **kwargs)

TODO
airflow-webserver 컨테이너안에서 subprocess로 호스트에서 실행중인 하둡 서버에 클라이언트로 접속하여
맵리듀스
airflow/Dockerfile 수정해서 구현 가능해보임
hadoop 설치하고
호스트 컴퓨터의 하둡 서버에 연결하도록 hdfs ,core-site xml 수정

## helm 설치

curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

## fast api

기본 8000번 포트
http://localhost:8000/
http://localhost:8000/predict/^IXIC

## todo

250116 목
kafka + minikube
fast api에서 요청 받았을 때 kafka produce
airflow에서 이걸 consume

250117 금
fast api 요청 받았을 때 특정일 예측 결과와
특정일 분봉 데이터 비교
예측 결과 보다 1프로 이상 차이나면 produce
현재
1 장중인지
장중이라면 분봉데이터와 오늘 날자 예측 결과 비교
2 아닌지
return
차트 렌더

250118 토
airflow minikube + dag git sync

250119 일
hadoop 도커화

250123 ~250126 목금토일
배포
minikube
kafka

minikube
airflow

docker
hadoop

fastapi + ci/cd
next + ci/cd
