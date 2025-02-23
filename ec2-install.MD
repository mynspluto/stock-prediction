## ssh 연결

ssh -i "stock-ml-keypair.pem" ec2-user@18.190.148.99

## 설치

sudo yum install python3 python3-pip
sudo yum -y install krb5-server krb5-libs

curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

sudo yum install docker -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
newgrp docker

curl -LO https://download.java.net/openjdk/jdk11/ri/openjdk-11+28_linux-x64_bin.tar.gz && \
 tar -xzf openjdk-11+28_linux-x64_bin.tar.gz
sudo mv jdk-11 /opt/
rm openjdk-11+28_linux-x64_bin.tar.gz

sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64

sudo yum install git

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm list-remote
nvm install --lts

## 하둡 설치

curl -LO https://dlcdn.apache.org/hadoop/common/hadoop-3.4.1/hadoop-3.4.1.tar.gz
tar -xzf hadoop-3.4.1.tar.gz
rm hadoop-3.4.1.tar.gz

sudo vi ~/.bashrc
export JAVA_HOME=/opt/jdk-11
export HADOOP_HOME=/home/ec2-user/hadoop-3.4.1
export PATH=$PATH:$JAVA_HOME/bin
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

sudo vi $HADOOP_HOME/etc/hadoop/hadoop-env.sh
export JAVA_HOME=/opt/jdk-11
export HDFS_NAMENODE_USER=ec2-user
export HDFS_DATANODE_USER=ec2-user
export HDFS_SECONDARYNAMENODE_USER=ec2-user
export YARN_RESOURCEMANAGER_USER=ec2-user
export YARN_NODEMANAGER_USER=ec2-user

hadoop 실행시
Stopping nodemanagers
localhost: ec2-user@localhost: Permission denied (publickey,gssapi-keyex,gssapi-with-mic). 발생

ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa && \
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
chmod 600 ~/.ssh/authorized_keys

sudo vi ~/.ssh/config

Host localhost
(탭)StrictHostKeyChecking no
(탭)UserKnownHostsFile=/dev/null
ssh localhost

sudo mkdir -p $HADOOP_HOME/logs
sudo chown -R ec2-user:ec2-user $HADOOP_HOME
sudo chmod -R 755 $HADOOP_HOME/logs

sudo mkdir -p $HADOOP_HOME/tmp
sudo chown -R ec2-user:ec2-user $HADOOP_HOME/tmp
sudo chmod -R 755 $HADOOP_HOME/tmp

$HADOOP_HOME/bin/hdfs namenode -format
$HADOOP_HOME/sbin/start-all.sh

hdfs dfs -mkdir /test
echo "Hello Hadoop" > test.txt
hdfs dfs -put test.txt /test/
hdfs dfs -ls /
hdfs dfs -ls /test
hdfs dfs -cat /test/test.txt

## kafka 설치

curl -LO https://dlcdn.apache.org/kafka/3.9.0/kafka_2.12-3.9.0.tgz
tar -xzf kafka_2.12-3.9.0.tgz

## todo

//하둡 설치 실행
//hdfs path 등록
자바설치
hadoop env에 자바 설정
core-site, hdfs 설정

node 설치

/etc/hosts
host.minikube.internal 등 등록

github action
