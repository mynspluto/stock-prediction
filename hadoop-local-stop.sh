cd /home/mynspluto/hadoop-3.4.1
sbin/stop-all.sh

rm -rf /tmp/hadoop-*
rm -rf /home/mynspluto/hadoop-3.4.1/logs/*
rm -rf $HADOOP_HOME/data/datanode  # 데이터노드 데이터 디렉토리