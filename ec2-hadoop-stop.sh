cd /home/ec2-user/hadoop-3.4.1
sbin/stop-all.sh

# 남은 프로세스 확인 및 종료
for PROC in `jps | grep -E "NameNode|DataNode|NodeManager|ResourceManager|SecondaryNameNode" | cut -d " " -f 1`
do
    kill -9 $PROC
done

rm -rf /tmp/hadoop-*
rm -rf /home/ec2-user/hadoop-3.4.1/logs/*
rm -rf $HADOOP_HOME/data/datanode  # 데이터노드 데이터 디렉토리