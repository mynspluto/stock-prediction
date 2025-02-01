$HADOOP_HOME/bin/hdfs namenode -format
$HADOOP_HOME/sbin/start-all.sh
#$HADOOP_HOME/sbin/stop-all.sh
sleep 20
./ec2-hadoop-test.sh