mkdir -p /home/mynspluto/hadoop_data/hdfs/namenode
mkdir -p /home/mynspluto/hadoop_data/hdfs/datanode
#hdfs namenode -format
/home/mynspluto/hadoop-3.4.1/sbin/start-all.sh

sleep 20
./hadoop-test.sh