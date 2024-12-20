에어플로우 + 하둡 오퍼레이터 따로 없고
python hadoop mapreduce <- 구글에 검색하여 참고

하둡의 노드
네임노드
데이터노드
클라이언트노드

subprocess로 hadoop 명령어쓸려면
코드가 실행되는 컴퓨터는 클라이언트 노드로 하둡 클러스터에 연결되어있어야함
클라이언트노드 설정
<configuration>
<property>
<name>fs.defaultFS</name>
<value>hdfs://namenode-hostname:9000</value>
</property>
</configuration>

데이터노드 설정
<configuration>

<!-- 네임노드 URL -->
<property>
<name>fs.defaultFS</name>
<value>hdfs://namenode-hostname:9000</value>
</property>

    <!-- 데이터노드 데이터 저장 디렉토리 -->
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>/path/to/hadoop/data</value>
    </property>

    <!-- 복제 개수 -->
    <property>
        <name>dfs.replication</name>
        <value>3</value>
    </property>

    <!-- 데이터노드가 네임노드에 등록할 자신의 주소 -->
    <property>
        <name>dfs.datanode.address</name>
        <value>datanode-hostname:50010</value>
    </property>

    <!-- 데이터노드 웹 UI 포트 -->
    <property>
        <name>dfs.datanode.http.address</name>
        <value>datanode-hostname:50075</value>
    </property>

</configuration>
