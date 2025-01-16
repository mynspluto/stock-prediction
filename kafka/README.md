# 실행

cd ../all-in-one
./start-minikube.sh
cd ../kafka-native-k8s
kubectl apply -f dep.yml
kubectl apply -f svc.yml

./consume.sh
curl http://localhost:8000/produce/hello
{"status":"success","message":"Message sent to Kafka: hello"}%

# 테스트

cd /home/mynspluto/다운로드/kafka_2.13-3.9.0/bin
./kafka-topics.sh --create --topic test-topic --bootstrap-server localhost:9092
./kafka-topics.sh --bootstrap-server localhost:9092 --list

# 에러

minikube 재시작 or
svc가 존재하는 상태에서 kubectl apply -f dep.yml
pod crashloopBackOff
=> svc지우고 deploy 지우고 다시 dep.yml svc.yml 적용시 정상작동

kubectl logs kafka-7ddb85dc4c-2qctv
===> User
uid=1000(appuser) gid=1000(appuser) groups=1000(appuser)
===> Setting default values of environment variables if not already set.
CLUSTER_ID not set. Setting it to default value: "5L6g3nShT-eMCtK--X86sw"
===> Configuring ...
===> Launching ...
log4j:ERROR Could not read configuration file from URL [file:/opt/kafka/config/tools-log4j.properties]. java.io.FileNotFoundException: /opt/kafka/config/tools-log4j.properties (No such file or directory) at java.base@21.0.2/java.io.FileInputStream.open0(Native Method) at java.base@21.0.2/java.io.FileInputStream.open(FileInputStream.java:213) at java.base@21.0.2/java.io.FileInputStream.<init>(FileInputStream.java:152) at java.base@21.0.2/java.io.FileInputStream.<init>(FileInputStream.java:106) at java.base@21.0.2/sun.net.www.protocol.file.FileURLConnection.connect(FileURLConnection.java:84) at java.base@21.0.2/sun.net.www.protocol.file.FileURLConnection.getInputStream(FileURLConnection.java:180) at org.apache.log4j.PropertyConfigurator.doConfigure(PropertyConfigurator.java:532) at org.apache.log4j.helpers.OptionConverter.selectAndConfigure(OptionConverter.java:485) at org.apache.log4j.LogManager.<clinit>(LogManager.java:115) at org.slf4j.impl.Reload4jLoggerFactory.<init>(Reload4jLoggerFactory.java:67) at org.slf4j.impl.StaticLoggerBinder.<init>(StaticLoggerBinder.java:72) at org.slf4j.impl.StaticLoggerBinder.<clinit>(StaticLoggerBinder.java:45) at org.slf4j.LoggerFactory.bind(LoggerFactory.java:150) at org.slf4j.LoggerFactory.performInitialization(LoggerFactory.java:124) at org.slf4j.LoggerFactory.getILoggerFactory(LoggerFactory.java:417) at org.slf4j.LoggerFactory.getLogger(LoggerFactory.java:362) at com.typesafe.scalalogging.Logger$.apply(Logger.scala:31) at kafka.utils.Log4jControllerRegistration$.<clinit>(Logging.scala:25) at kafka.docker.KafkaDockerWrapper$.<clinit>(KafkaDockerWrapper.scala:29) at kafka.docker.KafkaDockerWrapper.main(KafkaDockerWrapper.scala) at java.base@21.0.2/java.lang.invoke.LambdaForm$DMH/sa346b79c.invokeStaticInit(LambdaForm$DMH) log4j:ERROR Ignoring configuration file [file:/opt/kafka/config/tools-log4j.properties]. log4j:WARN No appenders could be found for logger (kafka.utils.Log4jControllerRegistration$). log4j:WARN Please initialize the log4j system properly. log4j:WARN See http://logging.apache.org/log4j/1.2/faq.html#noconfig for more info. Exception in thread "main" org.apache.kafka.common.config.ConfigException: Missing required configuration `zookeeper.connect` which has no default value. at kafka.server.KafkaConfig.validateValues(KafkaConfig.scala:1232) at kafka.server.KafkaConfig.<init>(KafkaConfig.scala:1223) at kafka.server.KafkaConfig.<init>(KafkaConfig.scala:545) at kafka.tools.StorageTool$.$anonfun$execute$1(StorageTool.scala:72) at scala.Option.flatMap(Option.scala:283) at kafka.tools.StorageTool$.execute(StorageTool.scala:72) at kafka.tools.StorageTool$.main(StorageTool.scala:53) at kafka.docker.KafkaDockerWrapper$.main(KafkaDockerWrapper.scala:48) at kafka.docker.KafkaDockerWrapper.main(KafkaDockerWrapper.scala) at java.base@21.0.2/java.lang.invoke.LambdaForm$DMH/sa346b79c.invokeStaticInit(LambdaForm$DMH)
