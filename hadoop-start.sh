#docker-compose -f ./hadoop/docker-compose.yml --project-name=hadoop up -d
./hadoop-local-start.sh

sleep 20
./hadoop-test.sh