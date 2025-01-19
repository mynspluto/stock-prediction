#!/bin/bash
#docker exec -it namenode hadoop fs -chmod -R 777 /
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd /home/mynspluto/hadoop-3.4.1
bin/hadoop fs -chmod -R 777 /

cd "${SCRIPT_DIR}"

curl -i 'http://localhost:9870/webhdfs/v1/?op=LISTSTATUS'
curl -i -X PUT 'http://localhost:9870/webhdfs/v1/airflow/test_data?op=MKDIRS'
curl -i -X PUT -T test_file.txt 'http://mynspluto-pc:9864/webhdfs/v1/airflow/test_data/test_file.txt?op=CREATE&namenoderpcaddress=localhost:9000&createflag=&createparent=true&overwrite=true'
curl -i -X GET 'http://127.0.0.1:9864/webhdfs/v1/airflow/test_data/test_file.txt?op=OPEN&namenoderpcaddress=localhost:9000&offset=0'

# minikube ssh
# curl -i 'http://host.minikube.internal:9870/webhdfs/v1/?op=LISTSTATUS'
# curl -i -X PUT 'http://host.minikube.internal:9870/webhdfs/v1/airflow/test_data?op=MKDIRS'
# curl -i -X PUT -T kic.txt 'http://host.minikube.internal:9870/webhdfs/v1/airflow/test_data/test_file.txt?op=CREATE&namenoderpcaddress=localhost:9000&createflag=&createparent=true&overwrite=true'
curl -i -X PUT -T kic.txt 'http://mynspluto-pc:9864/webhdfs/v1/airflow/test_data/test_file.txt?op=CREATE&namenoderpcaddress=localhost:9000&createflag=&createparent=true&overwrite=true&user.name=hadoop'
curl -i -X PUT -T kic.txtㅌㅇㅇ 'http://mynspluto-pc:9864/webhdfs/v1/stock-history/%5EIXIC/monthly/2023-03.json?op=CREATE&user.name=hadoop&namenoderpcaddress=localhost:9000&createflag=&createparent=true&overwrite=true&user.name=hadoop'

# curl -i -X GET 'http://host.minikube.internal:9864/webhdfs/v1/airflow/test_data/test_file.txt?op=OPEN&namenoderpcaddress=localhost:9000&offset=0'
asd
