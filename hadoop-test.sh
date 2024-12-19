curl -i 'http://localhost:9870/webhdfs/v1/?op=LISTSTATUS'
curl -i -X PUT 'http://localhost:9870/webhdfs/v1/airflow/test_data?op=MKDIRS'
curl -i -X PUT -T ./hadoop/test_file.txt 'http://localhost:9864/webhdfs/v1/airflow/test_data/test_file.txt?op=CREATE&namenoderpcaddress=localhost:9000&overwrite=true'
