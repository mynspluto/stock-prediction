FROM apache/airflow:2.9.2

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    vim \
    python3-pip \
    net-tools \
    iputils-ping \
    curl \
    wget \
    libkrb5-dev \
    krb5-user \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Java 11 installation
RUN wget https://download.java.net/openjdk/jdk11/ri/openjdk-11+28_linux-x64_bin.tar.gz && \
    tar -xzf openjdk-11+28_linux-x64_bin.tar.gz && \
    mv jdk-11 /opt/ && \
    rm openjdk-11+28_linux-x64_bin.tar.gz

# Set Java environment variables
ENV JAVA_HOME=/opt/jdk-11
ENV PATH=$PATH:$JAVA_HOME/bin

# Hadoop 3.4.1 installation
RUN wget https://downloads.apache.org/hadoop/common/hadoop-3.4.1/hadoop-3.4.1.tar.gz && \
    tar -xzf hadoop-3.4.1.tar.gz && \
    mv hadoop-3.4.1 /opt/hadoop && \
    rm hadoop-3.4.1.tar.gz

# Set Hadoop environment variables
ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
COPY ./hadoop-config/core-site.xml $HADOOP_CONF_DIR/core-site.xml
COPY ./hadoop-config/hdfs-site.xml $HADOOP_CONF_DIR/hdfs-site.xml
ENV HADOOP_USER_NAME=hadoop

COPY requirements.txt /requirements.txt

USER airflow

ENV AIRFLOW_ENV=local

RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

USER root

COPY ./dags/download_upload_stock_data.py /opt/airflow/dags/download_upload_stock_data.py
COPY ./dags/update_stock_prediction_model.py /opt/airflow/dags/update_stock_prediction_model.py
COPY ./dags/lstm_version.py /opt/airflow/dags/lstm_version.py
COPY ./dags/mapreduce/stock_mapper.py /opt/airflow/dags/mapreduce/stock_mapper.py
COPY ./dags/mapreduce/stock_reducer.py /opt/airflow/dags/mapreduce/stock_reducer.py
RUN ls -l /opt/airflow/dags

RUN mkdir -p /opt/airflow/stock_data
RUN ls -l /opt/airflow/stock_data

RUN mkdir -p /opt/airflow/hi
RUN ls -l /opt/airflow

RUN chmod -R 777 /opt/airflow/dags/
RUN chmod -R 777 /opt/airflow/stock_data
RUN chmod -R 777 /opt/airflow

# Verify Java installation
RUN java -version