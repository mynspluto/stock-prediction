FROM apache/airflow:2.9.2

USER root

RUN apt-get update && \
    apt-get install -y \
    vim \
    python3-pip \
    net-tools \
    curl \
    libkrb5-dev \
    krb5-user \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

USER airflow

ENV AIRFLOW_ENV=kubernetes

RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

USER root

COPY ./dags/download_upload_stock_data.py /opt/airflow/dags/download_upload_stock_data.py
COPY ./dags/update_stock_prediction_model.py /opt/airflow/dags/update_stock_prediction_model.py
COPY ./dags/mapreduce/stock_mapper.py /opt/airflow/dags/mapreduce/stock_mapper.py
COPY ./dags/mapreduce/stock_reducer.py /opt/airflow/dags/mapreduce/stock_reducer.py
RUN ls -l /opt/airflow/dags

#ADD ~/airflow/dags/ /opt/airflow/dags/
RUN ls -l /opt/airflow/dags

RUN mkdir -p /opt/airflow/stock_data
RUN ls -l /opt/airflow/stock_data

RUN mkdir -p /opt/airflow/hi
RUN ls -l /opt/airflow

RUN chmod -R 777 /opt/airflow/dags/
RUN chmod -R 777 /opt/airflow/stock_data
RUN chmod -R 777 /opt/airflow