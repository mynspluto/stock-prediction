# consumer_service.py
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'test_1',
    bootstrap_servers='localhost:9092',
    group_id='my-group',
    auto_offset_reset='earliest'
)

def process_message(message):
    # 비즈니스 로직 처리
    # DB 저장, API 호출, 계산 등
    print(message)
    pass

for message in consumer:
    process_message(message)