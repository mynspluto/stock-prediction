# stock_mapper.py
#!/usr/bin/env python3
import sys
import json

def map_stock_data():
    """
    주식 데이터를 읽어서 (날짜, 데이터) 형태로 매핑
    입력: JSON 형식의 주식 데이터
    출력: 탭으로 구분된 (날짜, 데이터) 쌍
    """
    for line in sys.stdin:
        try:
            # 입력 라인이 JSON 배열인 경우 처리
            if line.strip().startswith('['):
                records = json.loads(line)
                for record in records:
                    output_value = {
                        'Open': record['Open'],
                        'High': record['High'],
                        'Low': record['Low'],
                        'Close': record['Close'],
                        'Volume': record['Volume']
                    }
                    print(f"{record['Date']}\t{json.dumps(output_value)}")
            else:
                # 단일 JSON 객체인 경우
                record = json.loads(line.strip())
                output_value = {
                    'Open': record['Open'],
                    'High': record['High'],
                    'Low': record['Low'],
                    'Close': record['Close'],
                    'Volume': record['Volume']
                }
                print(f"{record['Date']}\t{json.dumps(output_value)}")
        except Exception as e:
            sys.stderr.write(f"Error processing record: {str(e)}\n")
            continue

if __name__ == '__main__':
    map_stock_data()