# stock_reducer.py
#!/usr/bin/env python3
import sys
import json

def reduce_stock_data():
    """
    같은 날짜의 데이터를 하나로 병합
    입력: 정렬된 (날짜, 데이터) 쌍
    출력: JSON 형식의 통합된 레코드
    """
    current_date = None
    current_data = None

    for line in sys.stdin:
        try:
            # 입력 라인을 날짜와 데이터로 분리
            date, data_str = line.strip().split('\t')
            data = json.loads(data_str)

            if current_date == date:
                # 같은 날짜의 데이터는 첫 번째 것만 유지
                continue
            else:
                # 새로운 날짜를 만나면 이전 데이터 출력
                if current_date:
                    result = {'Date': current_date}
                    result.update(current_data)
                    print(json.dumps(result))

                current_date = date
                current_data = data

        except Exception as e:
            sys.stderr.write(f"Error reducing record: {str(e)}\n")
            continue

    # 마지막 레코드 출력
    if current_date:
        result = {'Date': current_date}
        result.update(current_data)
        print(json.dumps(result))

if __name__ == '__main__':
    reduce_stock_data()