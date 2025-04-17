import json
import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()

# 특정 서비스의 API 키 가져오기
open_api_key = os.getenv("OPEN_API_KEY")

# 발급받은 API 키와 기본 URL 설정
api_key = open_api_key
base_url = 'https://api.odcloud.kr/api/gov24/v3/serviceList'

def fetch_data(page):
    params = {
        'serviceKey': api_key,
        'page': page,
        'perPage': 100,
    }
    try:
        response = requests.get(base_url, params=params)
        
        # 디버깅: 전체 응답 출력
        print("Response Status Code:", response.status_code)
        print("Response Headers:", response.headers)
        print("Response Content:", response.text)
        
        response.raise_for_status()
        data = response.json()
        
        print(f"Page {page}: {len(data.get('data', []))} items fetched")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Request Exception: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decoding Error: {e}")
        return None

# 모든 페이지에서 데이터를 수집하여 하나의 리스트에 저장
all_data = []
page = 1
max_retries = 3
retry_count = 0

while retry_count < max_retries:
    data = fetch_data(page)
    if data is None:
        retry_count += 1
        print(f"Retry attempt {retry_count}")
        time.sleep(2)  # 재시도 사이에 잠시 대기
        continue
    
    if not data.get('data'):
        break
    
    all_data.extend(data['data'])
    page += 1
    time.sleep(0.5)  # API 호출 사이의 간격 조절

# DataFrame 생성
if all_data:
    df = pd.DataFrame(all_data)
    
    # DataFrame 정보 출력
    print("\nDataFrame Information:")
    print(df.info())
    
    crawling_date = 20250327
    output_folder_path = r'C:\Users\r2com\Desktop\VSCode\bozo24'
    
    # 폴더 존재 확인 및 생성
    os.makedirs(output_folder_path, exist_ok=True)
    
    # CSV 파일로 저장
    csv_path = os.path.join(output_folder_path, f'{crawling_date}serviceList.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nCSV 파일 저장 완료: {csv_path}")
    
    # JSON 파일로 저장
    json_path = os.path.join(output_folder_path, f'{crawling_date}serviceList.json')
    df.to_json(json_path, orient='records', force_ascii=False)
    print(f"JSON 파일 저장 완료: {json_path}")
else:
    print("데이터를 가져오지 못했습니다.")