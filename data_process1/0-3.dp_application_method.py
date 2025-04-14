#신청 방법 (application_method) 세분화 작업

import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import os
import json

# ✅ 데이터 로드
file_path = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file\20250315_234747_norm_gender_household_personal.csv"
df = pd.read_csv(file_path)

# ✅ 신청 방법(application_method) 정규화 함수 (리스트 형태)
def extract_application_methods(text):
    if pd.isnull(text):
        return ["기타"]  # ✅ fillna() 대신 직접 리스트 반환
    
    text = str(text)
    methods = []

    if any(keyword in text for keyword in ["온라인", "홈페이지", "인터넷", "웹사이트"]):
        methods.append("온라인 신청")
    if any(keyword in text for keyword in ["방문", "센터", "기관 방문", "직접 접수"]):
        methods.append("방문 신청")
    if any(keyword in text for keyword in ["우편", "서류 제출", "우편 접수"]):
        methods.append("우편 신청")
    if any(keyword in text for keyword in ["전화", "콜센터", "유선 접수"]):
        methods.append("전화 신청")
    
    return methods if methods else ["기타"]

# ✅ `application_method` 컬럼 적용
df['application_method'] = df['신청방법'].apply(extract_application_methods)

# ✅ 원본 데이터 복사 (비교용)
df_original = df.copy()

# ✅ 변경된 행 개수 계산 (정제 전후 비교)
method_changed = (df['application_method'].astype(str) != df_original.get('application_method', ["기타"]).astype(str)).sum()

# ✅ 결측치 처리 방식 개선 (fillna 대신 apply 활용)
df['application_method'] = df['application_method'].apply(lambda x: x if isinstance(x, list) else ["기타"])

# ✅ AI 적용 후 결측치 개수 기록
missing_before = df_original.isnull().sum()
missing_after = df.isnull().sum()

# ✅ 정제율(%) 계산
fill_rate = ((missing_before - missing_after) / missing_before * 100).fillna(0)

# ✅ 변경된 행 개수 추가
eda_results = fill_rate.to_dict()
eda_results['application_method_changed_count'] = int(method_changed)

# ✅ 저장 경로 및 파일명 설정
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file"
os.makedirs(output_dir, exist_ok=True)

# ✅ CSV 저장
csv_filename = f"{timestamp}_norm_application_method.csv"
csv_path = os.path.join(output_dir, csv_filename)
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# ✅ JSON 저장
json_filename = f"{timestamp}_norm_application_method.json"
json_path = os.path.join(output_dir, json_filename)
df.to_json(json_path, orient="records", force_ascii=False, indent=4)

# ✅ 정제율 JSON 저장
eda_json_filename = f"{timestamp}_eda_results.json"
eda_json_path = os.path.join(output_dir, eda_json_filename)

with open(eda_json_path, "w", encoding="utf-8") as f:
    json.dump(eda_results, f, ensure_ascii=False, indent=4)

# ✅ 결과 출력
print(f"✅ CSV 저장 완료: {csv_path}")
print(f"✅ JSON 저장 완료: {json_path}")
print(f"✅ EDA 결과 저장 완료: {eda_json_path}")
print("✅ 정제 후 결측치 채운 비율 (컬럼별 %):")
print(fill_rate)
print(f"✅ application_method 변경된 개수: {method_changed}")
