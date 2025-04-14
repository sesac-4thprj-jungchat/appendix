#%%
# import datetime
# import os
# import pandas as pd
# import json

# # ✅ CSV 파일 로드
# file_path = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file\20250314_142300_0-1.dp_support.csv"
# df = pd.read_csv(file_path)

# # ✅ 지원 유형 정규화 함수 (리스트 형 유지)
# def normalize_support_type(support_str):
#     if not isinstance(support_str, str):
#         return support_str  # 문자열이 아니면 그대로 반환

#     support_list = support_str.split("||")  # 기존 데이터에서 리스트로 변환
#     normalized_list = []

#     for support in support_list:
#         if support.startswith("기타(") and support.endswith(")"):
#             normalized_list.append(support[3:-1])  # ✅ "기타(XX)" → "XX"
#         elif "현금" in support:
#             normalized_list.append("현금")  # ✅ "현금(장학금)" → "현금"
#         else:
#             normalized_list.append(support)  # 그 외 항목은 그대로 유지

#     return normalized_list

# # ✅ 지원유형 컬럼 정규화 적용
# df["지원유형"] = df["지원유형"].apply(normalize_support_type)

# # ✅ 현재 시간 (datetime 기반 파일명)
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# # ✅ 저장할 파일명 설정
# output_dir = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file"
# os.makedirs(output_dir, exist_ok=True)

# # ✅ CSV 저장
# csv_filename = f"{timestamp}_0-1.dp_support_type_normalized.csv"
# csv_path = os.path.join(output_dir, csv_filename)
# df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# # ✅ JSON 저장 (지원유형 JSON 변환)
# json_filename = f"{timestamp}_0-1.support_type_normalized.json"
# json_path = os.path.join(output_dir, json_filename)
# df.to_json(json_path, orient="records", force_ascii=False, indent=4)

# # ✅ 결과 출력
# print(f"✅ CSV 저장 완료: {csv_path}")
# print(f"✅ JSON 저장 완료: {json_path}")


#%%
import pandas as pd
import re

# ✅ 데이터 로드
file_path = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file\20250314_195020_0-1.dp_support_type_normalized.csv"  # 실제 파일 경로로 변경

df = pd.read_csv(file_path)

# ✅ 지원유형 통합 매핑 테이블
support_type_map = {
    "의료지원": "의료", "서비스(의료지원)": "의료", "서비스(의료)": "의료",
    "서비스(일자리)": "일자리", "일자리 지원": "일자리",
    "이용권(교통)": "이용권", "이용권(문화)": "이용권", "이용권(교육)": "이용권",
    "현금(장학금)": "현금", "현금(감면)": "현금",
    "상담/법률지원": "법률", "법률상담": "법률", "법률지원": "법률",
    "서비스(돌봄)": "돌봄", "돌봄서비스": "돌봄"
}

def normalize_support_type(types):
    if pd.isnull(types):
        return []
    
    type_list = types.split("||")  # 기존 구분자 기준 분리
    normalized_types = [support_type_map.get(t.strip(), t.strip()) for t in type_list]
    
    return list(set(normalized_types))  # 중복 제거 후 반환

df["지원유형"] = df["지원유형"].apply(normalize_support_type)

# ✅ 지원대상 정규화 함수
def normalize_support_target(text):
    if pd.isnull(text):
        return ""
    
    text = re.sub(r"^○", "", text).strip()  # 특수문자 제거
    
    # 연령대 범주화
    if re.search(r"\d{1,2}세.*\d{1,2}세", text):
        age_match = re.findall(r"(\d{1,2})세", text)
        if age_match:
            ages = [int(a) for a in age_match]
            min_age = min(ages)

            if 9 <= min_age <= 24:
                return "청소년"
            elif 19 <= min_age <= 34:
                return "청년"
            elif 40 <= min_age <= 64:
                return "장년"
            elif min_age >= 65:
                return "노인"

    
    # 기타(XX) 정리
    text = re.sub(r"기타\((.*?)\)", r"\1", text)
    
    # 특정 키워드 정리
    target_map = {"취약계층": "저소득층", "저소득층": "저소득층", "차상위계층": "저소득층"}
    for k, v in target_map.items():
        if k in text:
            return v
    
    return text

df["지원대상"] = df["지원대상"].apply(normalize_support_target)

# ✅ 결과 저장
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
df.to_csv(f"{timestamp}_norm_spt_target.csv", index=False, encoding="utf-8-sig")
print("✅ 데이터 정규화 완료!")

# ✅ 저장할 파일명 설정
output_dir = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file"
os.makedirs(output_dir, exist_ok=True)

# ✅ CSV 저장
csv_filename = f"{timestamp}_norm_spt_target.csv"
csv_path = os.path.join(output_dir, csv_filename)
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# ✅ JSON 저장 (지원유형 JSON 변환)
json_filename = f"{timestamp}_norm_spt_target.json"
json_path = os.path.join(output_dir, json_filename)
df.to_json(json_path, orient="records", force_ascii=False, indent=4)

# ✅ 결과 출력
print(f"✅ CSV 저장 완료: {csv_path}")
print(f"✅ JSON 저장 완료: {json_path}")

# %%
