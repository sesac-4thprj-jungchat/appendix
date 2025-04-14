# %% ✅ 📌 최적화된 카테고리 분류 및 데이터 정리
import pandas as pd
import re
import datetime
import os
import json
from tqdm import tqdm

# ✅ 최신 파일 자동 로드
data_dir = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file"
file_list = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
if not file_list:
    raise FileNotFoundError("📢 CSV 파일이 해당 폴더에 존재하지 않습니다!")

latest_file = max(file_list, key=lambda f: os.path.getmtime(os.path.join(data_dir, f)))
file_path = os.path.join(data_dir, latest_file)
print(f"📂 가장 최신 CSV 파일: {file_path}")

# ✅ 데이터 로드
df = pd.read_csv(file_path)
tqdm.pandas()

# ✅ 기존 불필요한 컬럼 삭제
df = df.drop(columns=["benefit_category_refined", "benefit_category_multi"], errors="ignore")

# ✅ 다중 카테고리 매핑 패턴
category_patterns = {
    "문화-환경": ["이용요금 감면", "무료 제공", "할인 혜택", "공원 이용"],
    "생활안정": ["재난", "사고 피해", "공공요금 지원", "경영안정자금", "생계 지원"],
    "보육-교육": ["기저귀 바우처", "학원비 지원", "교육비 감면", "장학금"],
    "보건-의료": ["태아 검진", "초음파 검진", "보건 서비스", "의료비", "유가족 장례비"],
    "임신-출산": ["임산부", "산후 조리", "출산 지원", "태아 초음파"],
    "행정-안전": ["국가유공자", "보훈", "장례비", "유가족 지원", "재난 지원"]
}

# ✅ 다중 카테고리 변환 함수 (최적화)
def refine_category(text):
    if pd.isnull(text):
        return ["기타"]

    text = str(text)
    new_categories = set()

    for category, patterns in category_patterns.items():
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            new_categories.add(category)

    return list(new_categories) if new_categories else ["기타"]

# ✅ 다중 카테고리 적용
df["benefit_category"] = df["서비스목적요약"].progress_apply(refine_category)

# ✅ "기타" 개수 변화 확인
initial_etc_count = df["benefit_category"].apply(lambda x: "기타" in x if isinstance(x, list) else x == "기타").sum()
final_etc_count = df["benefit_category"].apply(lambda x: "기타" in x if isinstance(x, list) else x == "기타").sum()
reduction = initial_etc_count - final_etc_count
reduction_percentage = (reduction / initial_etc_count) * 100 if initial_etc_count > 0 else 0

print(f"✅ '기타' 개수 감소 (최적화 후): {initial_etc_count} → {final_etc_count} ({reduction_percentage:.2f}%)")

# ✅ EDA 최종 결측치 확인
initial_nulls = df.isnull().sum()
final_nulls = df.isnull().sum()
null_reduction = initial_nulls - final_nulls

# ✅ 저장 경로 및 파일명 설정
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ✅ CSV 저장
csv_filename = f"{timestamp}_norm_benefit_category_final.csv"
csv_path = os.path.join(data_dir, csv_filename)
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# ✅ JSON 저장 (리스트 형태 유지)
json_filename = f"{timestamp}_norm_benefit_category_final.json"
json_path = os.path.join(data_dir, json_filename)
df.to_json(json_path, orient="records", force_ascii=False, indent=4)

# ✅ EDA 결과 저장
eda_results = {
    "initial_null_counts": {k: int(v) for k, v in initial_nulls.to_dict().items()},
    "final_null_counts": {k: int(v) for k, v in final_nulls.to_dict().items()},
    "null_reduction": {k: int(v) for k, v in null_reduction.to_dict().items()},
    "initial_etc_count": int(initial_etc_count),
    "final_etc_count": int(final_etc_count),
    "etc_reduction_percentage": float(reduction_percentage)
}

eda_filename = f"{timestamp}_eda_benefit_category_final.json"
eda_path = os.path.join(data_dir, eda_filename)
with open(eda_path, "w", encoding="utf-8") as f:
    json.dump(eda_results, f, ensure_ascii=False, indent=4)

# ✅ 결과 출력
print(f"✅ CSV 저장 완료: {csv_path}")
print(f"✅ JSON 저장 완료: {json_path}")
print(f"✅ EDA 결과 저장 완료: {eda_path}")

# %%잘못된 benefit_category 샘플링 분석
import pandas as pd

# ✅ 데이터 로드
file_path = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file\20250316_153706_norm_benefit_category_final.csv"
df = pd.read_csv(file_path)

# ✅ "생활안정"과 "보육-교육"이 겹치는 샘플 확인
df_conflict = df[df["benefit_category"].apply(lambda x: isinstance(x, list) and "생활안정" in x and "보육-교육" in x)]
if not df_conflict.empty:
    print("🔍 생활안정 & 보육-교육 중복 샘플")
    print(df_conflict[["서비스목적요약", "benefit_category"]].sample(min(10, len(df_conflict)), random_state=42))
else:
    print("❌ 생활안정 & 보육-교육 중복 데이터 없음")

# ✅ "보건-의료" vs "임신-출산" 분류가 애매한 데이터 확인
df_health_conflict = df[df["benefit_category"].apply(lambda x: isinstance(x, list) and "보건-의료" in x and "임신-출산" in x)]
if not df_health_conflict.empty:
    print("\n🔍 보건-의료 & 임신-출산 중복 샘플")
    print(df_health_conflict[["서비스목적요약", "benefit_category"]].sample(min(10, len(df_health_conflict)), random_state=42))
else:
    print("\n❌ 보건-의료 & 임신-출산 중복 데이터 없음")

# ✅ "기타"에 분류된 데이터 샘플링 (최종적으로 줄여야 할 데이터)
df_etc_sample = df[df["benefit_category"].apply(lambda x: isinstance(x, list) and "기타" in x)]
if not df_etc_sample.empty:
    print("\n🔍 기타로 분류된 데이터 샘플")
    print(df_etc_sample[["서비스목적요약", "benefit_category"]].sample(min(20, len(df_etc_sample)), random_state=42))
else:
    print("\n❌ '기타'로 분류된 데이터 없음")

# ✅ 불필요한 다중 카테고리 확인 (예: 생활안정, 보육-교육, 기타가 같이 있는 경우)
df_too_many_labels = df[df["benefit_category"].apply(lambda x: isinstance(x, list) and len(x) > 2)]
if not df_too_many_labels.empty:
    print("\n🔍 다중 카테고리 개수 3개 이상인 샘플")
    print(df_too_many_labels[["서비스목적요약", "benefit_category"]].sample(min(10, len(df_too_many_labels)), random_state=42))
else:
    print("\n❌ 다중 카테고리 개수 3개 이상 데이터 없음")


# %%
import pandas as pd

# ✅ 데이터 로드
file_path = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file\20250316_153706_norm_benefit_category_final.csv"
df = pd.read_csv(file_path)

# ✅ 데이터 타입 확인
print(df["benefit_category"].dtype)  # object라면 리스트가 아니라 문자열로 저장되었을 가능성 있음

# ✅ 다중 카테고리 샘플 10개 확인
print("\n🔍 다중 카테고리 적용된 데이터 샘플")
print(df[df["benefit_category"].apply(lambda x: "," in str(x))].sample(10))

# ✅ "기타"로 분류된 데이터 확인 (여전히 남아있는지)
df_etc = df[df["benefit_category"].apply(lambda x: "기타" in str(x))]
print("\n🔍 '기타' 포함된 데이터 개수:", len(df_etc))

# ✅ "생활안정" & "보육-교육" 같이 포함된 샘플 확인
df_conflict = df[df["benefit_category"].apply(lambda x: "생활안정" in str(x) and "보육-교육" in str(x))]
print("\n🔍 '생활안정' & '보육-교육' 중복 샘플 개수:", len(df_conflict))

# %%
import pandas as pd
import ast  # 문자열을 리스트로 변환

# ✅ 데이터 로드
file_path = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file\20250316_153706_norm_benefit_category_final.csv"
df = pd.read_csv(file_path)

# ✅ "benefit_category" 컬럼이 리스트처럼 보이지만 문자열인 경우 변환
def convert_to_list(value):
    try:
        return ast.literal_eval(value) if isinstance(value, str) else value
    except:
        return ["기타"]  # 변환 오류 발생 시 "기타" 처리

df["benefit_category"] = df["benefit_category"].apply(convert_to_list)

# ✅ 다중 카테고리 샘플 10개 확인 (정상적으로 리스트로 변환되었는지 확인)
print("\n🔍 다중 카테고리 적용된 데이터 샘플")
print(df[df["benefit_category"].apply(lambda x: isinstance(x, list) and len(x) > 1)].sample(10))

# ✅ "기타" 포함된 데이터 개수 확인
df_etc = df[df["benefit_category"].apply(lambda x: "기타" in x)]
print("\n🔍 '기타' 포함된 데이터 개수 (변환 후):", len(df_etc))

# ✅ CSV 저장 (변환된 데이터)
timestamp = "20250316_153706"  # 기존 파일과 동일한 타임스탬프 유지
csv_path = file_path.replace(".csv", "_fixed.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"✅ 변환 완료! 새로운 파일 저장: {csv_path}")

# %%
import pandas as pd
import ast  # 문자열을 리스트로 변환

# ✅ 데이터 로드
file_path = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file\20250316_153706_norm_benefit_category_final_fixed.csv"
df = pd.read_csv(file_path)

# ✅ "benefit_category" 컬럼이 리스트처럼 보이지만 문자열인 경우 변환
def convert_to_list(value):
    try:
        return ast.literal_eval(value) if isinstance(value, str) else value
    except:
        return ["기타"]  # 변환 오류 발생 시 "기타" 처리

df["benefit_category"] = df["benefit_category"].apply(convert_to_list)

# ✅ "기타"만 포함된 경우 제거
df["benefit_category"] = df["benefit_category"].apply(lambda x: ["기타"] if len(x) == 1 and "기타" in x else [cat for cat in x if cat != "기타"])

# ✅ 변환 후 "기타" 개수 확인
df_etc = df[df["benefit_category"].apply(lambda x: "기타" in x)]
print("\n🔍 '기타' 포함된 데이터 개수 (최종 정리 후):", len(df_etc))

# ✅ CSV 저장 (최적화된 데이터)
timestamp = "20250316_153706"  # 기존 파일과 동일한 타임스탬프 유지
csv_path = file_path.replace(".csv", "_cleaned.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"✅ '기타' 정리 완료! 새로운 파일 저장: {csv_path}")

# %%"기타" 데이터만 따로 추출 & 샘플링 (추가할 카테고리 패턴을 찾는 작업)
import pandas as pd
from collections import Counter
import re

# ✅ 데이터 로드
file_path = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file\20250316_153706_norm_benefit_category_final_fixed_cleaned.csv"
df = pd.read_csv(file_path)

# ✅ "기타"로 분류된 데이터만 추출
df_etc = df[df["benefit_category"].apply(lambda x: "기타" in x if isinstance(x, list) else x == "기타")]

# ✅ "기타" 데이터 개수 확인
etc_count = len(df_etc)
print(f"\n🔍 '기타'로 분류된 데이터 개수: {etc_count}")

# ✅ "기타" 데이터 샘플링 (데이터가 있을 경우에만)
if etc_count > 0:
    print("\n🔍 '기타'로 분류된 데이터 샘플 (최대 20개):")
    print(df_etc[["서비스목적요약", "benefit_category"]].sample(min(20, etc_count), random_state=42))

    # ✅ "기타" 데이터에서 키워드 추출
    keywords = []
    for text in df_etc["서비스목적요약"].dropna():
        words = re.findall(r"\b\w+\b", text.lower())  # 단어 추출
        keywords.extend(words)

    # ✅ 가장 많이 등장한 키워드 TOP 20 확인
    counter = Counter(keywords)
    print("\n🔍 '기타' 데이터에서 가장 많이 등장한 키워드 TOP 20:")
    print(counter.most_common(20))

else:
    print("\n✅ '기타'로 분류된 데이터가 없습니다! 🎉")


# %%
# ✅ 각 카테고리별 데이터 개수 확인
category_counts = df["benefit_category"].explode().value_counts()
print("\n🔍 최종 카테고리별 데이터 개수:")
print(category_counts)

# ✅ '기타'로 분류된 데이터 샘플 확인
df_etc = df[df["benefit_category"].apply(lambda x: "기타" in x)]
print("\n🔍 '기타'로 남아 있는 데이터 샘플:")
print(df_etc[["서비스목적요약", "benefit_category"]].sample(20, random_state=42))


# %% ✅ 📌 "기타" 최적화 및 다중 카테고리 강화
import pandas as pd
import re
import datetime
import os
import json
from tqdm import tqdm

# ✅ 데이터 로드
file_path = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file\20250316_153706_norm_benefit_category_final_fixed_cleaned.csv"
df = pd.read_csv(file_path)

# ✅ tqdm 설정
tqdm.pandas()

# ✅ 기존 "기타" 데이터 개수 확인
initial_etc_count = df["benefit_category"].apply(lambda x: "기타" in x).sum()
print(f"\n🔍 기존 '기타' 개수: {initial_etc_count}")

# ✅ 기존 카테고리에 새로운 키워드 추가 (확장)
updated_category_patterns = {
    "생활안정": ["생계비", "장제비", "해산비", "경영지원", "빈곤", "수급", "생계", "긴급 지원", "소득"],
    "보육-교육": ["청소년", "학원비", "교통비 지원", "장학금", "보육", "유아,""학비", "학교지원"],
    "보건-의료": ["예방접종", "의료비", "건강보험료", "입원", "간호", "치료","예방", "재활", "병원비"],
    "고용-창업": ["자영업", "스타트업", "소상공인"],
    "임신-출산": ["출산", "임부", "산후", "태아", "해산비"],
    "행정-안전": ["범죄피해자", "법률", "보호", "지원금"],
    "문화-환경": ["예술인", "체육", "관광", "문화", "공연"],
    "농림축산어업": ["농어업", "어민"]
}

# ✅ 다중 카테고리 매칭 함수 (기존 '기타'에만 적용)
def refine_category_with_more_patterns(text):
    if pd.isnull(text):
        return ["기타"]

    text = str(text)
    new_categories = set()

    for category, patterns in updated_category_patterns.items():
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            new_categories.add(category)

    return list(new_categories) if new_categories else ["기타"]

# ✅ 기존 '기타' 데이터만 새롭게 카테고리 매칭
df.loc[df["benefit_category"].apply(lambda x: "기타" in x), "benefit_category"] = df["서비스목적요약"].apply(refine_category_with_more_patterns)

# ✅ "기타" 개수 변화 확인
final_etc_count = df["benefit_category"].apply(lambda x: "기타" in x).sum()
reduction = initial_etc_count - final_etc_count
reduction_percentage = (reduction / initial_etc_count) * 100 if initial_etc_count > 0 else 0

print(f"\n✅ '기타' 개수 감소 (새로운 패턴 적용 후): {initial_etc_count} → {final_etc_count} ({reduction_percentage:.2f}%)")

# ✅ EDA 최종 결측치 확인
initial_nulls = df.isnull().sum()
final_nulls = df.isnull().sum()
null_reduction = initial_nulls - final_nulls

# ✅ 저장 경로 및 파일명 설정
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file"
os.makedirs(output_dir, exist_ok=True)

# ✅ CSV 저장
csv_filename = f"{timestamp}_norm_benefit_category_final_optimized.csv"
csv_path = os.path.join(output_dir, csv_filename)
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# ✅ JSON 저장 (리스트 형태 유지)
json_filename = f"{timestamp}_norm_benefit_category_final_optimized.json"
json_path = os.path.join(output_dir, json_filename)
df.to_json(json_path, orient="records", force_ascii=False, indent=4)

# ✅ EDA 결과 저장
eda_results = {
    "initial_null_counts": {k: int(v) for k, v in initial_nulls.to_dict().items()},
    "final_null_counts": {k: int(v) for k, v in final_nulls.to_dict().items()},
    "null_reduction": {k: int(v) for k, v in null_reduction.to_dict().items()},
    "initial_etc_count": int(initial_etc_count),
    "final_etc_count": int(final_etc_count),
    "etc_reduction_percentage": float(reduction_percentage)
}

eda_filename = f"{timestamp}_eda_benefit_category_final_optimized.json"
eda_path = os.path.join(output_dir, eda_filename)
with open(eda_path, "w", encoding="utf-8") as f:
    json.dump(eda_results, f, ensure_ascii=False, indent=4)

# ✅ 결과 출력
print(f"\n✅ CSV 저장 완료: {csv_path}")
print(f"✅ JSON 저장 완료: {json_path}")
print(f"✅ EDA 결과 저장 완료: {eda_path}")

# %%
# %% ✅ "기타" 데이터 샘플링 및 분석
import pandas as pd

# ✅ 데이터 로드
file_path = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file\20250316_160919_norm_benefit_category_final_optimized.csv"
df = pd.read_csv(file_path)

# ✅ "기타" 데이터 샘플링 (랜덤 100개)
df_etc = df[df["benefit_category"].apply(lambda x: "기타" in x)]
df_etc_sample = df_etc.sample(100, random_state=42)

# ✅ "기타" 데이터 출력
print("\n🔍 '기타'로 남아 있는 데이터 샘플:")
print(df_etc_sample[["서비스목적요약", "benefit_category"]])

# ✅ "기타" 데이터를 CSV로 저장해서 분석
output_path = r"C:\Users\hhhey\Desktop\SeSacAI\4.Proj\5. gov_money\1.data\0.data_fianl\250304_data_file\etc_samples.csv"
df_etc_sample.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✅ '기타' 데이터 샘플 저장 완료: {output_path}")
# %%
from kiwi import Kiwi

# Kiwi 초기화
kiwi = Kiwi()

region_map = {
    "서울": "서울특별시", "서울시": "서울특별시",
    "부산": "부산광역시", "부산시": "부산광역시",
    "대구": "대구광역시", "대구시": "대구광역시",
    "인천": "인천광역시", "인천시": "인천광역시",
    "광주": "광주광역시", "광주시": "광주광역시",
    "대전": "대전광역시", "대전시": "대전광역시",
    "울산": "울산광역시", "울산시": "울산광역시",
    "세종": "세종특별자치시", "세종시": "세종특별자치시",
    "경기": "경기도", "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도",
    "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도", "경남": "경상남도",
    "제주": "제주특별자치도"
}
# 서울 구별 리스트
seoul_districts = {
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구",
    "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구",
    "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"
}
