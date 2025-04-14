# 정제 로직_정확한 지역(area, district) 추출을 목표
import pandas as pd
import re
from kiwipiepy import Kiwi
from tqdm import tqdm
import datetime
import os
import json

# ✅ 데이터 로드
file_path = "C:\\Users\\hhhey\\Desktop\\SeSacAI\\4.Proj\\5. gov_money\\1.data\\0.data_fianl\\250304_data_file\\20250304.csv"
df = pd.read_csv(file_path)

# ✅ 날짜 형식 변환
df['등록일시'] = pd.to_datetime(df['등록일시'].astype(str), format='%Y%m%d%H%M%S', errors='coerce').dt.strftime('%Y-%m-%d')
df['수정일시'] = pd.to_datetime(df['수정일시'].astype(str), format='%Y%m%d%H%M%S', errors='coerce').dt.strftime('%Y-%m-%d')

# ✅ Kiwi 형태소 분석기 초기화
kiwi = Kiwi()

# ✅ 지역 매핑 (광역시도)
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

# ✅ 서울 구별 리스트
seoul_districts = {
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구",
    "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구",
    "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"
}

# ✅ 경기 및 기타 지역 리스트
gy_candidate_districts = {
    "수원시", "고양시", "성남시", "부천시", "용인시", "안양시", "평택시", "의정부시", "시흥시",
    "파주시", "김포시", "광명시", "군포시", "안산시", "남양주시", "이천시", "안성시", "포천시",
    "여주시", "양평군", "가평군"
}
other_candidate_districts = {"거제시", "임실군", "함평군", "천안시", "순천시", "포항시", "창원시", 
                             "달서구", "익산시", "충주시", "홍성군", "전주시", "원주시",
                             "아산시", "영남", "송광", "마산"}

# ✅ 모든 구/시/군 통합
all_districts = seoul_districts.union(gy_candidate_districts).union(other_candidate_districts)


# ✅ **지역 추출 함수 정의**
def extract_area_multi(row, priority_cols):
    extracted_areas = set()
    for col in priority_cols:
        val = row[col]
        if pd.notnull(val):
            extracted_areas.update([region_map[tok] for tok in region_map if tok in val])
    return ", ".join(sorted(extracted_areas)) if extracted_areas else ""

# ✅ district 추출 함수
def extract_district(text, area):
    if pd.isnull(text) or not area:
        return ""

    text = str(text).replace("\r", " ").replace("\n", " ").strip()
    tokens = [token.form for token in kiwi.tokenize(text)]
    
    refined_tokens = set()
    for token in tokens:
        if area == "서울특별시" and token in seoul_districts:
            refined_tokens.add(token)
        elif area == "경기도" and token in gy_candidate_districts:
            refined_tokens.add(token)
        elif token in other_candidate_districts:
            refined_tokens.add(token)

    return ", ".join(sorted(refined_tokens)) if refined_tokens else ""

def extract_area_multi(row, priority_cols):
    extracted_areas = set()
    for col in priority_cols:
        val = row[col]
        if pd.notnull(val):
            tokens = [tok for tok in region_map.keys() if tok in val]  # ✅ 이 부분을 수정
            extracted_areas.update(region_map[tok] for tok in tokens)
    return ", ".join(sorted(extracted_areas)) if extracted_areas else ""

# ✅ `district` 추출 함수 (정규표현식 추가)
def extract_district_multi(row, priority_cols):
    extracted_districts = set()
    for col in priority_cols:
        val = row[col]
        if pd.notnull(val):
            matches = re.findall(r"([가-힣]+(?:특별자치시|광역시|도|시|군|구))", val)  # ✅ 확장된 패턴
            for match in matches:
                if match in all_districts:  # ✅ 리스트에 있는 경우만 추가
                    extracted_districts.add(match)
    return ", ".join(sorted(extracted_districts)) if extracted_districts else ""



# ✅ **데이터프레임에 적용 (🔥 tqdm 추가)**
tqdm.pandas()
priority_cols = ['지원대상', '지원내용', '서비스명', '서비스목적요약']

df['area'] = df.progress_apply(lambda row: extract_area_multi(row, priority_cols), axis=1)
df['district'] = df.progress_apply(lambda row: extract_district_multi(row, priority_cols), axis=1)

print(df[['area', 'district']].sample(10))  # ✅ 지역 추출 확인


# 
df["area"].value_counts()
#
df["district"].value_counts()


# ✅ **데이터 정리**
df.loc[(df['district'].isna()) & (df['area'].isin(region_map.values())), 'district'] = "전체"
df = df[~df['area'].str.contains("전국", na=False)]
df.dropna(subset=['district', 'area'], how='all', inplace=True)

# ✅ JSON 저장을 위해 리스트 변환
df['area'] = df['area'].apply(lambda x: x.split(", ") if isinstance(x, str) and x != "" else [])
df['district'] = df['district'].apply(lambda x: x.split(", ") if isinstance(x, str) and x != "" else [])

df['area_json'] = df['area'].apply(lambda x: json.dumps(x, ensure_ascii=False))
df['district_json'] = df['district'].apply(lambda x: json.dumps(x, ensure_ascii=False))

print(df[['area_json', 'district_json']].sample(10))  # ✅ JSON 변환 확인




#
# ✅ CSV 및 JSON 저장 (4️⃣ 저장)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "C:\\Users\\hhhey\\Desktop\\SeSacAI\\4.Proj\\5. gov_money\\1.data\\0.data_fianl\\250304_data_file"
os.makedirs(output_dir, exist_ok=True)

# CSV 저장
csv_path = os.path.join(output_dir, f"{timestamp}_processed_data_region_district.csv")
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"CSV 파일 저장 완료: {csv_path}")

# JSON 저장
json_path = os.path.join(output_dir, f"{timestamp}_processed_data_region_district.json")
df.to_json(json_path, orient='records', force_ascii=False, indent=4)
print(f"JSON 파일 저장 완료: {json_path}")



# %% ❌ 검증 실패한 행 개수: 87
def validate_area_district(row):
    # ✅ area와 district가 리스트인지 확인하고, 아니라면 리스트로 변환
    area_list = row["area"] if isinstance(row["area"], list) else []
    district_list = row["district"] if isinstance(row["district"], list) else []
    
    if len(area_list) >= 2:  # ✅ area가 2개 이상이면
        return len(district_list) >= 2  # ✅ district도 최소 2개 이상인지 체크
    return True  # area가 1개 이하라면 검증 필요 없음

# ✅ 데이터프레임에 적용
df["valid_area_district"] = df.apply(validate_area_district, axis=1)

# ✅ 결과 확인 (False가 있는지 체크)
invalid_rows = df[df["valid_area_district"] == False]
print(f"❌ 검증 실패한 행 개수: {len(invalid_rows)}")

# ✅ 데이터 직접 확인 (에러 없이 출력됨)
import pandas as pd
from IPython.display import display

display(invalid_rows)  # 🚀 제대로 나오는지 확인

import os
import datetime

# ✅ 저장 경로 설정
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "C:\\Users\\hhhey\\Desktop\\SeSacAI\\4.Proj\\5. gov_money\\1.data\\0.data_fianl\\250304_data_file"
os.makedirs(output_dir, exist_ok=True)  # 폴더 없으면 생성

# ✅ CSV 저장
csv_path = os.path.join(output_dir, f"{timestamp}_processed_valid_area_district.csv")
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"📂 CSV 저장 완료: {csv_path}")




