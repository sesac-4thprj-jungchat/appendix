import openai
import os
from typing import Dict, Optional, List, Any, Union
import json
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY가 .env 파일에서 로드되지 않았습니다.")
else:
    print("API Key loaded successfully:", api_key[:5] + "...")

# OpenAI 클라이언트 초기화
client = openai.OpenAI(api_key=api_key)

# DB 데이터
AREA_DISTRICT_MAPPING = {
    "서울특별시": [
        "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", 
        "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", 
        "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", 
        "영등포구", "용산구", "은평구", "종로구", "중구"
    ],
    "부산광역시": [
        "강서구", "금정구", "기장군", "남구", "동구", "동래구", 
        "부산진구", "북구", "사상구", "사하구", "서구", 
        "수영구", "연제구", "영도구", "중구", "해운대구"
    ],
    "대구광역시": [
        "군위군", "남구", "달서구", "달성군", "동구", "북구", 
        "서구", "수성구", "중구"
    ],
    "인천광역시": [
        "강화군", "계양구", "남동구", "동구", "미추홀구", 
        "부평구", "서구", "연수구", "옹진군", "중구"
    ],
    "광주광역시": [
        "광산구", "남구", "동구", "북구", "서구"
    ],
    "대전광역시": [
        "대덕구", "동구", "서구", "유성구", "중구"
    ],
    "울산광역시": [
        "남구", "동구", "북구", "울주군", "중구"
    ],
    "세종특별자치시": ["세종특별자치시"],
    "경기도": [
        "가평군", "고양시", "과천시", "광명시", "광주시", "구리시", 
        "군포시", "김포시", "남양주시", "동두천시", "부천시", 
        "성남시", "수원시", "시흥시", "안산시", "안성시", 
        "안양시", "양주시", "양평군", "여주시", "연천군", 
        "오산시", "용인시", "의왕시", "의정부시", "이천시", 
        "파주시", "평택시", "포천시", "하남시", "화성시"
    ],
    "충청북도": [
        "괴산군", "단양군", "보은군", "영동군", "옥천군", 
        "음성군", "제천시", "증평군", "진천군", "청주시", "충주시"
    ],
    "충청남도": [
        "계룡시", "공주시", "금산군", "논산시", "당진시", 
        "보령시", "부여군", "서산시", "서천군", "아산시", 
        "예산군", "천안시", "청양군", "태안군", "홍성군"
    ],
    "전라남도": [
        "강진군", "고흥군", "곡성군", "광양시", "구례군", 
        "나주시", "담양군", "목포시", "무안군", "보성군", 
        "순천시", "신안군", "여수시", "영광군", "영암군", 
        "완도군", "장성군", "장흥군", "진도군", "함평군", 
        "해남군", "화순군"
    ],
    "경상북도": [
        "경산시", "경주시", "고령군", "구미시", "김천시", 
        "문경시", "봉화군", "상주시", "성주군", "안동시", 
        "영덕군", "영양군", "영주시", "영천시", "예천군", 
        "울릉군", "울진군", "의성군", "청도군", "청송군", 
        "칠곡군", "포항시"
    ],
    "경상남도": [
        "거제시", "거창군", "고성군", "김해시", "남해군", "밀양시", 
        "사천시", "산청군", "양산시", "의령군", "진주시", 
        "창녕군", "창원시", "통영시", "하동군", "함안군", 
        "함양군", "합천군"
    ],
    "제주특별자치도": [
        "서귀포시", "제주시"
    ],
    "강원특별자치도": [
        "강릉시", "고성군", "동해시", "삼척시", "속초시", 
        "양구군", "양양군", "영월군", "원주시", "인제군", 
        "정선군", "철원군", "춘천시", "태백시", "평창군", 
        "홍천군", "화천군", "횡성군"
    ],
    "전북특별자치도": [
        "고창군", "군산시", "김제시", "남원시", "무주군", 
        "부안군", "순창군", "완주군", "익산시", "임실군", 
        "장수군", "전주시", "덕진구", "완산구", 
        "정읍시", "진안군"
    ],
}
# ALLOWED_VALUES: DB에 저장된 형식에 맞는 허용된 값 목록
ALLOWED_VALUES = {
    "area": ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시", "세종특별자치시", "경기도", "충청북도", "충청남도", "전라남도", "경상북도", "경상남도", "제주특별자치도", "강원특별자치도", "전북특별자치도"],
    "district": ["강동구", "강북구", "강서구", "관악구", "광진구", "구로구", 
        "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", 
        "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", 
        "영등포구", "용산구", "은평구", "종로구", "중구",
        "강서구", "금정구", "기장군", "남구", "동구", "동래구", 
        "부산진구", "북구", "사상구", "사하구", "서구", 
        "수영구", "연제구", "영도구", "중구", "해운대구",
        "군위군", "남구", "달서구", "달성군", "동구", "북구", 
        "서구", "수성구", "중구",
        "강화군", "계양구", "남동구", "동구", "미추홀구", 
        "부평구", "서구", "연수구", "옹진군", "중구",
        "광산구", "남구", "동구", "북구", "서구",
        "대덕구", "동구", "서구", "유성구", "중구",
        "남구", "동구", "북구", "울주군", "중구",
        "가평군", "고양시", "과천시", "광명시", "광주시", "구리시", 
        "군포시", "김포시", "남양주시", "동두천시", "부천시", 
        "성남시", "수원시", "시흥시", "안산시", "안성시", 
        "안양시", "양주시", "양평군", "여주시", "연천군", 
        "오산시", "용인시", "의왕시", "의정부시", "이천시", 
        "파주시", "평택시", "포천시", "하남시", "화성시",
        "괴산군", "단양군", "보은군", "영동군", "옥천군", 
        "음성군", "제천시", "증평군", "진천군", "청주시", "충주시",
        "계룡시", "공주시", "금산군", "논산시", "당진시", 
        "보령시", "부여군", "서산시", "서천군", "아산시", 
        "예산군", "천안시", "청양군", "태안군", "홍성군",
        "강진군", "고흥군", "곡성군", "광양시", "구례군", 
        "나주시", "담양군", "목포시", "무안군", "보성군", 
        "순천시", "신안군", "여수시", "영광군", "영암군", 
        "완도군", "장성군", "장흥군", "진도군", "함평군", 
        "해남군", "화순군",
        "경산시", "경주시", "고령군", "구미시", "김천시", 
        "문경시", "봉화군", "상주시", "성주군", "안동시", 
        "영덕군", "영양군", "영주시", "영천시", "예천군", 
        "울릉군", "울진군", "의성군", "청도군", "청송군", 
        "칠곡군", "포항시",
        "거제시", "거창군", "고성군", "김해시", "남해군", "밀양시", 
        "사천시", "산청군", "양산시", "의령군", "진주시", 
        "창녕군", "창원시", "통영시", "하동군", "함안군", 
        "함양군", "합천군",
        "서귀포시", "제주시",
        "강릉시", "고성군", "동해시", "삼척시", "속초시", 
        "양구군", "양양군", "영월군", "원주시", "인제군", 
        "정선군", "철원군", "춘천시", "태백시", "평창군", 
        "홍천군", "화천군", "횡성군",
        "고창군", "군산시", "김제시", "남원시", "무주군", 
        "부안군", "순창군", "완주군", "익산시", "임실군", 
        "장수군", "전주시", "덕진구", "완산구", 
        "정읍시", "진안군"],
    "gender": ["남자", "여자"],
    "income_category": ["0 ~ 50%", "51 ~ 75%", "76 ~ 100%", "101 ~ 200%"], 
    "personal_category": ["예비부부/난임", "임신부", "출산/입양", "장애인", "국가보훈대상자", "농업인", "어업인", "축산인", "임업인", "초등학생", "중학생", "고등학생", "대학생/대학원생", "질병/질환자", "근로자/직장인", "구직자/실업자", "해당사항 없음"],
    "household_category": ["다문화가정", "북한이탈주민가정", "한부모가정/조손가정", "1인 가구", "다자녀가구", "무주택세대", "신규전입가구", "확대가족", "해당사항 없음"],
    "support_type": ["현금", "현물", "서비스", "이용권"],
    "application_method": ["온라인 신청", "타사이트 신청", "방문 신청", "기타"],
    "benefit_category": ["생활안정", "주거-자립", "보육-교육", "고용-창업", "보건-의료", "행정-안전", "임신-출산", "보호-돌봄", "문화-환경", "농림축산어업"]
}

# 쿼리 캐시를 위한 딕셔너리와 쓰레드 세이프한 lock
query_cache = {}
cache_lock = threading.Lock()

def validate_variable(key: str, value: Optional[str]) -> Optional[str]:
    """
    변수의 유효성을 검사하는 함수.
    """
    if key in ALLOWED_VALUES:
        if value in ALLOWED_VALUES[key]:
            return value
        return None
    return value

def extract_variables_from_query(query: str) -> Dict[str, Optional[str]]:
    """
    사용자 쿼리에서 변수를 추출하는 함수. 캐싱 적용.
    """
    # 캐시 확인
    with cache_lock:
        if query in query_cache:
            return query_cache[query]
    
    prompt = f"""
    사용자가 입력한 쿼리에서 명시적으로 언급된 정보만 추출하세요. 추측하거나 기본값을 적용하지 마세요.
    
    추출할 변수, 설명, 그리고 허용되는 값은 다음과 같습니다:
    
    area: 최상위 행정구역. {", ".join(ALLOWED_VALUES["area"])}
    district: 차상위 행정구역. {", ".join(ALLOWED_VALUES["district"])}
    age: 나이. 숫자로 출력 (예: 30)
    gender: 성별. {", ".join(ALLOWED_VALUES["gender"])}
    income_category: 소득 백분율 분류. {", ".join(ALLOWED_VALUES["income_category"])}
    personal_category: 직업, 결혼유무 등등의 개인 특성. {", ".join(ALLOWED_VALUES["personal_category"])}
    household_category: 가구형태. {", ".join(ALLOWED_VALUES["household_category"])}
    benefit_category: 받고 싶은 혜택 분류. {", ".join(ALLOWED_VALUES["benefit_category"])}
    enddate: 혜택을 받고 싶은 마지막 날짜. YYYY-MM-DD 형식으로 출력. (예: 2023-12-31)
    startdate: 혜택을 받고 싶은 처음 날짜.YYYY-MM-DD 형식으로 출력. (예: 2023-01-01)
    source_data: 혜택 출처. 문자열로 출력.
    
    사용자 쿼리에서 발견된 정보가 위의 허용된 값과 정확히 일치하도록 매핑하세요.
    사용자가 명시적으로 언급하지 않은 필드는 반드시 빈 문자열("")로 설정하세요.
    추측하거나 기본값을 적용하지 마세요.
    
    쿼리: "{query}"
    
    결과는 반드시 순수 JSON 형식으로만 반환하세요. 마크다운 태그는 포함시키지 마세요.
    """
    
    try:
        # 더 빠른 모델 사용 (gpt-3.5-turbo)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # 더 빠른 모델로 변경
            messages=[
                {"role": "system", "content": "You are a helpful assistant for data extraction."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            timeout=10  # 10초 타임아웃 설정
        )
        
        raw_response = response.choices[0].message.content
        
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        result = json.loads(cleaned_response)
        
        # 결과 캐싱
        with cache_lock:
            query_cache[query] = result
            
        return result
    
    except Exception as e:
        print(f"API 호출 오류: {e}")
        # 오류 발생 시 빈 딕셔너리 반환
        return {}

def generate_sql_query(params: Dict[str, Union[Optional[str], List[str]]]) -> str:
    """
    매개변수를 기반으로 SQL 쿼리를 생성합니다.
    """
    # 기본 SELECT 절
    select_clause = """
    SELECT *
    FROM benefits
    """
    
    conditions = []
    
    # 1. area와 district 조건 처리
    area_value = params.get("area")
    district_value = params.get("district")
    
    if area_value and district_value:
        # 해당 area에 속하는 district 목록 가져오기
        valid_districts = AREA_DISTRICT_MAPPING.get(area_value, [])
        
        # district가 해당 area에 속하는지 확인
        if district_value in valid_districts:
            conditions.append(f"(AREA IN ('', '전국', '{area_value}') AND DISTRICT IN ('', '{district_value}'))")
        else:
            # district가 해당 area에 속하지 않으면 매치되는 항목이 없도록 조건 설정
            conditions.append("(AREA = 'invalid_area_district_pair')")
    else:
        # area만 있는 경우
        if area_value:
            if isinstance(area_value, list):
                area_values = ', '.join(f"'{a}'" for a in area_value)
                conditions.append(f"AREA IN ('', '전국', {area_values})")
            else:
                conditions.append(f"AREA IN ('', '전국', '{area_value}')")
        
        # district만 있는 경우
        if district_value:
            conditions.append(f"DISTRICT IN ('', '{district_value}')")
    
    # 나머지 조건들 추가
    if params.get("age"):
        conditions.append(f"(MIN_AGE IS NULL OR MIN_AGE <= {params['age']}) AND (MAX_AGE IS NULL OR MAX_AGE >= {params['age']})")
    
    if params.get("gender"):
        conditions.append(f"GENDER LIKE '%{params['gender']}%'")
    
    if params.get("income_category"):
        conditions.append(f"INCOME_CATEGORY LIKE '%{params['income_category']}%'")
    
    if params.get("personal_category"):
        conditions.append(f"PERSONAL_CATEGORY LIKE '%{params['personal_category']}%'")
    
    if params.get("household_category"):
        conditions.append(f"HOUSEHOLD_CATEGORY LIKE '%{params['household_category']}%'")
    
    if params.get("benefit_category"):
        conditions.append(f"BENEFIT_CATEGORY LIKE '%{params['benefit_category']}%'")
    if params.get("source_data"):
        conditions.append(f"SOURCE_DATA IN ('', '{params['source_data']}')")
    
    # 날짜 조건 처리
    if params.get("enddate"):
        conditions.append(f"(END_DATE IS NULL OR END_DATE > '{params['enddate']}')")
    else:
        # 명시적으로 종료 날짜가 지정되지 않은 경우, 오늘 날짜보다 종료일이 이후인 항목만 선택
        conditions.append("(END_DATE IS NULL OR END_DATE > CURRENT_DATE())")
        
    if params.get("startdate"):
        conditions.append(f"(START_DATE IS NULL OR START_DATE <= '{params['startdate']}')")
    
    if params.get("source_data"):
        conditions.append(f"SOURCE_DATA IN ('', '{params['source_data']}')")
    
    # WHERE 절 추가
    if conditions:
        query = select_clause + " WHERE " + " AND ".join(conditions)
    else:
        query = select_clause

    # 마지막에 세미콜론 추가
    query = query.strip() + ";"
    
    return query

def process_query(query):
    """
    단일 쿼리를 처리하는 함수 (병렬 처리용)
    """
    try:
        query_vars = extract_variables_from_query(query)
        sql_query = generate_sql_query(query_vars)
        return sql_query
    except Exception as e:
        print(f"쿼리 처리 오류: {e}")
        return "ERROR: " + str(e)

def main():
    # 시작 시간 기록
    total_start_time = time.time()
    
    # Excel 파일을 한 번에 읽습니다
    print("Excel 파일 로딩 중...")
    df = pd.read_excel("results_v1.1.xlsx")
    
    print(f"데이터프레임 크기: {len(df)}")
    
    # 결과를 저장할 새 열 생성
    df['generated_sql'] = ''
    
    # 진행 상황 추적을 위한 변수들
    total_rows = len(df)
    print(f"총 {total_rows}개의 쿼리를 처리합니다.")
    
    # 배치 크기 설정
    batch_size = 100
    
    # 배치 단위로 병렬 처리
    results = []
    
    # 데이터를 배치로 나누기
    batches = [df['query'].iloc[i:i+batch_size].tolist() for i in range(0, total_rows, batch_size)]
    
    # 각 배치를 병렬로 처리
    with tqdm(total=total_rows, desc="쿼리 처리 중") as pbar:
        for batch in batches:
            with ThreadPoolExecutor(max_workers=10) as executor:
                batch_results = list(executor.map(process_query, batch))
                results.extend(batch_results)
                pbar.update(len(batch))
    
    # 결과를 데이터프레임에 추가
    df['generated_sql'] = results
    
    # 필요한 칼럼만 선택
    df = df[['query', 'generated_sql']]
    
    # 결과를 저장
    df.to_csv("results_processed.csv", index=False)
    
    # 총 소요 시간 계산 및 출력
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    avg_time_per_query = total_time / total_rows
    
    print(f"\n처리 완료!")
    print(f"총 소요 시간: {total_time:.2f}초")
    print(f"쿼리당 평균 처리 시간: {avg_time_per_query:.2f}초")
    print(f"결과가 results_processed.csv에 저장되었습니다.")

if __name__ == "__main__":
    main()