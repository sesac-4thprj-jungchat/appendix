import csv
import openai
import pymysql
import random
from contextlib import closing

# 환경변수를 통해 API 키를 관리하는 것을 권장 (예: os.environ)
openai.api_key = "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"  
model = "gpt-4o-mini"

# 예시 데이터
areas = [
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", 
    "대전광역시", "울산광역시", "세종특별자치시", "경기도", "충청북도", 
    "충청남도", "전라남도", "경상북도", "경상남도", "제주특별자치도", 
    "강원특별자치도", "전북특별자치도"
]

districts_map = {
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
        "장수군", "전주시", "전주시 덕진구", "전주시 완산구", 
        "정읍시", "진안군"
    ]
}

ages = range(20, 70)
genders = ["남자", "여자"]
income_ranges = [
    "0 ~ 50%", 
    "51 ~ 75%", 
    "76 ~ 100%", 
    "101 ~ 200%", 
    ]
personal_characteristics = [
    "예비부부/난임", "임신부", "출산/입양", "장애인", "국가보훈대상자", "농업인", 
    "어업인", "축산인", "임업인", "초등학생", "중학생", "고등학생", 
    "대학생/대학원생", "질병/질환자", "근로자/직장인", "구직자/실업자", "해당사항 없음"
]
household_characteristics = [
    "다문화가정", "북한이탈주민가정", "한부모가정/조손가정", "1인 가구", "다자녀가구", 
    "무주택세대", "신규전입가구", "확대가족", "해당사항 없음"
]
support_type = ["현금", "현물", "서비스", "이용권"]
application_method = ["온라인 신청", "타사이트 신청", "방문 신청", "기타"]
benefit_category = [
    "생활안정", "주거-자립", "보육-교육", "고용-창업", "보건-의료", 
    "행정-안전", "임신-출산", "보호-돌봄", "문화-환경", "농림축산어업"
]



def generate_queries(num_queries):
    """다양한 자연어 쿼리를 생성"""
    queries = []
    for _ in range(num_queries):
        area = random.choice(areas)
        district = random.choice(districts_map[area])
        age = random.choice(ages)
        gender = random.choice(genders)
        income = random.choice(income_ranges)
        personal = random.choice(personal_characteristics)
        household = random.choice(household_characteristics)
        # support = random.choice(support_type)
        # application = random.choice(application_method)
        # category = random.choice(benefit_category)
        
        # 다양한 쿼리 패턴
        queries.append(f"{district}에 사는 {age}세 {gender} {personal}에게 적합한  지원을 알려줘")
        queries.append(f"소득이 {income}인 {district} 거주 {age}세 {gender}에게 맞는 방법의 혜택을 알려줘")
        queries.append(f"{district}에 사는 {age}세 {gender} {personal}을 위한 프로그램을 알려줘")
        queries.append(f"{district}에 거주하는 {age}세 {gender} {household}을 위한 지원을 보여줘")
        
    return queries

def chatgpt_generate(query):
    """OpenAI API를 호출해 응답을 생성"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ]
    response = openai.ChatCompletion.create(model=model, messages=messages)
    answer = response['choices'][0]['message']['content']
    return answer

def query_to_sql(natural_query):
    
    """자연어 쿼리를 기반으로 SQL 문을 생성"""
    api_query = """당신은 주어진 MySQL 데이터베이스 스키마를 활용하여 적절한 SQL 쿼리를 생성하는 전문가입니다.
    아래 테이블을 참고하여 사용자의 질문을 해결할 수 있는 최적의 SQL 쿼리를 작성하세요.

    **SQL Dialect:** MySQL

    ### 데이터베이스 스키마:
    **테이블 1: benefits**
    **컬럼:**
    id INT AUTO_INCREMENT PRIMARY KEY,
    area VARCHAR(50) NOT NULL,  -- 지역 정보
    district VARCHAR(100),      -- 구/군 정보
    min_age INT,                -- 최소 연령
    max_age INT,                -- 최대 연령
    age_summary VARCHAR(255),   -- 연령 요약
    gender VARCHAR(10),         -- 성별
    income_category VARCHAR(50),-- 소득 카테고리
    income_summary VARCHAR(255),-- 소득 요약
    personal_category VARCHAR(255), -- 개인 카테고리
    personal_summary TEXT,      -- 개인 요약
    household_category VARCHAR(255), -- 가구 카테고리
    household_summary TEXT,     -- 가구 요약
    support_type VARCHAR(50),   -- 지원 유형
    support_summary TEXT,       -- 지원 요약
    application_method VARCHAR(100), -- 신청 방법
    application_summary TEXT,   -- 신청 요약
    benefit_category VARCHAR(100), -- 혜택 카테고리
    benefit_summary TEXT,       -- 혜택 요약
    start_date DATE,            -- 시작 날짜
    end_date DATE,              -- 종료 날짜
    date_summary VARCHAR(255),  -- 날짜 요약
    benefit_details TEXT,       -- 혜택 세부사항
    source VARCHAR(255),        -- 출처
    additional_data VARCHAR(10),-- 추가 데이터
    keywords TEXT,              -- 키워드
    service_id VARCHAR(50) UNIQUE -- 서비스 ID

    ### 질문 유형 예시 및 SQL 쿼리 예시:

    1. "30~50세 대상 혜택을 검색해줘."
    ```sql
        SELECT title, min_age, max_age, benefit_summary 
        FROM benefits 
        WHERE min_age <= 30 AND max_age >= 50;
        ```

    2. "여성만 신청할 수 있는 혜택을 찾아줘."
    ```sql
        SELECT title, gender, benefit_summary 
        FROM benefits 
        WHERE gender = '여성';
        ```


    3. "현재 신청 가능한 혜택 목록을 보여줘."
    ```sql
        SELECT title, application_method 
        FROM benefits 
        WHERE start_date <= CURDATE() AND end_date >= CURDATE();
        ```

    4. "청년 관련 혜택을 찾아줘."
    ```sql
        SELECT title, keywords, benefit_summary 
        FROM benefits 
        WHERE keywords LIKE '%청년%';

    ### Question:
    쿼리: {natural_query}

    ### SQL:
    """

    answer = chatgpt_generate(api_query)
    return answer

def go_db(sql):
    """MySQL 데이터베이스에 연결하여 SQL 실행 및 결과 반환"""
    try:
        conn = pymysql.connect(
            host='127.0.0.1', user='root', password='1234', 
            db='final_project', charset='utf8'
        )
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                result = cur.fetchall()
        return result
    except pymysql.MySQLError as e:
        raise pymysql.MySQLError(f"데이터베이스 오류: {e}")

def make_natural_answer(natural_query, sql_result):
    """SQL 결과를 기반으로 자연어 응답 생성"""
    sql_result_str = "\n".join([str(item) for item in sql_result])
    prompt = f"""다음과 같은 쿼리가 주어졌을 때, 쿼리로 데이터베이스를 조회한 결과가 있다.
쿼리와 데이터베이스 조회 결과로 답변 문장을 자연스럽게 생성하시오.

쿼리: {natural_query}
데이터베이스 조회 결과:
{sql_result_str}
"""
    print(prompt)
    print('-----------------------------')
    answer = chatgpt_generate(prompt)
    return answer

def main():
    queries = generate_queries(5)
    results = []  # 결과 저장 리스트
    for query in queries:
        try:
            answer = query_to_sql(query)
            start = answer.index("SELECT")
            end = answer.index(";")
            sql = answer[start:end+1]
            print(sql)
            print('------------------------')
            sql_result = go_db(sql)
            natural_answer = make_natural_answer(query, sql_result)
            print(natural_answer)
            results.append((query, sql, natural_answer))
        except ValueError:
            print("SQL 변환에 실패했습니다.")
        except pymysql.MySQLError as e:
            print(e)
    
    # CSV 파일에 결과 기록
    with open("testSet.csv", mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(["query", "sql", "natural_answer"])
        writer.writerows(results)

if __name__ == "__main__":
    main()