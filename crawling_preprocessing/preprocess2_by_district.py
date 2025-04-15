import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini Langchain 통합 모듈 임포트
import logging
from dotenv import load_dotenv
# 로깅 설정 (필요 시)
logging.basicConfig(filename="processing.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# llm 객체 (이미 초기화된 것으로 가정)
## API 사용하기 전 기본 설정 및 모델 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API KEY 환경 변수 (사용하지 않지만 기존 코드 유지)
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")    # Gemini API KEY 추가
model = "gemini-2.0-flash"
logging.basicConfig(filename="processing.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Gemini 모델 초기화
llm = ChatGoogleGenerativeAI(model=model, google_api_key=GOOGLE_API_KEY, api_version="v1", temperature=0)

# 미리 정의된 area-district 매핑
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

def resolve_ambiguous_area(district, item, candidate_areas):
    """
    district가 여러 area 후보에 해당할 경우, LLM에게 추가 프롬프트를 보내어 
    후보 목록(candidate_areas) 중 적절한 area를 결정합니다.
    
    Args:
        district (str): 모호한 기초 행정 구역명 (예: "동구")
        item (dict): 원본 데이터 항목 (서비스명, 지원내용 등 문맥 제공)
        candidate_areas (list): 해당 district에 해당하는 후보 area 리스트
        
    Returns:
        str: 선택된 area (없으면 빈 문자열)
    """
    title = item.get("서비스명", "")
    support_content = item.get("지원내용", "")
    
    prompt = f"""당신은 행정구역 전문가입니다. 다음 정보를 참고하여 모호한 district '{district}'에 대해 아래 후보 중 
가장 적합한 광역 행정 구역(area)를 하나 선택하세요.

[후보 목록]:
{', '.join(candidate_areas)}

[서비스 정보]:
{json.dumps(item, ensure_ascii=False, indent=2)}

출력은 반드시 유효한 JSON 형식으로 다음과 같이 응답하세요:
{{"chosen_area": "선택한 area명"}}
"""
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        res_json = json.loads(response.content)
        chosen_area = res_json.get("chosen_area", "")
        return chosen_area
    except Exception as e:
        return candidate_areas[0] if candidate_areas else ""

# 입력 및 출력 파일 경로 설정
file_path = r"preprocessing\final_merged_output.json"  
out_path = r"preprocessing\area_preprocessed_output1.json"

# 결과 파일 읽기
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 이미 처리 결과가 저장된 result_dict가 있다고 가정합니다.
# 예: result_dict = {서비스ID: { "area": ..., "district": ..., "area_summary": ..., "modified": ...}, ...}
# result_dict가 없다면, 이 부분은 기존 처리 결과와 병합하는 로직에 맞게 수정하세요.

# 최종 결과 병합 단계 (원본 데이터에 처리 결과를 추가)
for item in data:
    item["area"] = item.get("area", "")
    item["district"] = item.get("district", "")
    item["area_summary"] = item.get("area_summary", "")
    item["modified"] = item.get("modified", False)
        
    # district가 존재할 경우 매핑을 통해 area 채우기
    if item["district"]:
        candidate_areas = [area_key for area_key, districts in AREA_DISTRICT_MAPPING.items() 
                            if item["district"] in districts]
        if candidate_areas:
            if item["area"] in candidate_areas:
                pass  # 이미 적절한 area가 설정된 경우
            else:
                if len(candidate_areas) == 1:
                    item["area"] = candidate_areas[0]
                else:
                    chosen_area = resolve_ambiguous_area(item["district"], item, candidate_areas)
                    item["area"] = chosen_area
# 처리 결과가 없는 경우는 그대로 두어 추가 변경 없이 진행

# 결과 파일 저장
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"처리 결과가 '{out_path}'에 저장되었습니다.")
