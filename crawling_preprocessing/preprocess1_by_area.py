## 필요한 라이브러리 불러오기
import os
import time
import json
import re
from dotenv import load_dotenv
import pandas as pd
import concurrent.futures
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini Langchain 통합 모듈 임포트
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
import logging

## API 사용하기 전 기본 설정 및 모델 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API KEY 환경 변수 (사용하지 않지만 기존 코드 유지)
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")    # Gemini API KEY 추가
model = "gemini-2.0-flash"
logging.basicConfig(filename="processing.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Gemini 모델 초기화
llm = ChatGoogleGenerativeAI(model=model, google_api_key=GOOGLE_API_KEY, api_version="v1", temperature=0)

# 허용된 지역 및 지역구 값 설정
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
}

## 쓸모없는 특수문자 제거
def clean_text(text):
    if text is None:
        return ""  # None이면 빈 문자열 반환
    text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣A-Za-z0-9\s\-\~\<\>\.\:\/\@\%\+\*\#\°\₩\$]', '', text)
    text = re.sub(r'[\n\r]', ' ', text)
    text = text.lower()
    text = text.strip()
    return text

def make_prompt(title, content, area="", district=""):
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""당신은 혜택 정보를 분석하는 전문가입니다. 주어진 제목과 본문에서 혜택이 제공되는 지역을 추출해 JSON으로 반환하세요.

**지시:**
- 초기값: area="{area}", district="{district}".
- 초기값이 ""이 아니면, 텍스트와 일치할 경우 수정하지 마세요. 불일치 시에만 정확한 행정 구역명으로 수정하고, 수정 여부를 modified에 true로 표시하세요.
- 초기값이 ""이면, 텍스트에서 지역 정보를 추출해 채우세요.
- 여러 지역이 해당되면 중복 출력 가능.

**출력 형식:**
- area: 광역 행정 구역 (예: "서울특별시"). 정보 없으면 "".
- district: 기초 행정 구역 (예: "강남구"). 정보 없으면 "".
- area_summary: 추가 지역 정보 (예: "농어촌"). 없으면 "".
- modified: 초기값에서 수정했으면 true, 아니면 false.

**주의:**
- 반드시 유효한 JSON만 반환하세요. 설명이나 마크다운을 포함하지 마세요.

**예시:**
입력: 제목: "서울시 강남구 지원", 본문: "서울특별시 강남구에서 지원 제공."
출력: {{"area": "서울특별시", "district": "강남구", "area_summary": "", "modified": false}}
"""),
        HumanMessagePromptTemplate.from_template("""# 혜택 정보:
제목: {title}
본문: {content}

# JSON으로 응답:
""")
    ])
    return chat_template.format_messages(title=title, content=content)

def area_by_agency(agency_type, agency_name):
    """
    기관 유형과 기관명을 기반으로 지역 정보를 추출합니다.
    
    Args:
        agency_type (str): 기관 유형 (시군구, 광역시도, 교육청 등)
        agency_name (str): 기관명
        
    Returns:
        tuple: (area, district) 형태의 튜플
    """
    if agency_type == "시군구":
        # 예: "서울특별시 종로구" → ("서울특별시", "종로구")
        parts = agency_name.split(" ", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            return "", ""
    elif agency_type == "광역시도":
        # 예: "서울특별시" → ("서울특별시", "")
        return agency_name, ""
    elif agency_type == "교육청":
        # 예: "서울특별시교육청" → ("서울특별시", "")
        if agency_name.endswith("교육청"):
            return agency_name[:-3], ""
        else:
            return "", ""
    elif agency_type in {"지방공기업", "지방출자_출연기관"}:
        temp_area = ""
        temp_district = ""
        # 허용된 area 목록에서 접두어(예: "제주" from "제주특별자치도")가 기관명 시작부분과 일치하면 매칭
        for area in ALLOWED_VALUES.get("area", []):
            if "특별" in area:
                prefix = area.split("특별")[0]
            elif "광역" in area:
                prefix = area.split("광역")[0]
            else:
                prefix = area
            if agency_name.startswith(prefix):
                temp_area = area
        for district in ALLOWED_VALUES["district"]:
            if district[:-1] in agency_name:
                temp_district = district
        return temp_area, temp_district
    else:
        return "", ""

def extract_json_from_response(response_text):
    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_text = response_text[start_idx:end_idx]
            return json.loads(json_text)
        raise ValueError("JSON 형식을 찾을 수 없습니다.")
    except json.JSONDecodeError as e:
        return {"error": f"JSON 파싱 오류: {str(e)}", "raw_response": response_text[:200]}

def process_single_item(title, content, service_id, agency_type, agency_name, retry_delay, max_retries):
    logging.info(f"처리 시작: {service_id}")
    """
    단일 항목을 처리하여 혜택 정보를 추출합니다.
    
    Args:
        title (str): 혜택 서비스 제목
        content (str): 혜택 서비스 내용
        service_id (str): 서비스 ID
        agency_type (str): 기관 유형
        agency_name (str): 기관명
        retry_delay (int): 재시도 지연 시간(초)
        max_retries (int): 최대 재시도 횟수
        
    Returns:
        dict: 추출된 혜택 정보
    """
    # 기관 유형과 기관명에서 지역 정보 추출
    area, district = area_by_agency(agency_type, agency_name)
    
    # 프롬프트 생성
    messages = make_prompt(title, content, area, district)
    
    # 재시도 로직
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            benefit_info = extract_json_from_response(response.content)
            if "error" in benefit_info:
                benefit_info.update({"service_id": service_id, "title": title})
                return benefit_info
            benefit_info["service_id"] = service_id
            benefit_info["agency_name"] = agency_name
            benefit_info["agency_type"] = agency_type
            return benefit_info
        except Exception as e:
            time.sleep(retry_delay)
    return {"error": "최대 재시도 횟수 초과", "service_id": service_id, "title": title}

def process_data_parallel(titles, contents, service_ids, agency_types, agency_names, retry_delay, max_retries, max_workers=4):
    """
    데이터를 병렬로 처리합니다.
    
    Args:
        titles (list): 서비스 제목 리스트
        contents (list): 서비스 내용 리스트
        service_ids (list): 서비스 ID 리스트
        agency_types (list): 기관 유형 리스트
        agency_names (list): 기관명 리스트
        retry_delay (int): 재시도 지연 시간(초)
        max_retries (int): 최대 재시도 횟수
        max_workers (int): 최대 작업자 수
        
    Returns:
        list: 처리된 결과 리스트
    """
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                process_single_item, 
                title, content, service_id, agency_type, agency_name, 
                retry_delay, max_retries
            ): idx 
            for idx, (title, content, service_id, agency_type, agency_name) 
            in enumerate(zip(titles, contents, service_ids, agency_types, agency_names))
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                results.append(result)
                if (idx + 1) % 10 == 0:
                    print(f"처리 진행 상황: {idx + 1}/{len(titles)}")
            except Exception as e:
                logging.error(f"오류 발생: {e}, service_id: {service_ids[idx]}")
                results.append({
                    "title": titles[idx], 
                    "service_id": service_ids[idx], 
                    "agency_name": agency_names[idx],
                    "agency_type": agency_types[idx],
                    "error": str(e)
                })
    
    return results

def save_results_to_file_incrementally(results, output_file_path_filename, chunk_size=100):
    with open(output_file_path_filename, "w", encoding="utf-8") as f:
        f.write("[")
        for i, result in enumerate(results):
            if i > 0:
                f.write(",")
            json.dump(result, f, ensure_ascii=False)
            if (i + 1) % chunk_size == 0:
                f.flush()
        f.write("]")

if __name__ == "__main__":
    # 처리할 파일 불러오기 및 출력 경로
    file_path = r"C:\Users\r2com\Desktop\VSCode\bozo24\20250304.json"  
    output_folder_path = r'C:\Users\r2com\Desktop\VSCode\preprocessing'
    output_file_path_filename = os.path.join(output_folder_path, "preprocess_by_area.json")
    
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"출력 폴더가 생성되었습니다: {output_folder_path}")
    
    # 전처리에 사용할 칼럼들
    content_column_list = ["부서명", "사용자구분", "서비스목적요약", "서비스분야", "선정기준", "신청기한", "신청방법", "전화문의", "접수기관", "지원내용", "지원대상", "지원유형"]
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 원본 데이터와 함께 처리에 사용할 필드 추출
            titles = [item["서비스명"] for item in data]
            contents = ["\n".join(f"{col}: {clean_text(item.get(col, ''))}" for col in content_column_list) for item in data]
            service_ids = [item["서비스ID"] for item in data]
            agency_types = [item.get("소관기관유형", "") for item in data]
            agency_names = [item.get("소관기관명", "") for item in data]
            
            print(f"총 {len(titles)}개의 데이터를 처리합니다.")
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        exit(1)
    except json.JSONDecodeError:
        print("JSON 파일 형식이 올바르지 않습니다.")
        exit(1)
    except Exception as e:
        print(f"데이터 로딩 중 오류 발생: {e}")
        exit(1)
    
    # 재시도 설정
    max_retries = 3
    retry_delay = 1  # 초 단위
    
    # 테스트 모드 (예: 2000~2200 인덱스 처리)
    test_mode = False
    start, end = 2000, 2200
    if test_mode:
        titles = titles[start:end]
        contents = contents[start:end]
        service_ids = service_ids[start:end]
        agency_types = agency_types[start:end]
        agency_names = agency_names[start:end]
        print("테스트 모드: 선택한 데이터만 처리합니다.")
    
    print("데이터 처리를 시작합니다...")
    max_workers = min(os.cpu_count() or 4, 8)
    results = process_data_parallel(titles, contents, service_ids, agency_types, agency_names, retry_delay, max_retries, max_workers)
    
    # 처리 결과를 서비스ID를 기준으로 딕셔너리로 변환 (빠른 조회를 위함)
    result_dict = {item["service_id"]: item for item in results if "service_id" in item}
    
    # 원본 데이터에 처리 결과(지역 정보)를 병합합니다.
    for item in data:
        sid = item.get("서비스ID")
        if sid in result_dict:
            # 결과의 key들 (area, district, area_summary, modified) 추가
            item["area"] = result_dict[sid].get("area", "")
            item["district"] = result_dict[sid].get("district", "")
            item["area_summary"] = result_dict[sid].get("area_summary", "")
            item["modified"] = result_dict[sid].get("modified", False)
        else:
            # 처리 결과가 없을 경우 기본값 추가
            item["area"] = ""
            item["district"] = ""
            item["area_summary"] = ""
            item["modified"] = False
    
    # 최종 결과를 저장 (병합된 데이터를 출력)
    final_output_path = os.path.join(output_folder_path, "final_merged_output.json")
    try:
        with open(final_output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n최종 병합 결과가 '{final_output_path}' 파일에 저장되었습니다.")
        print(f"총 {len(data)}개의 항목이 처리되었습니다.")
    except Exception as e:
        print(f"최종 결과 저장 중 오류 발생: {e}")
