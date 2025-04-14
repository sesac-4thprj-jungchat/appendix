import re
import pandas as pd
import datetime
import calendar
from datetime import date
from typing import Tuple, List, Dict, Optional, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API 사용하기 전 기본 설정 및 모델 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
model = "gemini-2.0-flash"
llm = ChatGoogleGenerativeAI(model=model, google_api_key=GOOGLE_API_KEY, api_version="v1", temperature=0)

# 결과 캐싱을 위한 딕셔너리 (LLM 중복 호출 방지)
llm_cache: Dict[str, Tuple[str, str, str]] = {}

def llm_process_dates(timeline: str, item: str, base_date: Optional[date] = None) -> Tuple[str, str, str]:
    """
    날짜 정보가 기존 처리 규칙으로 처리하기 힘든 경우, LLM을 호출해서 처리합니다.
    캐싱을 적용하여 동일한 입력에 대해 중복 호출을 방지합니다.
    
    Args:
        timeline: 처리할 신청기간 텍스트
        item: 추가 컨텍스트 정보
        base_date: 기준 날짜 (기본값: 오늘)
        
    Returns:
        시작 날짜(YYYY-MM-DD), 종료 날짜(YYYY-MM-DD), 설명
    """
    if base_date is None:
        base_date = date.today()
    
    current_year = base_date.year
    
    # 캐시 키 생성
    cache_key = f"{timeline}_{item[:100]}"  # 아이템은 너무 길 수 있으므로 앞부분만 사용
    
    if cache_key in llm_cache:
        return llm_cache[cache_key]
    
    prompt = (
        f"아래 텍스트에서 혜택 신청 시작 날짜와 종료 날짜를 추출해줘.\n\n"
        f"신청기간: {timeline}\n"
        f"내용: {item}\n\n"
        f"현재 연도는 {current_year}년이고, 날짜가 모호한 경우 이를 기준으로 판단해줘.\n\n"
        f"'접수기관별 상이', '상반기 3월, 하반기 9월', '2025년 4월중' 등 특수한 표현이 있을 경우:\n"
        f"- '상반기'는 해당 연도 1월 1일부터 6월 30일까지\n"
        f"- '하반기'는 해당 연도 7월 1일부터 12월 31일까지\n"
        f"- '상반기 3월'은 해당 연도 3월 1일부터 3월 31일까지\n"
        f"- '하반기 9월'은 해당 연도 9월 1일부터 9월 30일까지\n"
        f"- '연중'이나 '수시'는 해당 연도 1월 1일부터 12월 31일까지\n"
        f"- '매월'은 가장 가까운 미래의 해당 월 전체 기간\n\n"
        "결과를 JSON 형식으로 반환해줘(추출 불가능한 경우 빈 문자열 반환): {\"start_date\": \"YYYY-MM-DD\", \"end_date\": \"YYYY-MM-DD\", \"date_summary\": \"추가 설명\"}"
    )
    
    try:
        for attempt in range(3):
            try:
                response = llm.invoke([{"role": "user", "content": prompt}])
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
                raise e
                
        try:
            json_match = re.search(r'({.*?})', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                res_json = json.loads(json_str)
            else:
                res_json = json.loads(response.content)
                
            start_date = res_json.get("start_date", "")
            end_date = res_json.get("end_date", "")
            date_summary = res_json.get("date_summary", "")
            
        except json.JSONDecodeError:
            start_date_match = re.search(r'start_date[":]?\s*[":]?(\d{4}-\d{2}-\d{2})', response.content)
            end_date_match = re.search(r'end_date[":]?\s*[":]?(\d{4}-\d{2}-\d{2})', response.content)
            
            start_date = start_date_match.group(1) if start_date_match else ""
            end_date = end_date_match.group(1) if end_date_match else ""
            date_summary = "정규식으로 추출"
            
        if start_date:
            try:
                datetime.datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                start_date = ""
                
        if end_date:
            try:
                datetime.datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                end_date = ""
        
        result = (start_date, end_date, date_summary)
        llm_cache[cache_key] = result
        return result
    except Exception as e:
        logging.error("LLM 호출 오류: %s", str(e))
        result = ("", "", f"에러: {str(e)}")
        llm_cache[cache_key] = result
        return result

# 불필요한 special_month_range 함수는 제거(또는 추후 필요 시 별도 모듈로 관리)

# 전역 또는 함수 내부에 패턴 리스트 정의
DATE_PATTERNS = [
    re.compile(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일'),
    re.compile(r'(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?'),
    re.compile(r'(\d{4})-(\d{1,2})-(\d{1,2})'),
    re.compile(r'(\d{4})(\d{2})(\d{2})'),
    re.compile(r'(\d{4})년\s*(\d{1,2})월(?:중|초|말|하순|상순|중순)?'),
    re.compile(r'(\d{4})\.\s*(\d{1,2})\.?'),
    re.compile(r'(\d{4})-(\d{1,2})'),
    re.compile(r'(\d{4})(\d{2})'),
    re.compile(r'(\d{1,2})월(?:중|초|말|하순|상순|중순)?')
]

def extract_single_date(text: str, current_year: int, is_start: bool = True) -> Tuple[str, str]:
    """
    단일 날짜 문자열에서 날짜 정보를 추출합니다.
    
    Args:
        text: 날짜를 추출할 텍스트
        current_year: 현재 연도
        is_start: 시작 날짜인지 여부
        
    Returns:
        (날짜 문자열, 요약 설명) 튜플
    """
    text = re.sub(r'\.\s+', '.', text)
    text = re.sub(r'\s+', ' ', text).strip()

    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                groups = match.groups()
                if len(groups) == 3:
                    year, month, day = groups
                    datetime.date(int(year), int(month), int(day))
                    date_str = f"{year}-{int(month):02d}-{int(day):02d}"
                    return date_str, "날짜 (일 포함)"
                elif len(groups) >= 2:
                    year_match = re.search(r'\d{4}', text)
                    year = year_match.group(0) if year_match else str(current_year)
                    month = groups[-1]
                    month_int = int(month)
                    if month_int < 1 or month_int > 12:
                        return "", "월 범위 오류"
                    day = '01' if is_start else str(calendar.monthrange(int(year), month_int)[1])
                    date_str = f"{year}-{month_int:02d}-{int(day):02d}"
                    return date_str, f"날짜 (월 포함, {'시작' if is_start else '종료'})"
            except ValueError:
                continue
    return "", "날짜 추출 실패"

def _llm_fallback(text: str, index: int, raw_data: Optional[pd.DataFrame] = None, 
                  base_date: Optional[date] = None) -> Tuple[str, str, str]:
    """
    LLM fallback 로직을 담는 내부 함수
    
    Args:
        text: 날짜 텍스트
        index: 데이터 인덱스
        raw_data: 원본 데이터프레임
        base_date: 기준 날짜
        
    Returns:
        (시작 날짜, 종료 날짜, 설명) 튜플
    """
    content = ""
    if raw_data is not None and index < len(raw_data):
        content_column_list = ["서비스목적요약", "서비스분야", "선정기준", "신청방법", "전화문의", "접수기관", "지원내용", "지원대상"]
        content = "\n".join(f"{col}: {raw_data.iloc[index].get(col, '')}" 
                            for col in content_column_list if col in raw_data.columns)

    llm_start, llm_end, date_summary = llm_process_dates(text, content, base_date=base_date)
    if llm_start or llm_end:
        return llm_start, llm_end, "5. LLM fallback: " + date_summary

    return "", "", "6. 처리 실패"

def extract_dates(text: str, index: int, raw_data: Optional[pd.DataFrame] = None, 
                  base_date: Optional[str] = None) -> Tuple[str, str, str]:
    """
    텍스트에서 날짜 범위를 추출하는 함수.
    
    Args:
        text: 날짜를 추출할 텍스트
        index: raw_data에서 참조할 인덱스
        raw_data: 추가 정보가 포함된 데이터프레임
        base_date: 기준 날짜 (YYYY-MM-DD 형식)
    
    Returns:
        (시작 날짜, 종료 날짜, 요약 설명) 튜플
    """
    if not text or not isinstance(text, str) or not text.strip():
        return "", "", "빈 텍스트"

    if base_date is None:
        base_date_obj = date.today()
    else:
        try:
            base_date_obj = datetime.datetime.strptime(base_date, "%Y-%m-%d").date()
        except ValueError:
            base_date_obj = date.today()

    current_year = base_date_obj.year
    text = text.strip()
    
    split_parts = re.split(r'\s*(?:~|부터)\s*', text)

    if len(split_parts) > 2:
        return _llm_fallback(text, index, raw_data, base_date_obj) 
    elif len(split_parts) == 2:
        start_part, end_part = split_parts
        start_date, start_summary = extract_single_date(start_part.strip(), current_year, is_start=True)
        end_date, end_summary = extract_single_date(end_part.strip(), current_year, is_start=False)

        if start_date and end_date:
            try:
                s_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
                e_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
                
                if s_date > e_date:
                    if "년" not in end_part and "." not in end_part and "-" not in end_part:
                        year = s_date.year
                        month = e_date.month
                        day = e_date.day
                        end_date = f"{year}-{month:02d}-{day:02d}"
                        
                        if s_date > datetime.datetime.strptime(end_date, "%Y-%m-%d").date():
                            end_date = f"{year+1}-{month:02d}-{day:02d}"
            except ValueError:
                return _llm_fallback(text, index, raw_data, base_date_obj)
                
            return start_date, end_date, f"~ 기호 분리 처리: {start_summary}, {end_summary}"
        else:
            return _llm_fallback(text, index, raw_data, base_date_obj)
    else:
        special_cases = {
            '상반기': (f'{current_year}-01-01', f'{current_year}-06-30'),
            '하반기': (f'{current_year}-07-01', f'{current_year}-12-31'),
            '연중': (f'{current_year}-01-01', f'{current_year}-12-31'),
            '상시신청': (f'{current_year}-01-01', f'{current_year}-12-31'),
            '수시': (f'{current_year}-01-01', f'{current_year}-12-31'),
            '연중신청': (f'{current_year}-01-01', f'{current_year}-12-31'),
            '상시': (f'{current_year}-01-01', f'{current_year}-12-31'),
        }
        for key, (start, end) in special_cases.items():
            if re.fullmatch(rf'\s*{re.escape(key)}\s*', text):
                return start, end, "1. 특수 표현 처리 (규칙)"
            if re.search(rf'\b{re.escape(key)}\b', text) and len(text) > 8:
                return _llm_fallback(text, index, raw_data, base_date_obj)
        
        half_year_month_pattern = r'(상반기|하반기)\s*(\d{1,2})월'
        half_year_month_matches = re.findall(half_year_month_pattern, text)
        if half_year_month_matches:
            results = []
            for half, month_str in half_year_month_matches:
                try:
                    month = int(month_str)
                    if 1 <= month <= 12:
                        last_day = calendar.monthrange(current_year, month)[1]
                        start_date = f"{current_year}-{month:02d}-01"
                        end_date = f"{current_year}-{month:02d}-{last_day:02d}"
                        results.append((start_date, end_date))
                except ValueError:
                    continue
            if results:
                results.sort()
                return results[0][0], results[-1][1], "2. 상반기/하반기 월 패턴 (규칙)"
        
        year_month_pattern = r'(\d{4})년\s*(\d{1,2})월(?:중|초|말|하순|상순|중순)?'
        m = re.search(year_month_pattern, text)
        if m:
            try:
                year = int(m.group(1))
                month = int(m.group(2))
                if 1 <= month <= 12:
                    last_day = calendar.monthrange(year, month)[1]
                    start_date = f"{year}-{month:02d}-01"
                    end_date = f"{year}-{month:02d}-{last_day:02d}"
                    return start_date, end_date, "3. 연도-월 명시 패턴 (규칙)"
            except ValueError:
                pass
        
        monthly_specific_pattern = r'매월\s*(\d{1,2})일\s*부터\s*(?:말일|끝일)'
        m_specific = re.search(monthly_specific_pattern, text)
        if m_specific:
            try:
                start_day_num = int(m_specific.group(1))
                year = base_date_obj.year
                month = base_date_obj.month
                last_day = calendar.monthrange(year, month)[1]
                if start_day_num < 1 or start_day_num > last_day:
                    start_day_num = 1
                start_date_obj = date(year, month, start_day_num)
                end_date_obj = date(year, month, last_day)
                return start_date_obj.strftime("%Y-%m-%d"), end_date_obj.strftime("%Y-%m-%d"), "4a. 매월 구체적 패턴 (시작일 명시)"
            except ValueError:
                pass
        
        monthly_deadline_pattern = r'매월\s*(\d{1,2})일까지'
        m_deadline = re.search(monthly_deadline_pattern, text)
        if m_deadline:
            try:
                end_day = int(m_deadline.group(1))
                year = base_date_obj.year
                month = base_date_obj.month
                last_day = calendar.monthrange(year, month)[1]
                if end_day < 1 or end_day > last_day:
                    end_day = last_day
                start_date_obj = date(year, month, 1)
                end_date_obj = date(year, month, end_day)
                return start_date_obj.strftime("%Y-%m-%d"), end_date_obj.strftime("%Y-%m-%d"), "4b. 매월 마감일 명시 패턴"
            except ValueError:
                pass
    
    return _llm_fallback(text, index, raw_data, base_date_obj)

def process_batch(batch_data: List[str], batch_indices: List[int], 
                  raw_data: Optional[pd.DataFrame], base_date: date) -> List[Dict[str, Any]]:
    results = []
    for text, idx in zip(batch_data, batch_indices):
        if not isinstance(text, str):
            text = str(text)
        text = text.strip('\' ')
        if text:
            start_date, end_date, date_summary = extract_dates(
                text, idx, raw_data, base_date.strftime("%Y-%m-%d"))
            results.append({
                'index': idx,
                'original_text': text,
                'start_date': start_date,
                'end_date': end_date,
                'date_summary': date_summary if isinstance(date_summary, str) else ""
            })
        else:
            results.append({
                'index': idx,
                'original_text': "",
                'start_date': "",
                'end_date': "",
                'date_summary': "빈 텍스트"
            })
    return results

def get_today_date() -> str:
    """오늘 날짜를 YYYY-MM-DD 형식으로 반환합니다."""
    today = date.today()
    return today.strftime("%Y-%m-%d")

if __name__ == "__main__":
    input_file = r"preprocessing\area_preprocessed_output1.json"
    output_file = "bydate_partial.csv"
    
    try:
        logging.info("데이터 로딩 시작...")
        raw_data = pd.read_json(input_file)
        data = raw_data["신청기한"].tolist()
        
        today_date = get_today_date()
        logging.info("병렬 처리 시작...")
        
        import multiprocessing
        max_workers = min(multiprocessing.cpu_count(), 8)
        logging.info(f"사용할 worker 수: {max_workers}")
        
        total_processed = 0
        # 기존 출력 파일 삭제
        if os.path.exists(output_file):
            os.remove(output_file)
        
        start_time = time.time()
        batch_count = (len(data) + 50 - 1) // 50  # 배치 크기 50
        
        all_futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(batch_count):
                start_idx = i * 50
                end_idx = min(start_idx + 50, len(data))
                batch_data = data[start_idx:end_idx]
                batch_indices = list(range(start_idx, end_idx))
                all_futures.append(
                    executor.submit(process_batch, batch_data, batch_indices, raw_data, 
                                    datetime.datetime.strptime(today_date, "%Y-%m-%d").date())
                )
            
            # futures가 완료되는 대로 바로 CSV에 저장
            for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="Processing batches"):
                batch_results = future.result()
                total_processed += len(batch_results)
                df_partial = pd.DataFrame(batch_results)
                df_partial.to_csv(output_file, mode='a', index=False, 
                                  header=not os.path.exists(output_file), encoding="utf-8-sig")
                logging.info(f"{total_processed}건 처리 후 저장됨.")
        
        end_time = time.time()
        logging.info(f"처리 완료! 총 {total_processed}건, 소요 시간: {end_time - start_time:.2f}초")
        logging.info(f"LLM API 호출 횟수: {len(llm_cache)}")
        
        df_results = pd.read_csv(output_file)
        df_results = df_results.sort_values(by='index')
        
        pattern_counts = df_results['date_summary'].str.split('.').str[0].value_counts()
        logging.info("패턴별 처리 통계:")
        for pattern, count in pattern_counts.items():
            logging.info(f"패턴 {pattern}: {count}개 ({count/len(df_results)*100:.1f}%)")
        
        raw_data["start_date"] = df_results["start_date"].values
        raw_data["end_date"] = df_results["end_date"].values
        raw_data["date_summary"] = df_results["date_summary"].values
        
        raw_data.to_csv("bydate.csv", index=False, encoding="utf-8-sig")
        logging.info("최종 CSV 파일이 'bydate.csv' 이름으로 저장되었습니다.")
        
    except Exception as e:
        logging.exception("처리 중 오류가 발생했습니다: %s", e)
