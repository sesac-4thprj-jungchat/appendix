import pymysql
import pandas as pd
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

# MySQL 데이터베이스 연결 설정
DB_CONFIG = {
    "host": "testdb.cfmq6wqw0199.ap-northeast-2.rds.amazonaws.com",
    "user": "admin",
    "password": "Saltlux12345!",
    "port": 3306,
    "database": "multimodal_final_project",
    "charset": "utf8mb4"
}

# 결과를 저장하기 위한 리스트와 락
valid_queries = []
invalid_queries = []
results_lock = threading.Lock()

def get_db_connection():
    """데이터베이스 연결을 생성하고 반환"""
    return pymysql.connect(**DB_CONFIG)

def test_sql_query(row):
    """SQL 쿼리를 실행하고 결과를 반환합니다."""
    original_query = row['query']
    sql_query = row['generated_sql']
    
    # SQL 쿼리에 오류 메시지가 포함되어 있는지 확인
    if isinstance(sql_query, str) and sql_query.startswith("ERROR:"):
        with results_lock:
            invalid_queries.append({
                'query': original_query,
                'generated_sql': sql_query,
                'error_message': "Invalid SQL format"
            })
        return
    
    # 각 스레드마다 새 연결 사용
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # LIMIT 1을 추가하여 빠르게 유효성 검사만 수행
            # 원래 쿼리의 세미콜론을 제거하고 새로운 LIMIT 절 추가
            test_query = sql_query.rstrip(';') + " LIMIT 1;"
            cursor.execute(test_query)
            
            # 유효성만 확인하고 쿼리 자체는 원본 사용
            with results_lock:
                valid_queries.append({
                    'query': original_query,
                    'generated_sql': sql_query,
                    'is_valid': True
                })
    except Exception as e:
        # 오류 정보 저장
        with results_lock:
            invalid_queries.append({
                'query': original_query,
                'generated_sql': sql_query,
                'error_message': str(e)
            })
    finally:
        if connection:
            connection.close()

def main():
    # 시작 시간 기록
    start_time = time.time()
    
    # CSV 파일에서 쿼리와 생성된 SQL 읽기
    print("CSV 파일 로딩 중...")
    df = pd.read_csv("results_processed.csv")
    total_queries = len(df)
    print(f"총 {total_queries}개의 SQL 쿼리를 테스트합니다.")
    
    # 데이터베이스 연결 테스트
    try:
        test_connection = get_db_connection()
        test_connection.close()
        print("데이터베이스 연결 성공!")
    except Exception as e:
        print(f"데이터베이스 연결 실패: {e}")
        return
    
    # 배치 크기 설정
    batch_size = 1000
    max_workers = 20  # 병렬 실행할 최대 스레드 수
    
    # 데이터를 배치로 나누기
    batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    with tqdm(total=total_queries, desc="SQL 쿼리 테스트 중") as pbar:
        for batch_idx, batch in enumerate(batches):
            # 각 배치를 병렬로 처리
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 병렬 실행하고 결과는 무시 (valid_queries와 invalid_queries에 직접 추가됨)
                list(executor.map(test_sql_query, [row for _, row in batch.iterrows()]))
                
            # 진행 상황 업데이트
            pbar.update(len(batch))
            
            # 메모리 관리를 위해 주기적으로 중간 결과 저장
            if (batch_idx + 1) % 5 == 0 or batch_idx == len(batches) - 1:
                with results_lock:
                    if valid_queries:
                        temp_valid_df = pd.DataFrame(valid_queries)
                        temp_valid_df.to_csv(f"valid_queries_temp_batch_{batch_idx}.csv", index=False)
                        
                    if invalid_queries:
                        temp_invalid_df = pd.DataFrame(invalid_queries)
                        temp_invalid_df.to_csv(f"invalid_queries_temp_batch_{batch_idx}.csv", index=False)
                        
                    print(f"임시 저장 완료: {len(valid_queries)}개 유효 쿼리, {len(invalid_queries)}개 유효하지 않은 쿼리")
    
    # 최종 결과 저장
    with results_lock:
        if valid_queries:
            valid_df = pd.DataFrame(valid_queries)
            valid_df.to_csv("valid_queries.csv", index=False)
            valid_df.to_excel("valid_queries.xlsx", index=False)
            print(f"유효한 쿼리 수: {len(valid_df)}")
        
        if invalid_queries:
            invalid_df = pd.DataFrame(invalid_queries)
            invalid_df.to_csv("invalid_queries.csv", index=False)
            invalid_df.to_excel("invalid_queries.xlsx", index=False)
            print(f"유효하지 않은 쿼리 수: {len(invalid_df)}")
    
    # 종료 시간 기록 및 출력
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n처리 완료!")
    print(f"총 소요 시간: {total_time:.2f}초")
    print(f"유효한 쿼리 비율: {len(valid_queries)}/{total_queries} ({len(valid_queries)/total_queries*100:.2f}%)")
    print(f"유효하지 않은 쿼리 비율: {len(invalid_queries)}/{total_queries} ({len(invalid_queries)/total_queries*100:.2f}%)")
    print(f"결과가 valid_queries.csv, invalid_queries.csv 및 Excel 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()