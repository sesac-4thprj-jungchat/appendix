import pymysql
import json

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "database": "final_project",
    "charset": "utf8mb4"
}

def connect_db():
    return pymysql.connect(**DB_CONFIG)

def insert_benefit_data(data):
    conn = connect_db()
    cursor = conn.cursor()

    sql = """
    INSERT INTO benefits (area, district, min_age, max_age, age_summary, gender, 
                          income_category, income_summary, personal_category, personal_summary, 
                          household_category, household_summary, support_type, support_summary, 
                          application_method, application_summary, benefit_category, benefit_summary, 
                          start_date, end_date, date_summary, benefit_details, source, additional_data, 
                          keywords, service_id)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    try:
        cursor.execute(sql, (
            data.get("area"),
            data.get("district"),
            data.get("min_age"),
            data.get("max_age"),
            data.get("age_summary"),
            data.get("gender"),
            data.get("income_category"),
            data.get("income_summary"),
            data.get("personal_category"),
            data.get("personal_summary"),
            data.get("household_category"),
            data.get("household_summary"),
            data.get("support_type"),
            data.get("support_summary"),
            data.get("application_method"),
            data.get("application_summary"),
            data.get("benefit_category"),
            data.get("benefit_summary"),
            data.get("start_date") or None,  # Handle empty strings for dates
            data.get("end_date") or None,
            data.get("date_summary"),
            data.get("benefit_details"),
            data.get("source"),
            data.get("additional_data"),
            data.get("keywords"),
            data.get("service_id")
        ))
        conn.commit()
        print(f"✅ 데이터 삽입 완료! (Service ID: {data.get('service_id')})")
    except pymysql.MySQLError as e:
        print(f"❌ 데이터 삽입 실패: {e}")
    finally:
        cursor.close()
        conn.close()

def process_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data_list = json.load(file)

    for data in data_list:
        insert_benefit_data(data)

# JSON 파일 경로를 지정하세요
process_json_file("output_all_results_gemini.json")