import pandas as pd

ALLOWED_VALUES = {
    "gender": ["남자", "여자"],
    "income_category": ["0 ~ 50%", "51 ~ 75%", "76 ~ 100%", "101 ~ 200%"],
    "personal_category": ["예비부부/난임", "임신부", "출산/입양", "장애인", "국가보훈대상자", "농업인", "어업인", "축산인", "임업인", "초등학생", "중학생", "고등학생", "대학생/대학원생", "질병/질환자", "근로자/직장인", "구직자/실업자", "해당사항 없음"],
    "household_category": ["다문화가정", "북한이탈주민가정", "한부모가정/조손가정", "1인 가구", "다자녀가구", "무주택세대", "신규전입가구", "확대가족", "해당사항 없음"],
    "support_type": ["현금", "현물", "서비스", "이용권"],
}

if __name__ == "__main__":
    input_file2 = "merged_services_categorized_20250327.csv"
    input_file1 = "output/to_rds_v1.json"
    
    # Read the files
    df1 = pd.read_json(input_file1)
    df2 = pd.read_csv(input_file2)
    
    # Limit to first 100 records for testing if needed
    df1_sample = df1  #.head(100).copy()
    
    # 매칭된 레코드 수 추적
    match_count = 0
    update_count = 0
    fill_empty_count = 0
    
    # Iterate through rows in the sample
    for index, data1 in df1_sample.iterrows():
        # Filter df2 where 서비스ID matches service_id
        matching_rows = df2[df2["서비스ID"] == data1["service_id"]]
        
        # Only proceed if we found a match
        if not matching_rows.empty:
            match_count += 1
            data2 = matching_rows.iloc[0]  # Get the first matching row
            
            # Update age-related fields
            if pd.notna(data2["대상연령(시작)"]):
                df1_sample.at[index, "min_age"] = data2["대상연령(시작)"]
                update_count += 1
                
            if pd.notna(data2["대상연령(종료)"]):
                df1_sample.at[index, "max_age"] = data2["대상연령(종료)"]
                update_count += 1
                
            # Update other fields - using exact column names from CSV
            if pd.notna(data2["gender"]):
                df1_sample.at[index, "gender"] = data2["gender"]
                update_count += 1
            elif "gender" in df1_sample.columns and (pd.isna(df1_sample.at[index, "gender"]) or df1_sample.at[index, "gender"] == ""):
                df1_sample.at[index, "gender"] = ", ".join(ALLOWED_VALUES["gender"])
                fill_empty_count += 1
                
            if pd.notna(data2["income_category"]):
                df1_sample.at[index, "income_category"] = data2["income_category"]
                update_count += 1
            elif "income_category" in df1_sample.columns and (pd.isna(df1_sample.at[index, "income_category"]) or df1_sample.at[index, "income_category"] == ""):
                df1_sample.at[index, "income_category"] = ", ".join(ALLOWED_VALUES["income_category"])
                fill_empty_count += 1
                
            if pd.notna(data2["personal_category"]) and data2["personal_category"] != "해당사항 없음":
                df1_sample.at[index, "personal_category"] = data2["personal_category"]
                update_count += 1
            elif "personal_category" in df1_sample.columns and (pd.isna(df1_sample.at[index, "personal_category"]) or df1_sample.at[index, "personal_category"] == "" or df1_sample.at[index, "personal_category"] == "해당사항 없음"):
                df1_sample.at[index, "personal_category"] = ", ".join(ALLOWED_VALUES["personal_category"])
                fill_empty_count += 1
                
            if pd.notna(data2["household_category"]) and data2["household_category"] != "해당사항 없음":
                df1_sample.at[index, "household_category"] = data2["household_category"]
                update_count += 1
            elif "household_category" in df1_sample.columns and (pd.isna(df1_sample.at[index, "household_category"]) or df1_sample.at[index, "household_category"] == "" or df1_sample.at[index, "household_category"] == "해당사항 없음"):
                df1_sample.at[index, "household_category"] = ", ".join(ALLOWED_VALUES["household_category"])
                fill_empty_count += 1
                
            if "support_type" in df1_sample.columns and (pd.isna(df1_sample.at[index, "support_type"]) or df1_sample.at[index, "support_type"] == ""):
                df1_sample.at[index, "support_type"] = ", ".join(ALLOWED_VALUES["support_type"])
                fill_empty_count += 1
    
    # Write the updated sample to a JSON file
    with open("combined_sample.json", "w", encoding="utf-8-sig") as f:
        f.write(df1_sample.to_json(orient="records", force_ascii=False, indent=2))
    
    print(f"Processed {len(df1_sample)} records, found {match_count} matches.")
    print(f"Made {update_count} field updates and {fill_empty_count} empty field fills.")
    print(f"Results saved to combined_sample.json")