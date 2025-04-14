def classify_departments(department_list, policy_categories):
    # 결과 저장용 딕셔너리
    department_categories = {}
    multi_category_depts = []
    llm_processing_needed = []
    
    # 카테고리별 키워드 우선순위 설정 (숫자가 클수록 더 중요)
    category_priority = {
        "생활안정": 1,
        "주거-자립": 1,
        "보육-교육": 1,
        "고용-창업": 1,
        "보건-의료": 1,
        "행정-안전": 0.8,  # 행정 관련 키워드는 다른 분야와 중복될 가능성이 높음
        "임신-출산": 1.5,  # 더 구체적인 분야이므로 우선순위 높임
        "보호-돌봄": 1.2,
        "문화-환경": 1,
        "농림축산어업": 1.2,
    }
    
    
    for department in department_list:
        # 2. 키워드 기반 점수 계산
        category_scores = {}
        for category, keywords in policy_categories.items():
            score = 0
            for keyword in keywords:
                if keyword in department:
                    # 키워드가 정확하게 부서명의 주요 부분인지 확인 (더 정밀한 매칭)
                    if keyword + "과" in department or keyword + "팀" in department:
                        score += 0.5  # 정확한 매칭에 추가 점수
                    score += 1 * category_priority[category]
            
            if score > 0:
                category_scores[category] = score
        
        # 3. 결과 결정
        if not category_scores:  # 매칭되는 카테고리가 없는 경우
            department_categories[department] = "LLM 처리 필요"
            llm_processing_needed.append(department)
        elif len(category_scores) == 1:  # 단일 카테고리 매칭
            department_categories[department] = list(category_scores.keys())[0]
        else:  # 다중 카테고리 매칭
            # 최고 점수 카테고리 찾기
            max_score = max(category_scores.values())
            top_categories = [cat for cat, score in category_scores.items() if score == max_score]
            
            if len(top_categories) == 1:  # 최고 점수가 단일 카테고리
                department_categories[department] = top_categories[0]
            else:  # 여전히 다중 카테고리가 동점
                # 특수 규칙: 부서명에 "정책"이 포함되면 첫 번째 키워드 우선
                if "정책" in department:
                    for cat, keywords in policy_categories.items():
                        if cat in top_categories:
                            for keyword in keywords:
                                if keyword in department and (keyword + "정책" in department or "정책" + keyword in department):
                                    department_categories[department] = cat
                                    break
                            if department in department_categories:
                                break
                
                # 여전히 결정되지 않았다면
                if department not in department_categories:
                    department_categories[department] = "다중 카테고리: " + ", ".join(top_categories)
                    multi_category_depts.append((department, top_categories))
    
    return department_categories, multi_category_depts, llm_processing_needed

# 카테고리 확장 - 경제 카테고리 추가
policy_categories["경제"] = ["금융", "경제", "투자", "통상", "무역", "산업", "에너지", "지역경제", "벤처"]

# 분류 실행
results, multi_cats, llm_needed = classify_departments(department_list, policy_categories)

# 결과 분석
categories_count = {}
for dept, cat in results.items():
    if "다중 카테고리" not in cat and "LLM 처리 필요" not in cat:
        categories_count[cat] = categories_count.get(cat, 0) + 1

print("카테고리별 부서 수:")
for cat, count in sorted(categories_count.items(), key=lambda x: x[1], reverse=True):
    print(f"{cat}: {count}개 부서")

print(f"\n다중 카테고리 부서: {len(multi_cats)}개")
print(f"LLM 처리 필요 부서: {len(llm_needed)}개")