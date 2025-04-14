import json
import openai
from dotenv import load_dotenv
import os

# 🔐 API 키 로딩
load_dotenv()
openai.api_key = os.getenv("OPEN_API_KEY")

# 📄 정책 데이터 불러오기
with open("/Users/minjoo/Desktop/SeSac/final/fundit_promport/data/20250304.json", "r", encoding="utf-8") as f:
    policies = json.load(f)

# 🧠 LLM으로 정책 요약
def summarize_policy(policy):
    with open("/Users/minjoo/Desktop/SeSac/final/fundit_promport/prompts/policy_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    prompt = prompt_template.format(
        서비스목적요약=policy.get("서비스목적요약", ""),
        지원대상=policy.get("지원대상", ""),
        지원내용=policy.get("지원내용", "")
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "너는 정부 보조금 요약 전문가야. 주어진 정보를 2~3줄로 간결하게 요약해줘. 정책명이나 ID는 포함하지 마."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# ✅ 요약된 정책 리스트 (서비스ID + 요약 텍스트)
summaries = [
    {
        "서비스ID": p["서비스ID"],
        "summary": summarize_policy(p)
    }
    for p in policies[:5]  # <- 테스트용. 전체 쓰고 싶으면 [:]로 바꿔줘
]

# 💾 정책 요약 결과 저장
with open("/Users/minjoo/Desktop/SeSac/final/fundit_promport/data/summaries.json", "w", encoding="utf-8") as f:
    json.dump(summaries, f, ensure_ascii=False, indent=2)

# 🧠 사용자 맞춤 전체 요약 생성
def personalized_summary(user_info, summaries):
    with open("/Users/minjoo/Desktop/SeSac/final/fundit_promport/prompts/top_recommendation_prompt.txt", "r", encoding="utf-8") as f:
        template = f.read()

    # 요약 리스트만 뽑아 붙이기
    summary_texts = [s["summary"] for s in summaries]
    prompt = template.format(
        지역=user_info["지역"],
        연령대=user_info["연령대"],
        가구형태=user_info["가구형태"],
        요약된_정책_리스트="\n".join(summary_texts)
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "너는 정부 정책 추천 전문가야. 주어진 정보를 갖고 사용자의 상황을 설명하고, 왜 이 정책들이 적합한지, 최대 얼마까지 지원 가능한지 알려줘. 마지막에 '자세한 내용은 각 정책의 링크를 참고해 주세요.'로 끝내."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# 👤 사용자 정보 예시
user_info = {
    "지역": "서울",
    "연령대": "30대",
    "가구형태": "맞벌이 가구"
}

# 🔄 전체 요약 생성
user_summary = {
    "user_info": user_info,
    "summary": personalized_summary(user_info, summaries)
}

# 💾 사용자 맞춤 요약 저장
with open("/Users/minjoo/Desktop/SeSac/final/fundit_promport/data/user_summary.json", "w", encoding="utf-8") as f:
    json.dump(user_summary, f, ensure_ascii=False, indent=2)

# ✅ 출력 확인
print("✅ 요약된 정책:", len(summaries), "건 저장됨")
print("✅ 맞춤형 전체 요약 저장 완료: data/user_summary.json")