import openai
import json
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPEN_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key

# 📄 파일 경로
INPUT_FILE = "data/to_rds_v2.json"
OUTPUT_FILE = "data/filled_policies.json"
PROMPT_FILE = "prompts/base_prompt.txt"

# 📋 프롬프트 템플릿 로딩
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt_template = f.read()

# 📚 입력 데이터 로딩
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    policies = json.load(f)

# ✏️ 프롬프트 구성
def build_prompt(policy):
    return prompt_template.replace("${benefit_summary}", policy.get("benefit_summary", ""))\
                          .replace("${benefit_details}", policy.get("benefit_details", ""))\
                          .replace("${source}", policy.get("source", ""))\
                          .replace("${keywords}", policy.get("keywords", ""))

# 💬 GPT-4 호출 함수
def get_area_from_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "너는 정책 내용을 읽고 지역 정보를 추론하는 분석가야."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    content = response["choices"][0]["message"]["content"]
    try:
        area_data = json.loads(content)
        return area_data.get("area", ""), area_data.get("district", "")
    except:
        return "", ""

# 🏃 메인 처리 루프
for policy in tqdm(policies):
    if policy["area"] == "" and policy["district"] == "":
        prompt = build_prompt(policy)
        area, district = get_area_from_gpt(prompt)
        policy["area_filled"] = area
        policy["district"] = district

# 💾 저장
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(policies, f, ensure_ascii=False, indent=2)
