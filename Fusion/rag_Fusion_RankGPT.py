# test
import os
import json
import re
import torch
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
from langsmith import Client
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

# .env 파일 로드
load_dotenv('api.env')
if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
    raise ValueError("HUGGINGFACE_HUB_TOKEN이 .env 파일에 설정되어 있지 않습니다.")
if not os.getenv("LANGSMITH_API_KEY"):
    raise ValueError("LANGSMITH_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

# LangSmith Client 초기화
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
client = Client(api_key=LANGSMITH_API_KEY)
rag_fusion_prompt = client.pull_prompt("langchain-ai/rag-fusion-query-generation", include_model=True)

# 모델 및 토크나이저 로드
MODEL_NAME = "beomi/KoAlpaca-Polyglot-5.8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 중인 장치: {device}")

# KoAlpaca 모델 로드 (GPU에 할당)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
tokenizer.pad_token = tokenizer.eos_token

# None 또는 빈 문자열 처리 함수
def clean_metadata(value):
    return value if value not in [None, ""] else "정보 없음"

# JSON 데이터 로드 및 Document 객체 생성
json_file_path = r"C:\Users\tkdgh\Desktop\pythonWorkspace\FinalPJ\20250304.json"  # 경로 수정
try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"JSON 파일 '{json_file_path}'을 찾을 수 없습니다.")
except json.JSONDecodeError:
    raise ValueError(f"JSON 파일 '{json_file_path}'의 형식이 잘못되었습니다.")

if not isinstance(json_data, list):
    raise ValueError("JSON 데이터가 리스트 형태가 아닙니다.")

documents = []
for obj in json_data:
    text = (
        f"서비스명: {clean_metadata(obj.get('서비스명'))}\n"
        f"서비스ID: {clean_metadata(obj.get('서비스ID'))}\n"
        f"부서명: {clean_metadata(obj.get('부서명'))}\n"
        f"서비스분야: {clean_metadata(obj.get('서비스분야'))}\n"
        f"서비스목적요약: {clean_metadata(obj.get('서비스목적요약'))}\n"
        f"지원대상: {clean_metadata(obj.get('지원대상'))}\n"
        f"지원내용: {clean_metadata(obj.get('지원내용'))}\n"
        f"선정기준: {clean_metadata(obj.get('선정기준'))}\n"
        f"지원유형: {clean_metadata(obj.get('지원유형'))}\n"
        f"신청기한: {clean_metadata(obj.get('신청기한'))}\n"
        f"신청방법: {clean_metadata(obj.get('신청방법'))}\n"
        f"접수기관: {clean_metadata(obj.get('접수기관'))}\n"
    )
    metadata = {
        "서비스ID": clean_metadata(obj.get("서비스ID")),
        "서비스명": clean_metadata(obj.get("서비스명")),
        "서비스목적요약": clean_metadata(obj.get("서비스목적요약")),
        "신청기한": clean_metadata(obj.get("신청기한")),
        "지원내용": clean_metadata(obj.get("지원내용")),
        "서비스분야": clean_metadata(obj.get("서비스분야")),
        "선정기준": clean_metadata(obj.get("선정기준")),
        "신청방법": clean_metadata(obj.get("신청방법")),
        "부서명": clean_metadata(obj.get("부서명")),
        "접수기관": clean_metadata(obj.get("접수기관"))
    }
    documents.append(Document(page_content=text, metadata=metadata))

# 임베딩 모델 설정 (GPU 사용)
embedding_model = HuggingFaceEmbeddings(
    model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

# FAISS 벡터스토어 로드
persist_directory = r"C:\Users\tkdgh\Desktop\pythonWorkspace\FinalPJ"  # 경로 수정
faiss_index_file = os.path.join(persist_directory, "index.faiss")
if not os.path.exists(faiss_index_file):
    raise FileNotFoundError(f"FAISS 인덱스 파일이 존재하지 않습니다: {faiss_index_file}")
vectorstore = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)

# 일반 RAG 검색
def retrieve_documents(query, top_k=20):
    return vectorstore.similarity_search(query, k=top_k)

# RAG Fusion 쿼리 번역
def translate_query_with_rag_fusion(query):
    input_text = f"{rag_fusion_prompt}\n사용자 쿼리: {query}\n번역된 쿼리:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    print("🚀 Generating translation...")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=10,
        do_sample=False,
        no_repeat_ngram_size=3
    )
    translated_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("✅ Translation complete!")
    return translated_query

# KoGPT-6B-4bit으로 문서 랭킹 (4bit 양자화 지원)
def rank_documents_with_kogpt(query, documents):
    kogpt_model_name = "rycont/kakaobrain__kogpt-6b-4bit"
    kogpt_tokenizer = AutoTokenizer.from_pretrained(kogpt_model_name)
    kogpt_model = AutoModelForCausalLM.from_pretrained(
        kogpt_model_name,
        load_in_4bit=True,  # 4bit 양자화 로드
        device_map="auto"   # GPU 자동 할당
    )

    prompt = "다음 사용자 질문에 대해 가장 관련 있는 문서를 순서대로 나열하고 각각의 관련성 점수를 1점에서 100점 사이로 매겨주세요.\n\n"
    for i, doc in enumerate(documents):
        prompt += f"{i+1}. {doc.page_content}\n"
    prompt += "\n각 문서의 순위와 점수를 다음과 같은 형식으로 출력해주세요: (순위). (문서 내용 일부) - (점수)점"

    inputs = kogpt_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = kogpt_model.generate(inputs["input_ids"], max_new_tokens=200)
    ranked_text = kogpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def extract_rank_score_kogpt(ranked_text, document):
        doc_snippet = document.page_content[:30]
        pattern = re.compile(rf"{re.escape(doc_snippet)}.*?(\d+)점", re.DOTALL)
        match = pattern.search(ranked_text)
        return int(match.group(1)) if match else 0

    ranked_docs = sorted(documents, key=lambda x: extract_rank_score_kogpt(ranked_text, x), reverse=True)
    return ranked_docs[:15]

# 파이프라인 실행
query = "서울특별시 도봉구에 사는 34세 청년 남자에게 맞는 서비스를 찾아줘."

print("🔍 1단계: RAG Fusion을 이용한 쿼리 번역")
translated_query = translate_query_with_rag_fusion(query)
print("번역된 쿼리:", translated_query)

print("🔍 2단계: 일반 RAG 검색")
retrieved_docs = retrieve_documents(translated_query)

print("📊 3단계: GPT Rank를 이용한 문서 랭킹 (KoGPT-6B-4bit)")
ranked_docs = rank_documents_with_kogpt(translated_query, retrieved_docs)

print("✅ 최종 랭크 결과:")
for i, doc in enumerate(ranked_docs):
    print(f"{i+1}. {doc.metadata.get('서비스명', '서비스명 없음')}")

# 리소스 정리
import gc

if __name__ == "__main__":
    print("🔄 리소스 정리 중...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # CUDA 메모리 해제
    for p in multiprocessing.active_children():
        p.terminate()
    gc.collect()
    print("✅ 리소스 정리 완료!")