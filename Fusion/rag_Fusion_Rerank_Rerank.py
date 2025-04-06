import os
import json
import traceback
from dotenv import load_dotenv
from langsmith import Client
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM
from sentence_transformers import CrossEncoder
from langchain.llms.base import LLM
from pydantic import PrivateAttr, Field

# 커스텀 LLM 래퍼 정의
class CustomHuggingFacePipeline(LLM):
    _pipeline: any = PrivateAttr()
    def __init__(self, pipeline, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = pipeline
    def _call(self, prompt: str, stop=None) -> str:
        result = self._pipeline(prompt)
        return result[0]["generated_text"]
    @property
    def _identifying_params(self):
        return {"pipeline": self._pipeline}
    @property
    def _llm_type(self) -> str:
        return "custom_hf_pipeline"

# 배치 처리를 위한 데이터셋 정의
class RerankingDataset(Dataset):
    def __init__(self, query, documents):
        self.query = query
        self.documents = documents
    def __len__(self):
        return len(self.documents)
    def __getitem__(self, idx):
        return self.query, self.documents[idx].page_content

# 환경 변수 로드 및 API 토큰 체크
load_dotenv()
if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
    raise ValueError("HUGGINGFACE_HUB_TOKEN이 .env 파일에 설정되어 있지 않습니다.")
if not os.getenv("COHERE_API_KEY"):
    raise ValueError("COHERE_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
if not os.getenv("LANGSMITH_API_KEY"):
    raise ValueError("LANGSMITH_API_KEY가 .env 파일에 설정되어 있지 않습니다.")

# LangSmith 클라이언트 초기화
client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
prompt_data = client.pull_prompt("langchain-ai/rag-fusion-query-generation", include_model=True)
print("불러온 prompt_data:", prompt_data)

# 헬퍼 함수 정의
def clean_metadata(value):
    return value if value not in [None, ""] else "정보 없음"

# FAQ + 자연어 문장 스타일을 결합한 함수
def generate_selected_text(obj):
    return (
        # FAQ 스타일
        f"Q: 이 서비스의 이름은 무엇인가요?\nA: {clean_metadata(obj.get('서비스명'))}\n\n"
        f"Q: 이 서비스의 ID는 무엇인가요?\nA: {clean_metadata(obj.get('서비스ID'))}\n\n"
        f"Q: 이 서비스는 어떤 기관에서 제공하나요?\nA: {clean_metadata(obj.get('부서명'))}\n\n"
        f"Q: 이 서비스는 어떤 분야에 속하나요?\nA: {clean_metadata(obj.get('서비스분야'))}\n\n"
        f"Q: 이 서비스의 목적은 무엇인가요?\nA: {clean_metadata(obj.get('서비스목적요약'))}\n\n"
        f"Q: 이 서비스를 받을 수 있는 대상은 누구인가요?\nA: {clean_metadata(obj.get('지원대상'))}\n\n"
        f"Q: 지원 내용은 무엇인가요?\nA: {clean_metadata(obj.get('지원내용'))}\n\n"
        f"Q: 이 서비스의 선정 기준은 무엇인가요?\nA: {clean_metadata(obj.get('선정기준'))}\n\n"
        f"Q: 이 서비스의 지원 유형은 무엇인가요?\nA: {clean_metadata(obj.get('지원유형'))}\n\n"
        f"Q: 언제까지 신청할 수 있나요?\nA: {clean_metadata(obj.get('신청기한'))}\n\n"
        f"Q: 신청 방법은 무엇인가요?\nA: {clean_metadata(obj.get('신청방법'))}\n\n"
        f"Q: 어디에서 신청할 수 있나요?\nA: {clean_metadata(obj.get('접수기관'))}\n\n"
        # 자연어 문장 스타일
        f"{clean_metadata(obj.get('서비스명'))} 서비스 (서비스ID: {clean_metadata(obj.get('서비스ID'))})는 "
        f"{clean_metadata(obj.get('부서명'))}에서 운영하는 {clean_metadata(obj.get('서비스분야'))} 분야의 서비스입니다. "
        f"이 서비스는 {clean_metadata(obj.get('서비스목적요약'))}을 목표로 합니다. "
        f"이 서비스를 받을 수 있는 대상은 {clean_metadata(obj.get('지원대상'))}입니다. "
        f"주요 지원 내용은 다음과 같습니다: {clean_metadata(obj.get('지원내용'))}. "
        f"이 서비스의 선정 기준은 다음과 같습니다: {clean_metadata(obj.get('선정기준'))}. "
        f"지원 유형은 {clean_metadata(obj.get('지원유형'))}입니다. "
        f"신청 기한은 {clean_metadata(obj.get('신청기한'))}까지이며, 신청 방법은 {clean_metadata(obj.get('신청방법'))}입니다. "
        f"관련 문의 및 신청은 {clean_metadata(obj.get('접수기관'))}에서 할 수 있습니다."
    )

# 1. 내부 LLM (AutoGPTQForCausalLM) 설정
model_name = "TheBloke/openchat-3.5-1210-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoGPTQForCausalLM.from_quantized(
    model_name, revision="main", use_safetensors=True, device=device, trust_remote_code=True
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
llm_gptq = CustomHuggingFacePipeline(pipeline=pipe)
print("AutoGPTQ 기반 LLM 초기화 완료.")

# 2. RAG Fusion 쿼리 변환 함수
def translate_query(query, prompt_data, llm):
    prompt_template = prompt_data.template if hasattr(prompt_data, 'template') else "{original_query}"
    formatted_prompt = prompt_template.format(original_query=query)
    translated_result = llm(formatted_prompt)
    return translated_result.strip()

# 3. JSON 데이터 로드 및 Document 객체 생성
json_file_path = r"C:\Users\tkdgh\Desktop\pythonWorkspace\FinalPJ\20250304.json"
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
    text = generate_selected_text(obj)
    metadata = {
        "서비스ID": clean_metadata(obj.get("서비스ID")),
        "서비스명": clean_metadata(obj.get("서비스명")),
        "서비스목적요약": clean_metadata(obj.get("서비스목적요약")),
        "신청기한": clean_metadata(obj.get("신청기한")),
        "지원내용": clean_metadata(obj.get("지원내용")),
        "서비스분야": clean_metadata(obj.get("서비스분야")),
        "선정기준": clean_metadata(obj.get("선정기준")),
        "지원유형": clean_metadata(obj.get("지원유형")),
        "신청방법": clean_metadata(obj.get("신청방법")),
        "부서명": clean_metadata(obj.get("부서명")),
        "접수기관": clean_metadata(obj.get("접수기관")),
    }
    documents.append(Document(page_content=text, metadata=metadata))
print(f"총 {len(documents)}개의 문서가 로드됨.")

# 4. FAISS 벡터스토어 생성/로드
embedding_model = HuggingFaceEmbeddings(
    model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)
persist_directory = r"C:\Users\tkdgh\Desktop\pythonWorkspace\FinalPJ"
try:
    vectorstore = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
    print("기존 FAISS 벡터스토어 로드 성공.")
except:
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(persist_directory)
    print(f"총 {len(documents)}개의 문서가 FAISS 벡터 DB에 저장됨.")

# 5. Cross-Encoder 리랭킹 (BaseDocumentCompressor 상속)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L2-v2', device=device)

def cross_encoder_rerank(query, documents, top_n=10, batch_size=32):
    dataset = RerankingDataset(query, documents)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_scores = []
    all_docs = []
    for batch_query, batch_docs in dataloader:
        pairs = [[query, doc] for doc in batch_docs]
        scores = cross_encoder.predict(pairs, batch_size=batch_size)
        all_scores.extend(scores)
        all_docs.extend(documents[len(all_scores)-len(scores):len(all_scores)])
    sorted_pairs = sorted(zip(all_scores, all_docs), reverse=True)
    return [doc for _, doc in sorted_pairs][:top_n]

class CrossEncoderCompressor(BaseDocumentCompressor):
    cross_encoder: CrossEncoder = Field(...)
    top_n: int = Field(...)
    class Config:
        arbitrary_types_allowed = True
    def __init__(self, cross_encoder, top_n):
        super().__init__(cross_encoder=cross_encoder, top_n=top_n)
    def compress_documents(self, documents, query, callbacks=None):
        return cross_encoder_rerank(query, documents, self.top_n)

# 6. 리랭킹 파이프라인 설정
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
cross_encoder_compressor = CrossEncoderCompressor(cross_encoder=cross_encoder, top_n=20)
retriever_cross_compressed = ContextualCompressionRetriever(
    base_compressor=cross_encoder_compressor, base_retriever=base_retriever
)
cohere_compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=15)
retriever_cohere_compressed = ContextualCompressionRetriever(
    base_compressor=cohere_compressor, base_retriever=retriever_cross_compressed
)
final_compressor = CrossEncoderCompressor(cross_encoder=cross_encoder, top_n=10)
retriever_final = ContextualCompressionRetriever(
    base_compressor=final_compressor, base_retriever=retriever_cohere_compressed
)

# 7. RAG Fusion 최종 프롬프트 생성
def rag_fusion_final(query, docs, llm):
    top_docs = docs[:10]
    docs_text = "\n\n".join(
        f"서비스ID: {doc.metadata.get('서비스ID', 'N/A')}\n"
        f"서비스명: {doc.metadata.get('서비스명', 'N/A')}\n"
        f"{((doc.metadata.get('서비스목적요약', '') or '') + ' ' + (doc.metadata.get('지원내용', '') or '')).strip()}"
        for doc in top_docs
    )
    final_prompt = f"Query: {query}\n\nDocuments:\n{docs_text}\n\nAnswer:"
    response = llm(final_prompt)
    return response

# 8. RAG 체인 생성
system_prompt = (
    "검색된 상위 10개 문서의 정보에서, 오직 서비스명과 서비스ID만을 번호별로 출력하세요.\n"
    "출력 형식: [번호]. 서비스명 (서비스ID)"
)
rag_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{context}")])
combine_docs_chain = create_stuff_documents_chain(llm_gptq, rag_prompt)
chain = create_retrieval_chain(retriever_final, combine_docs_chain)
print("Cross-Encoder (20) + CohereRerank (15) + Cross-Encoder (10) 체인 생성 완료.")

# 9. 쿼리 실행
if __name__ == "__main__":
    queries = [
        "서울특별시 도봉구에 사는 34세 청년 남자에게 맞는 서비스를 찾아줘.",
        "전라남도 목포시에 사는 48세 중년 남자에게 맞는 귀어 관련 창업에 대해 알려줘.",
        "강원도 동해시에 사는 68세 노인 여자에게 맞는 정책을 찾아줘.",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n=== 쿼리 {i}: {query} ===")
        try:
            translated_query = translate_query(query, prompt_data, llm_gptq)
            print(f"Translated Query: {translated_query}")

            retrieved_docs = base_retriever.invoke(translated_query)
            print(f"기본 검색된 문서 수: {len(retrieved_docs)}")
            cross_docs = retriever_cross_compressed.invoke(translated_query)
            print(f"1차 Cross-Encoder 후 문서 수: {len(cross_docs)}")
            cohere_docs = retriever_cohere_compressed.invoke(translated_query)
            print(f"Cohere Rerank 후 문서 수: {len(cohere_docs)}")
            final_docs = retriever_final.invoke(translated_query)
            print("2차 Cross-Encoder 후 상위 10개 문서:")
            for j, doc in enumerate(final_docs, 1):
                print(f"문서 {j}: {doc.metadata['서비스명']} (서비스ID: {doc.metadata['서비스ID']})")

            fusion_response = rag_fusion_final(translated_query, final_docs, llm_gptq)
            print("RAG Fusion 응답:")
            print(fusion_response)

            response_chain = chain.invoke({"input": translated_query})
            print("기존 체인 응답:")
            print(response_chain["answer"])

        except Exception as e:
            print(f"쿼리 실행 중 오류 발생: {e}")
            traceback.print_exc()