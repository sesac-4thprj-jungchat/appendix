import os
import time
import json
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma, Weaviate, Qdrant
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
import annoy
import redis
import openai
from sklearn.metrics import precision_score
import weaviate
from qdrant_client import QdrantClient
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
from concurrent.futures import ThreadPoolExecutor

# 환경 설정
openai.api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# 임베딩 저장/로드 함수
def save_embeddings(embeddings, filename):
    np.save(filename, embeddings)

def load_embeddings(filename):
    return np.load(filename)

# 임베딩 모델 정의
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

class USEEmbeddings(Embeddings):
    def __init__(self, model_url):
        self.model = hub.load(model_url)
    def embed_documents(self, texts):
        return self.model(texts).numpy().tolist()
    def embed_query(self, text):
        return self.model([text]).numpy()[0].tolist()

embedding_models = {
    "OpenAI": OpenAIEmbeddings(model="text-embedding-ada-002"),
    "MiniLM": SentenceTransformerEmbeddings("all-MiniLM-L6-v2"),
    "MPNet": SentenceTransformerEmbeddings("paraphrase-multilingual-mpnet-base-v2"),
    "USE": USEEmbeddings("https://tfhub.dev/google/universal-sentence-encoder/4"),
    "KLUE-BERT": SentenceTransformerEmbeddings("klue/bert-base"),
    "Sroberta": SentenceTransformerEmbeddings("jhgan/ko-sroberta-multitask")
}

# 벡터 DB 함수 정의
def clear_weaviate_classes(embeddings, docs, emb_name):
    client = weaviate.Client("http://localhost:8080")
    class_name = f"Test{emb_name.replace('-', '')}"
    if client.schema.exists(class_name):
        client.schema.delete_class(class_name)
    client.schema.create_class({
        "class": class_name,
        "vectorizer": "none",
        "properties": [
            {"name": "content", "dataType": ["string"]},
            {"name": "metadata", "dataType": ["string"]}
        ]
    })
    with client.batch as batch:
        batch.batch_size = 100
        for doc in docs:
            batch.add_data_object(
                data_object={"content": doc.page_content, "metadata": json.dumps(doc.metadata)},
                class_name=class_name,
                vector=embedding_models[emb_name].embed_query(doc.page_content)
            )
    return Weaviate(client, class_name, "content")

def optimized_qdrant(embeddings, docs, emb_name):
    client = QdrantClient("http://localhost:6333")
    collection_name = f"test_{emb_name}"
    if not client.collection_exists(collection_name):
        return Qdrant.from_documents(docs, embedding_models[emb_name], url="http://localhost:6333", collection_name=collection_name, force_recreate=True)
    return Qdrant(client, collection_name, embedding_models[emb_name])

def build_redis_index(embeddings, docs, emb_name):
    schema = IndexSchema.from_dict({
        "index": {"name": f"{emb_name}_index", "prefix": f"{emb_name}:doc"},
        "fields": [
            {"name": "embedding", "type": "vector", "attrs": {"dims": len(embeddings[0]), "algorithm": "HNSW"}},
            {"name": "content", "type": "text"},
            {"name": "metadata", "type": "text"}
        ]
    })
    index = SearchIndex(schema=schema, redis_client=redis_client)
    index.create(overwrite=True)
    data = [
        {"embedding": emb.tolist(), "content": doc.page_content, "metadata": json.dumps(doc.metadata)}
        for emb, doc in zip(embeddings, docs)
    ]
    index.load(data)
    return index

def build_annoy_index(embeddings, docs):
    dim = len(embeddings[0])
    index = annoy.AnnoyIndex(dim, 'angular')
    for i, emb in enumerate(embeddings):
        index.add_item(i, emb)
    index.build(100)
    return index

def build_chroma_parallel(docs, emb_model, emb_name):
    with ThreadPoolExecutor() as executor:
        embedded_docs = list(executor.map(lambda d: (d, emb_model.embed_query(d.page_content)), docs))
    docs, embeddings = zip(*embedded_docs)
    return Chroma.from_documents(list(docs), emb_model, collection_name=f"test_{emb_name}")

vector_db_options = {
    "FAISS": lambda embeddings, docs, emb_name: FAISS.from_embeddings(
        [(doc.page_content, embeddings[i]) for i, doc in enumerate(docs)], embedding_models[emb_name], metadatas=[doc.metadata for doc in docs]),
    "Chroma": lambda embeddings, docs, emb_name: build_chroma_parallel(docs, embedding_models[emb_name], emb_name),
    "Weaviate": clear_weaviate_classes,
    "Qdrant": optimized_qdrant,
    "Annoy": lambda embeddings, docs, emb_name: build_annoy_index(embeddings, docs),
    "Redis": lambda embeddings, docs, emb_name: build_redis_index(embeddings, docs, emb_name),
}

# Redis 검색 함수
def redis_similarity_search(redis_index, query_vec, emb_name, k=200):
    query_vec_list = query_vec.tolist() if hasattr(query_vec, 'tolist') else query_vec
    query = VectorQuery(
        vector=query_vec_list,
        vector_field_name="embedding",
        return_fields=["content", "metadata"],
        num_results=k
    )
    results = redis_index.search(query.query)
    return [Document(page_content=r["content"], metadata=json.loads(r["metadata"])) for r in results]

# 데이터 로드
json_file_path = r"C:\Users\tkdgh\Desktop\pythonWorkspace\FinalPJ\transformed_output_all_results.json"
with open(json_file_path, 'r', encoding='utf-8') as f:
    documents = [Document(page_content=item['title'], metadata={"application_method": item.get('application_method', "정보 없음"), "service_id": item['service_id']}) 
                 for item in json.load(f)] if os.path.exists(json_file_path) else [
        Document(page_content="신청 방법이 온라인인 서비스 알려줘", metadata={"application_method": "온라인 신청", "service_id": "default1"}),
        Document(page_content="현장 방문으로 신청 가능한 혜택", metadata={"application_method": "현장 방문", "service_id": "default2"})
    ]
total_relevant_docs = sum(1 for doc in documents if doc.metadata.get("application_method") == "온라인 신청")
print(f"로드된 문서 수: {len(documents)}, 온라인 신청 문서 수: {total_relevant_docs}")

# 메트릭 계산 함수 (필터링 추가)
def calculate_metrics(query_result, relevant_condition="온라인 신청", k_values=[3, 10, 50, 100, 200]):
    # 온라인 신청 문서만 필터링
    filtered_result = [doc for doc in query_result if doc.metadata.get("application_method") == relevant_condition]
    y_true = [1 if doc.metadata.get("application_method") == relevant_condition else 0 for doc in query_result]
    metrics = {}
    for k in k_values:
        if len(query_result) < k:
            metrics[f"Precision@{k}"] = 0
            metrics[f"Recall@{k}"] = 0
            continue
        precision = sum(y_true[:k]) / k
        recall = sum(y_true[:k]) / total_relevant_docs if total_relevant_docs > 0 else 0
        metrics[f"Precision@{k}"] = precision
        metrics[f"Recall@{k}"] = recall
    mrr = 0
    for rank, is_relevant in enumerate(y_true, 1):
        if is_relevant:
            mrr = 1 / rank
            break
    metrics["MRR"] = mrr
    return metrics

# 실험 실행
results_summary = []
query_text = "신청 방법이 온라인인 서비스 알려줘"

for emb_name, emb_model in embedding_models.items():
    print(f"\n=== {emb_name} ===")
    embedding_file = f"embeddings_{emb_name}.npy"
    texts = [doc.page_content for doc in documents]
    
    if os.path.exists(embedding_file):
        embeddings = load_embeddings(embedding_file)
        if len(embeddings) == len(documents):
            print(f"{emb_name} 임베딩 로드 중... (문서 수: {len(embeddings)})")
        else:
            print(f"{emb_name} 임베딩 재생성 중...")
            start_time = time.time()
            embeddings = emb_model.embed_documents(texts)
            save_embeddings(embeddings, embedding_file)
            print(f"임베딩 생성 시간: {time.time() - start_time:.2f}초")
    else:
        print(f"{emb_name} 임베딩 생성 중...")
        start_time = time.time()
        embeddings = emb_model.embed_documents(texts)
        save_embeddings(embeddings, embedding_file)
        print(f"임베딩 생성 시간: {time.time() - start_time:.2f}초")
    
    embedded_docs = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in documents]
    
    for db_name, create_db_func in vector_db_options.items():
        try:
            start_time = time.time()
            vectorstore = create_db_func(embeddings, embedded_docs, emb_name)
            creation_time = time.time() - start_time
            
            query_start = time.time()
            query_vec = emb_model.embed_query(query_text)
            if db_name == "FAISS":
                query_result = vectorstore.similarity_search(query_text, k=200)
            elif db_name == "Chroma":
                results = vectorstore.similarity_search_with_score(query_text, k=200)
                unique_results = {doc.page_content: doc for doc, score in results}
                query_result = list(unique_results.values())[:200]
            elif db_name in ["Weaviate", "Qdrant"]:
                query_result = vectorstore.similarity_search(query_text, k=200)
            elif db_name == "Annoy":
                ids, _ = vectorstore.get_nns_by_vector(query_vec, 200, include_distances=True)
                query_result = [documents[i] for i in ids][:200]
            elif db_name == "Redis":
                query_result = redis_similarity_search(vectorstore, query_vec, emb_name, k=200)
            query_time = time.time() - query_start
            
            metrics = calculate_metrics(query_result)
            print(f"[{emb_name} - {db_name}] 생성: {creation_time:.2f}초, 쿼리: {query_time:.2f}초")
            print(f"결과: {[doc.page_content for doc in query_result[:10]]}")
            print(f"Metrics: {metrics}")
            
            results_summary.append({
                "embedding_model": emb_name,
                "vector_db": db_name,
                "creation_time": creation_time,
                "query_time": query_time,
                **metrics
            })
        except Exception as e:
            print(f"[{emb_name} - {db_name}] 오류: {e}")
            results_summary.append({"embedding_model": emb_name, "vector_db": db_name, "error": str(e)})

# 결과 출력
for result in results_summary:
    print(result)