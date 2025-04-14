import os
import json
import torch
import pandas as pd
import logging
import random
import time
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

# HF libraries
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, DatasetDict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_ID = "mistralai/Mistral-7B-v0.3"
OUTPUT_DIR = "./mistral-7b-sql-finetuned"
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
VECTORSTORE_PATH = "./question_vectorstore"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./plots", exist_ok=True)
os.makedirs(VECTORSTORE_PATH, exist_ok=True)

# 프롬프트 변형 정의
variation1 = '''
### SQL 변환 작업

데이터베이스 스키마 정보:
{schema}

질문을 분석하여 아래 규칙에 따라 SQL 쿼리로 변환해주세요:
1. 스키마에 존재하는 테이블과 컬럼만 사용
2. 질문에 직접 관련된 필드만 SELECT
3. 와일드카드 검색 시 '%keyword%' 형식 활용
4. 날짜는 'YYYY-MM-DD' 형식 유지
5. 허용된 카테고리 값만 사용
6. 주석 없이 쿼리 작성
7. 필요한 경우에만 JOIN 사용
8. 단순 비교 연산자 우선 활용
9. 스키마에 정의된 값만 참조

### 예시:
질문: {example_question1}
SQL 쿼리: 
{code_start}
{example_sql1}
{code_end}

질문: {example_question2}
SQL 쿼리:
{code_start}
{example_sql2}
{code_end}

변환할 질문: {question}

SQL 쿼리 결과:
{code_start}
'''

variation2 = '''
## SQL Query Generator

**Database Schema:**
{schema}

**Instructions:**
Convert the input question into a valid SQL query following these technical guidelines:
- Use only tables and columns defined in the schema
- Select only fields relevant to the question
- Properly implement wildcard searches with '%'
- Format dates as 'YYYY-MM-DD'
- Use only allowed category values from schema
- Exclude comments from final query
- Join tables only when necessary
- Prefer simple comparison operators when possible
- Do not generate arbitrary text values

**Examples:**
Question: {example_question1}
SQL Query: 
{code_start}
{example_sql1}
{code_end}

Question: {example_question2}
SQL Query:
{code_start}
{example_sql2}
{code_end}

**Input Question:**
{question}

**Generated SQL:**
{code_start}
'''

variation3 = '''
### 요청: 다음 질문을 SQL 쿼리로 변환하세요.

### 데이터베이스 스키마:
{schema}

### SQL 작성 지침:
- 스키마에 정의된 테이블과 컬럼만 사용하세요
- 질문에서 요구하는 정보만 포함하는 컬럼을 선택하세요
- LIKE 검색 시 '%' 와일드카드를 적절히 사용하세요
- 날짜는 'YYYY-MM-DD' 형식으로 처리하세요
- 카테고리 필드는 스키마에 정의된 값만 사용하세요
- 주석은 포함하지 마세요
- 불필요한 JOIN은 피하세요
- 복잡한 CASE/IF 문은 가능한 사용하지 마세요
- 임의의 텍스트를 생성하지 마세요

### 예시:
질문: {example_question1}
SQL:
{code_start}
{example_sql1}
{code_end}

질문: {example_question2}
SQL:
{code_start}
{example_sql2}
{code_end}

### 질문: {question}

### SQL:
{code_start}
'''

variation4 = '''
# SQL 쿼리 변환 프로세스

## 스키마 정의
{schema}

## 변환 단계:
1단계: 질문에서 필요한 정보 식별
2단계: 스키마에서 관련 테이블과 컬럼 확인
3단계: 적절한 WHERE 조건 구성
4단계: 필요한 SELECT 필드만 포함
5단계: 스키마 제약사항 준수 확인

## 중요 규칙:
- 스키마 외 요소 사용 금지
- 필요한 정보만 선택
- 적절한 LIKE 검색 구현
- 날짜 형식 준수
- 허용된 카테고리 값만 사용
- 주석 미포함
- 필요시에만 JOIN
- 단순 연산자 우선
- 임의 텍스트 생성 금지

## 예시:
질문: {example_question1}
SQL:
{code_start}
{example_sql1}
{code_end}

질문: {example_question2}
SQL:
{code_start}
{example_sql2}
{code_end}

## 질문:
{question}

## SQL:
{code_start}
'''

variation5 = '''
### 스키마
{schema}

### 규칙
- 스키마 준수
- 필요 필드만 선택
- 와일드카드 적절히 사용
- 날짜 형식: YYYY-MM-DD
- 허용된 카테고리 값만 사용
- 주석 없음
- 필요시에만 JOIN
- 단순 연산자 우선
- 임의 값 생성 금지

### 예시
질문: {example_question1}
SQL:
{code_start}
{example_sql1}
{code_end}

질문: {example_question2}
SQL:
{code_start}
{example_sql2}
{code_end}

### 질문
{question}

### SQL:
{code_start}
'''

# 프롬프트 변형 목록
PROMPT_VARIATIONS = [variation1, variation2, variation3, variation4, variation5]

def get_db_schema(db_id):
    """데이터베이스 스키마 정보 반환"""
    DEFAULT_SCHEMA = """
Table: benefits
Columns: area, district, min_age, max_age, income_category, personal_category, household_category, support_type, application_method, benefit_category, start_date, end_date"""
    
    return DEFAULT_SCHEMA

def extract_sql_from_output(output_text, code_start="SQL_BEGIN", code_end="SQL_END"):
    """생성된 출력에서 SQL 쿼리 추출"""
    # 코드 블록 내 SQL 찾기
    start_idx = output_text.find(code_start)
    
    if start_idx != -1:
        start_idx += len(code_start)
        end_idx = output_text.find(code_end, start_idx)
        
        if end_idx != -1:
            return output_text[start_idx:end_idx].strip()
        else:
            return output_text[start_idx:].strip()
    
    # SQL 키워드로 찾기
    select_pattern = r"SELECT\s+.*?FROM\s+.*?"
    matches = re.search(select_pattern, output_text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        sql_text = matches.group(0)
        # 전체 SQL 쿼리 추출 시도
        rest_of_text = output_text[matches.start():]
        
        # 문장 끝 찾기
        for end_marker in ["\n\n", "```", "**", "###"]:
            if end_marker in rest_of_text:
                sql_end = rest_of_text.find(end_marker)
                if sql_end > 10:  # 최소한의 SQL 내용 확인
                    return rest_of_text[:sql_end].strip()
        
        return rest_of_text.strip()
    
    # SQL: 마커 찾기
    sql_marker = output_text.find("SQL:")
    if sql_marker != -1:
        sql_text = output_text[sql_marker + 4:].strip()
        # 닫는 마커가 있으면 찾기
        for end_marker in ["```", "**", "###", "\n\n"]:
            if end_marker in sql_text:
                end_pos = sql_text.find(end_marker)
                if end_pos > 0:
                    return sql_text[:end_pos].strip()
        return sql_text
    
    # SELECT 포함하지만 다른 방법으로 찾지 못한 경우
    if "SELECT" in output_text.upper():
        start = output_text.upper().find("SELECT")
        return output_text[start:].strip()
    
    # 아무것도 찾지 못하면 전체 텍스트 반환
    return output_text.strip()

def load_embedding_model(device=None):
    """임베딩 모델 로드"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embedding_model

def load_or_create_vectorstore(embedding_model, data_df, store_path):
    """기존 벡터스토어 로드 또는 새로 생성"""
    if os.path.exists(store_path) and os.path.isdir(store_path):
        logger.info(f"기존 벡터스토어 로드: {store_path}")
        try:
            vectorstore = FAISS.load_local(store_path, embedding_model, allow_dangerous_deserialization=True)
            logger.info("벡터스토어 로드 성공!")
            return vectorstore
        except Exception as e:
            logger.warning(f"벡터스토어 로드 오류: {e}. 새로 생성합니다.")
    
    logger.info("새 벡터스토어 생성")
    texts = data_df['query'].tolist()
    metadatas = [{"sql_query": sql, "index": i} for i, sql in enumerate(data_df['generated_sql'].tolist())]
    
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )
    
    logger.info(f"벡터스토어 저장: {store_path}")
    vectorstore.save_local(store_path)
    
    return vectorstore

def batch_similarity_search(questions, vectorstore, batch_size=16, k=4):
    """배치 방식으로 유사한 질문 검색"""
    results = []
    for i in tqdm(range(0, len(questions), batch_size), desc="유사 질문 검색 중"):
        batch = questions[i:i+batch_size]
        batch_results = []
        
        for question in batch:
            similar = vectorstore.similarity_search(question, k=k)
            batch_results.append(similar)
            
        results.extend(batch_results)
    return results

def prepare_model_and_tokenizer():
    """Load model and tokenizer for single GPU"""
    logger.info(f"Loading model: {MODEL_ID}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure for single GPU
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load model for single GPU
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",  # Will use single GPU
            torch_dtype=torch.bfloat16
        )
        
        # Enable gradient checkpointing
        base_model.gradient_checkpointing_enable()
        
        # Prepare for QLoRA
        model = prepare_model_for_kbit_training(base_model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        
        # Apply LoRA adapters
        model = get_peft_model(model, lora_config)
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params} / {total_params} "
                  f"({100 * trainable_params / total_params:.2f}%)")
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise

def tokenize_dataset(dataset, tokenizer):
    """데이터셋 토큰화"""
    logger.info("데이터셋 토큰화 중...")
    
    def tokenize_function(examples):
        # 전체 text 토큰화
        tokenized_text = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )
        input_ids = tokenized_text["input_ids"]
        attention_mask = tokenized_text["attention_mask"]
        
        # input_text 토큰화하여 길이 계산
        tokenized_input = tokenizer(
            examples["input_text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False
        )
        # 각 input_text의 토큰 길이
        len_inputs = [len(ids) for ids in tokenized_input["input_ids"]]
        
        # 레이블 설정 (입력 부분은 -100으로 마스킹)
        labels = []
        for i, (ids, len_input) in enumerate(zip(input_ids, len_inputs)):
            label = ids.copy()
            if len_input < MAX_SEQ_LENGTH:
                label[:len_input] = [-100] * len_input
            labels.append(label)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "input_text", "output_text"]
    )
    return tokenized_dataset

def generate_sql_with_model(model, tokenizer, question, schema, examples=None):
    """모델을 사용하여 SQL 쿼리 생성"""
    # 예시 데이터 준비
    if examples is None or len(examples) < 2:
        example1 = {
            "query": "30대 미혼 여성을 위한 주거 지원 서비스는?",
            "generated_sql": "SELECT * FROM benefits WHERE min_age <= 30 AND max_age >= 39 AND personal_category LIKE '%미혼%' AND personal_category LIKE '%여성%' AND benefit_category = '주거'"
        }
        example2 = {
            "query": "서울시 노인 의료 지원 서비스 알려줘",
            "generated_sql": "SELECT * FROM benefits WHERE area LIKE '%서울%' AND min_age >= 65 AND benefit_category = '의료'"
        }
    else:
        example1 = examples[0]
        example2 = examples[1]
    
    # 코드 블록 구분자
    code_start = "SQL_BEGIN"
    code_end = "SQL_END"
    
    # 랜덤 프롬프트 변형 선택
    prompt_template = random.choice(PROMPT_VARIATIONS)
    
    # 프롬프트 포맷팅
    formatted_prompt = prompt_template.format(
        schema=schema,
        example_question1=example1["query"],
        example_sql1=example1["generated_sql"],
        example_question2=example2["query"],
        example_sql2=example2["generated_sql"],
        question=question,
        code_start=code_start,
        code_end=code_end
    )
    
    # 토큰화
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 생성 설정
    generation_config = {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    # SQL 생성
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)
    
    # 입력 길이 계산
    input_length = inputs["input_ids"].shape[1]
    # 새로 생성된 토큰만 가져오기
    new_tokens = outputs[0][input_length:]
    
    # 생성된 텍스트 디코딩
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # 출력에서 SQL 추출
    sql_query = extract_sql_from_output(generated_text, code_start, code_end)
    
    # 메모리 정리
    del inputs, outputs
    torch.cuda.empty_cache()
    
    return sql_query

class TrainingMonitorCallback(TrainerCallback):
    """학습 진행 상황을 모니터링하는 콜백 클래스"""
    def __init__(self, output_dir, tokenizer, val_examples):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.val_examples = val_examples
        self.best_loss = float('inf')
        self.best_epoch = -1
        
        # 학습 상태 기록
        self.train_losses = []
        self.eval_losses = []
        self.epochs = []
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """평가 후 호출되는 메소드"""
        if metrics is None:
            return control
            
        epoch = int(state.epoch)
        self.epochs.append(epoch)
        
        # 최신 로그에서 훈련 손실 가져오기
        train_loss = 0
        if state.log_history:
            train_logs = [log for log in state.log_history if 'loss' in log]
            if train_logs:
                train_loss = train_logs[-1].get('loss', 0)
        
        # 검증 손실
        eval_loss = metrics.get('eval_loss', 0)
        
        # 손실 기록
        self.train_losses.append(train_loss)
        self.eval_losses.append(eval_loss)
        
        logger.info(f"에폭 {epoch}: 훈련 손실: {train_loss:.4f}, 검증 손실: {eval_loss:.4f}")
        
        # SQL 생성 예시 평가
        if self.val_examples is not None and len(self.val_examples) > 0:
            model = kwargs.get('model')
            if model is not None:
                self._evaluate_samples(model, epoch)
        
        # 그래프 그리기
        self._plot_loss_curves()
        
        # 최고 모델 저장
        if eval_loss < self.best_loss:
            logger.info(f"에폭 {epoch}에서 최고 모델 발견: {eval_loss:.4f} (이전: {self.best_loss:.4f})")
            self.best_loss = eval_loss
            self.best_epoch = epoch
            
            # 최고 모델 저장
            best_model_dir = os.path.join(self.output_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            
            model = kwargs.get('model')
            if model is not None:
                model.save_pretrained(best_model_dir)
                self.tokenizer.save_pretrained(best_model_dir)
                
                with open(os.path.join(best_model_dir, "best_model_info.json"), "w") as f:
                    json.dump({
                        "epoch": epoch,
                        "eval_loss": float(eval_loss),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, f)
        
        return control
    
    def _evaluate_samples(self, model, epoch):
        """검증 샘플로 SQL 생성 능력 평가"""
        logger.info(f"에폭 {epoch}에서 샘플 SQL 생성 평가 중...")
        
        # 최대 5개 샘플 선택
        sample_size = min(5, len(self.val_examples))
        samples = self.val_examples.sample(sample_size)
        
        # DB 스키마 가져오기
        db_schema = get_db_schema("default")
        
        results = []
        for _, row in samples.iterrows():
            question = row["query"]
            reference_sql = row["generated_sql"]
            
            # SQL 생성
            generated_sql = generate_sql_with_model(model, self.tokenizer, question, db_schema)
            
            # 결과 기록
            results.append({
                "question": question,
                "reference_sql": reference_sql,
                "generated_sql": generated_sql
            })
        
        # 결과 저장
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        with open(os.path.join(epoch_dir, "sample_sql_generation.json"), "w") as f:
            json.dump(results, f, indent=2)
    
    def _plot_loss_curves(self):
        """손실 그래프 생성"""
        if len(self.epochs) < 1:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(self.epochs, self.eval_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # 그래프 저장
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(f"./plots/loss_curve_{timestamp}.png")
        plt.savefig(f"./plots/loss_curve_latest.png")  # 최신 그래프
        plt.close()
        
        # 손실값 CSV 저장
        df = pd.DataFrame({
            'epoch': self.epochs,
            'train_loss': self.train_losses,
            'validation_loss': self.eval_losses
        })
        df.to_csv('./plots/metrics_history.csv', index=False)

def process_data(data, embedding_model, vectorstore):
    """훈련 데이터 형식으로 처리"""
    examples = []
    
    # 질문 목록 생성
    questions = [item["query"] for item in data]
    
    # 배치 처리로 유사 예제 찾기
    batch_size = min(16, len(questions))
    similar_examples_batch = batch_similarity_search(questions, vectorstore, batch_size=batch_size, k=4)
    
    for i, item in enumerate(tqdm(data, desc="데이터 처리 중")):
        question = item["query"]
        sql = item["generated_sql"]
        current_index = item.get("index", -1)
        
        # 유사 예제 가져오기
        similar_examples = similar_examples_batch[i]
        filtered_examples = [ex for ex in similar_examples if ex.metadata.get("index", -999) != current_index]
        
        # 최소 2개 이상의 예시 확보
        if len(filtered_examples) < 2:
            filtered_examples = [{
                "page_content": "30대 미혼 여성을 위한 주거 지원 서비스는?",
                "metadata": {"sql_query": "SELECT * FROM benefits WHERE min_age <= 30 AND max_age >= 39 AND personal_category LIKE '%미혼%' AND personal_category LIKE '%여성%' AND benefit_category = '주거'"}
            }, {
                "page_content": "서울시 노인 의료 지원 서비스 알려줘",
                "metadata": {"sql_query": "SELECT * FROM benefits WHERE area LIKE '%서울%' AND min_age >= 65 AND benefit_category = '의료'"}
            }]
        
        # 예시 데이터 준비
        example1 = {
            "query": filtered_examples[0].page_content,
            "generated_sql": filtered_examples[0].metadata.get("sql_query", "")
        }
        example2 = {
            "query": filtered_examples[1].page_content,
            "generated_sql": filtered_examples[1].metadata.get("sql_query", "")
        }
        
        # DB 스키마 가져오기
        db_schema = get_db_schema("default")
        
        # 랜덤 프롬프트 변형 선택
        prompt_template = random.choice(PROMPT_VARIATIONS)
        
        # 코드 시작/종료 구분자 설정
        code_start = "SQL_BEGIN"
        code_end = "SQL_END"
        
        # 템플릿 포맷팅
        formatted_prompt = prompt_template.format(
            schema=db_schema,
            example_question1=example1["query"],
            example_sql1=example1["generated_sql"],
            example_question2=example2["query"],
            example_sql2=example2["generated_sql"],
            question=question,
            code_start=code_start,
            code_end=code_end
        )
        
        # 입력 프롬프트와 정답 SQL 분리
        input_prompt = formatted_prompt
        
        examples.append({
            "text": formatted_prompt + sql + f"\n{code_end}",  # 전체: 프롬프트 + 정답 SQL + 종료 태그
            "input_text": input_prompt,
            "output_text": sql
        })
    
    return examples

def load_dataset():
    """단일 GPU용 데이터셋 로드 및 처리"""
    logger.info("데이터셋 로드 중...")
    
    # 엑셀 파일 로드
    df = pd.read_excel("1.1_refined.xlsx")
    
    # 데이터 분할
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    
    # 인덱스 추가
    train_df['index'] = range(len(train_df))
    val_df['index'] = range(len(train_df), len(train_df) + len(val_df))
    test_df['index'] = range(len(train_df) + len(val_df), len(train_df) + len(val_df) + len(test_df))
    
    # 테스트 데이터 저장
    test_df.to_csv(f"{OUTPUT_DIR}/test_data.csv", index=False)
    
    # 임베딩 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = load_embedding_model(device)
    
    # 벡터스토어 경로 설정
    train_vs_path = os.path.join(VECTORSTORE_PATH, "train_vectorstore")
    val_vs_path = os.path.join(VECTORSTORE_PATH, "val_vectorstore")
    test_vs_path = os.path.join(VECTORSTORE_PATH, "test_vectorstore")
    
    # 벡터스토어 생성 또는 로드
    train_vectorstore = load_or_create_vectorstore(
        embedding_model, 
        pd.concat([val_df, test_df]),  # 학습용 예시로 검증/테스트 데이터 사용
        train_vs_path
    )
    
    val_vectorstore = load_or_create_vectorstore(
        embedding_model, 
        pd.concat([train_df, test_df]),  # 검증용 예시로 학습/테스트 데이터 사용
        val_vs_path
    )
    
    test_vectorstore = load_or_create_vectorstore(
        embedding_model, 
        pd.concat([train_df, val_df]),  # 테스트용 예시로 학습/검증 데이터 사용
        test_vs_path
    )
    
    # 데이터 처리
    train_data = process_data(train_df.to_dict('records'), embedding_model, train_vectorstore)
    val_data = process_data(val_df.to_dict('records'), embedding_model, val_vectorstore)
    test_data = process_data(test_df.to_dict('records'), embedding_model, test_vectorstore)
    
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))
    
    return DatasetDict({
        "train": train_dataset, 
        "validation": val_dataset, 
        "test": test_dataset
    }), val_df

def train():
    """Single GPU training function"""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Force single GPU usage
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only first GPU
    
    # 데이터셋 로드
    dataset, val_df = load_dataset()
    
    # 모델 및 토크나이저 로드
    model, tokenizer = prepare_model_and_tokenizer()
    
    # 데이터셋 토큰화
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # 데이터 콜레이터 설정
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Setup training arguments for single GPU
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        bf16=True,  # Use bfloat16 if available
        bf16_full_eval=True,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        # No distributed training
        local_rank=-1,
        ddp_find_unused_parameters=False
    )
    
    # 트레이너 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 모니터링 콜백 추가
    monitor_callback = TrainingMonitorCallback(
        OUTPUT_DIR, 
        tokenizer, 
        val_df
    )
    trainer.add_callback(monitor_callback)
    
    # 학습 시작
    logger.info("훈련 시작...")
    
    try:
        train_result = trainer.train()
        
        # 최종 모델 저장
        logger.info("최종 모델 저장...")
        final_model_dir = os.path.join(OUTPUT_DIR, "final")
        trainer.save_model(final_model_dir)
        
        # 최종 메트릭 기록
        metrics = train_result.metrics
        logger.info(f"훈련 완료. 메트릭: {metrics}")
        
        # 테스트셋 평가
        logger.info("테스트셋에서 최종 평가 수행 중...")
        test_results = trainer.evaluate(tokenized_dataset["test"])
        logger.info(f"테스트 결과: {test_results}")
        
        return metrics
            
    except Exception as e:
        logger.error(f"훈련 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    try:
        # Disable tokenizers parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Specifically for using single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only first GPU
        
        # Run training
        train()
        
        logger.info("Training completed successfully!")
            
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise