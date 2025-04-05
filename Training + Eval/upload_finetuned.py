import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# 환경 변수에서 HF 토큰 불러오기
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN 환경변수가 설정되어 있지 않습니다.")

# 베이스 모델 이름 (예: "gpt2" 또는 원하는 다른 모델)
base_model_name = "gpt2"  # 예시입니다. 실제 베이스 모델 이름으로 변경하세요.

# 베이스 모델과 토크나이저 로딩
print("베이스 모델 및 토크나이저 로딩 중...")
model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 파인튜닝 어댑터 파일 경로
adapter_config_path = "path/to/adapter_config.json"       # 실제 경로로 변경하세요.
adapter_model_path = "path/to/adapter_model.safetensors"     # 실제 경로로 변경하세요.

# 어댑터 설정 로딩 (PEFT Config)
print("어댑터 설정 로딩 중...")
peft_config = PeftConfig.from_json_file(adapter_config_path)

# 베이스 모델에 어댑터 로딩 (파인튜닝된 파라미터 병합)
print("어댑터 파라미터 병합 중...")
model = PeftModel.from_pretrained(model, adapter_model_path, is_trainable=False)

# 전체 모델을 저장할 로컬 디렉토리 지정
output_dir = "./full_finetuned_model"
print("전체 모델 저장 중...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# 업로드할 Hugging Face Hub 리포지토리 이름 설정 (예: "your_username/full-finetuned-model")
repo_id = "your_username/full-finetuned-model"  # 자신의 사용자명과 원하는 리포지토리 이름으로 변경하세요.

print("전체 모델 Hugging Face Hub에 업로드 중...")
model.push_to_hub(repo_id, use_auth_token=hf_token)
tokenizer.push_to_hub(repo_id, use_auth_token=hf_token)
 
print("전체 모델 업로드 완료!")