from huggingface_hub import snapshot_download
import os
from huggingface_hub import login

hf_token = "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"
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN이 설정되지 않았습니다.")
login(token=hf_token)

def download_model(repo_id: str, local_dir: str):
    """
    Hugging Face Hub에서 repo_id에 해당하는 모델 리포지토리의 스냅샷을
    local_dir 경로에 다운로드합니다.
    """
    print(f"모델 리포지토리 '{repo_id}' 다운로드를 시작합니다...")
    # 모델 리포지토리의 스냅샷을 다운로드합니다.
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"모델 리포지토리 '{repo_id}' 다운로드 완료! 저장 경로: {local_dir}")

if __name__ == "__main__":
    # 다운로드할 모델 리포지토리 ID (예시)
    repo_id = "kky2455/mistral-7b-v3-tuned"
    # 모델 파일들을 저장할 로컬 디렉터리
    local_dir = "./full_finetuned_model"
    os.makedirs(local_dir, exist_ok=True)
    download_model(repo_id, local_dir)
