# Fundit

편딧은 다양한 보조금 정보를 한 곳에 모아 사용자에게 제공하고,  
보조금 신청 과정을 간소화하여 빠르고 쉽게 지원을 받을 수 있도록 필요한 도움을 실시간으로 제공합니다.

---

## 🖼 주요 화면 구성

- [메인화면](https://github.com/user-attachments/assets/56ee5772-8d1b-4e1d-b08d-a1d79967b4bd)
- [회원가입 / 로그인](https://github.com/user-attachments/assets/e3b6abeb-0fae-4db3-9988-edade04593b8)
- [진입화면](https://github.com/user-attachments/assets/cd6eb6a4-af5c-46e4-98f7-7c02ba3771bd)
- [채팅화면](https://github.com/user-attachments/assets/74b048aa-f866-4cd0-bfe6-d7ddff698671)

---

## 📁 Repo 구조

- `appendix/` – 데이터 전처리, 프롬프트 정리
- `local/` – 웹 프론트엔드(React) 및 백엔드(FastAPI)
- `src_aws_eks_iac_cicd/` – AWS 인프라 배포 자동화 (IaC, EKS 등)

---

## 👥 Team
![SeSAC_final_project](https://github.com/user-attachments/assets/415c0144-fb58-42b2-a208-909f3dac5ae9)

## ✅ Commit Convention

| Prefix | 설명 |
|--------|------|
| `Feat` | 새로운 기능 추가 |
| `Fix` | 버그 수정 |
| `Docs` | 문서 추가 또는 수정 |
| `Style` | 코드 포맷 변경, 세미콜론 누락 등 |
| `Refactor` | 코드 리팩토링 (기능 변화 없음) |
| `Test` | 테스트 코드 추가 |
| `Chore` | 빌드 설정, 패키지 관리 등 기타 작업 |
| `Rename` | 파일 또는 폴더 이름 변경 |
| `Remove` | 파일 삭제 작업 |

---

# appendix - Data Preprocessing

이 폴더는 Fundit 프로젝트의 **정책 데이터 전처리 Python 스크립트**를 포함합니다.  
특히 `data_process1/` 디렉토리는 정책 관련 데이터를 AI 챗봇이 이해하기 쉽도록 정제하고 분류하는 데 목적이 있습니다.

---

## 📁 appendix 폴더 구조

 ⚠️ 본 프로젝트에는 핵심 기능 구현에 필요한 코드만 엄선하여 포함했습니다.

---

## 📄 주요 스크립트 설명

| 파일명 | 설명 |
|--------|------|
| `0-1.dp_area8780_...py` | 정책 데이터를 행정구역 기준으로 분류합니다. |
| `0-2.dp_support.py` | 지원 항목(현금, 바우처 등)을 기준으로 분류합니다. |
| `0-3.dp_application_method.py` | 신청 방법을 온라인/방문 등으로 분류합니다. |
| `0-4.dp_benefit_multi_cate.py` | 수혜 기준을 여러 조건으로 다중 분류 처리합니다. |

---

## ▶ 실행 방법

```bash
# 예: 수혜 기준 분류 실행
cd appendix/data_process1
python 0-4.dp_benefit_multi_cate.py
---
'''