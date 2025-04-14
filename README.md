# appendix - Data Preprocessing

이 폴더는 Fundit 프로젝트의 **정책 데이터 전처리 Python 스크립트**를 포함합니다.  
특히 `data_process1/` 디렉토리는 정책 관련 데이터를 AI 챗봇이 이해하기 쉽도록 정제하고 분류하는 데 목적이 있습니다.

---

## 📁 폴더 구조

> ❗ 참고: 위 파일들 외에도 다양한 전처리 스크립트가 있었지만,  
> 본 프로젝트에는 **핵심 기능 구현에 필요한 코드만 엄선하여 포함**했습니다.

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
