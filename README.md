<<<<<<< HEAD
# appendix
=======
1.Crawling
OPENAPI API KEY 부분은 돌려볼려면 API 발급받고 키 값 .env 파일 만들어서 새로 집어넣어야됨
블로그 크롤링은 주기적으로 네이버에서 형식을 바꿔서, 프로젝트 당시에는 작동했던 코드 제대로 안 돌아갈 수 있음

2.Preprocessing
프로젝트에서는 GEMINI로 진행을 했는데, API 발급받고 키 값을 .env 파일 만들어서 새로 집어넣어야됨. API 불러오는 부분만 바꾼다면 다른 LLM API를 써도 돌아갈 것

3.Training + Eval
Text2SQL에 쓸 LLM fine tuning 할 때 썻던 코드. 돌려볼려면 GPU 리소스 필요함. 그리고 프로젝트에선 튜닝한 모델 Huggingface에 올리는게 편해서 허깅페이스에 업로드 까지 진행했지만 LLM을 로컬로 돌리거나 다른 곳에 파일 옮길 필요가 없다면 저 부분은 생략해도 무방. 

4. VLLM
pip install vllm
이 명령어로 vllm을 깔아주고, llm 모델이 로컬에 있다면 
python -m vllm.entrypoints.api_server --model /llm모델 위치 --port 원하는 포트 번호
llm 모델이 허깅페이스에 있다면 
python -m vllm.entrypoints.api_server --model 허깅페이스 모델 주소 --port 원하는 포트 번호
커맨드로 실행시키면 된다. 