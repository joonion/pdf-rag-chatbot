# PDF RAG Chatbot

FastAPI와 Streamlit으로 만든 교육용 PDF 기반 RAG 챗봇입니다. 사용자는 PDF를 업로드하고, OpenAI Embeddings와 Chroma DB로 저장된 문맥을 기반으로 질문할 수 있습니다.

- FastAPI는 PDF 저장, 텍스트 추출, 임베딩, Chroma 검색, 답변 생성을 담당합니다.
- Streamlit은 PDF 업로드와 질문/답변 화면만 담당합니다.

## Project Structure

```text
backend/
  api.py               # FastAPI API 서버
  rag.py               # PDF 추출, chunking, embedding, 검색, 답변 생성
  requirements.txt
frontend/
  app.py               # Streamlit 화면
  requirements.txt
data/
  uploads/             # 업로드된 PDF 저장 위치
  chroma/              # Chroma DB 저장 위치
```

## Setup

```bash
copy .env.example .env
```

`.env` 파일에 `OPENAI_API_KEY` 값을 입력합니다.

예시:

```bash
OPENAI_API_KEY=sk-...
```

## Run Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

## Run Frontend

새 터미널에서 실행합니다.

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 표시되는 Streamlit 주소로 접속한 뒤 PDF를 업로드하고 질문을 입력합니다.

FastAPI 서버는 `http://localhost:8000`, Streamlit 화면은 보통 `http://localhost:8501`에서 실행됩니다.

## Run with Docker Compose

Docker Compose는 FastAPI와 Streamlit을 함께 실행합니다.

```bash
copy .env.example .env
docker compose up --build
```

실행 후 접속 주소:

- FastAPI: `http://localhost:8000`
- FastAPI Docs: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`

## Run Tests

외부 API 호출 없이 기본 로직을 테스트합니다.

```bash
python -m unittest discover -s tests
```

## API

- `POST /upload`: PDF 파일을 업로드하고 Chroma DB에 chunk를 저장합니다.
- `POST /ask`: 질문을 받아 관련 문맥을 검색하고 답변을 생성합니다.

`/ask` 응답 예시:

```json
{
  "answer": "답변 내용",
  "contexts": [
    {
      "text": "참고 문맥",
      "source": "sample.pdf",
      "chunk": 0
    }
  ]
}
```
