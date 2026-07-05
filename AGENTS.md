이 프로젝트는 교육용 PDF 기반 RAG 챗봇 서비스입니다.

기술 스택:

* Backend: FastAPI
* Frontend: Streamlit
* Vector DB: Chroma
* PDF Loader: PyMuPDF 또는 pypdf
* Embedding: OpenAI Embeddings
* Answer Generation: OpenAI Chat Completions API

개발 원칙:

* 교육 목적이므로 코드는 최대한 단순하게 작성한다.
* 불필요한 추상화, 복잡한 클래스 구조, 과도한 예외 처리는 피한다.
* backend와 frontend를 명확히 분리한다.
* API 키는 코드에 직접 작성하지 않고 `.env`에서 읽는다.
* 업로드된 PDF는 `uploads/`에 저장한다.
* Chroma DB는 `chroma/`에 저장한다.
* 답변 API는 답변과 검색된 문맥을 함께 반환한다.
* 학생들이 이해하기 쉽도록 주요 함수에는 짧은 주석을 작성한다.
