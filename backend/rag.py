import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
COLLECTION_NAME = "pdf_chunks"


def get_openai_client():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(".env 파일에 OPENAI_API_KEY를 설정해 주세요.")

    from openai import OpenAI

    return OpenAI()


def get_collection():
    import chromadb

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def extract_text_from_pdf(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    chunks = []
    start = 0
    clean_text = " ".join(text.split())

    while start < len(clean_text):
        end = start + chunk_size
        chunk = clean_text[start:end]
        if chunk:
            chunks.append(chunk)
        if end >= len(clean_text):
            break
        start = end - overlap

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_openai_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def index_pdf(pdf_path: Path) -> dict:
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)

    if not chunks:
        return {"chunks": 0}

    embeddings = embed_texts(chunks)
    ids = [f"{pdf_path.stem}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_path.name, "chunk": i} for i in range(len(chunks))]

    collection = get_collection()
    collection.delete(where={"source": pdf_path.name})
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {"chunks": len(chunks)}


def answer_question(question: str, top_k: int = 4) -> dict:
    question_embedding = embed_texts([question])[0]
    collection = get_collection()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    contexts = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not contexts:
        return {
            "answer": "검색된 문맥이 없습니다. 먼저 PDF를 업로드해 주세요.",
            "contexts": [],
        }

    context_text = "\n\n".join(
        f"[{i + 1}] {context}" for i, context in enumerate(contexts)
    )

    prompt = f"""
다음 문맥을 참고해서 사용자의 질문에 한국어로 답변하세요.
문맥에 답이 없으면 모른다고 말하세요.

문맥:
{context_text}

질문:
{question}
"""

    client = get_openai_client()
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "당신은 PDF 문서를 기반으로 답변하는 도우미입니다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return {
        "answer": completion.choices[0].message.content,
        "contexts": [
            {
                "text": context,
                "source": metadata.get("source"),
                "chunk": metadata.get("chunk"),
            }
            for context, metadata in zip(contexts, metadatas)
        ],
    }
