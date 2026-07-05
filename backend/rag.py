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


def ensure_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(".env 파일에 OPENAI_API_KEY를 설정해 주세요.")


def get_openai_client():
    ensure_openai_api_key()

    from openai import OpenAI

    return OpenAI()


def get_embeddings():
    ensure_openai_api_key()

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_vectorstore():
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    clean_text = " ".join(text.split())
    if not clean_text:
        return []

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunk_overlap = min(overlap, max(chunk_size - 1, 0))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(clean_text)


def load_pdf_documents(pdf_path: Path):
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents: list, chunk_size: int = 1000, overlap: int = 150):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunk_overlap = min(overlap, max(chunk_size - 1, 0))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def index_pdf(pdf_path: Path) -> dict:
    documents = load_pdf_documents(pdf_path)
    chunks = split_documents(documents)

    if not chunks:
        return {"chunks": 0}

    for index, chunk in enumerate(chunks):
        chunk.metadata["source"] = pdf_path.name
        chunk.metadata["chunk"] = index

    ids = [f"{pdf_path.stem}-{i}" for i in range(len(chunks))]

    vectorstore = get_vectorstore()
    vectorstore.delete(where={"source": pdf_path.name})
    vectorstore.add_documents(documents=chunks, ids=ids)

    return {"chunks": len(chunks)}


def answer_question(question: str, top_k: int = 4) -> dict:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    documents = retriever.invoke(question)

    if not documents:
        return {
            "answer": "검색된 문맥이 없습니다. 먼저 PDF를 업로드해 주세요.",
            "contexts": [],
        }

    context_text = "\n\n".join(
        f"[{i + 1}] {document.page_content}"
        for i, document in enumerate(documents)
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 PDF 문서를 기반으로 답변하는 도우미입니다."),
            (
                "user",
                "다음 문맥을 참고해서 사용자의 질문에 한국어로 답변하세요.\n"
                "문맥에 답이 없으면 모른다고 말하세요.\n\n"
                "문맥:\n{context}\n\n"
                "질문:\n{question}",
            ),
        ]
    )
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.2,
    )
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": question})

    return {
        "answer": response.content,
        "contexts": [
            {
                "text": document.page_content,
                "source": document.metadata.get("source"),
                "chunk": document.metadata.get("chunk"),
            }
            for document in documents
        ],
    }
