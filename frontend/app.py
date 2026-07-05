import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("PDF 기반 RAG 챗봇")

uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])

if uploaded_file and st.button("업로드 및 인덱싱"):
    with st.spinner("PDF를 업로드하고 인덱싱하는 중입니다..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_URL}/upload", files=files, timeout=120)

    if response.ok:
        result = response.json()
        st.success(f"{result['filename']} 인덱싱 완료: {result['chunks']}개 chunk")
    else:
        st.error(response.text)

question = st.text_input("질문 입력")

if st.button("질문하기"):
    if not question.strip():
        st.warning("질문을 입력해 주세요.")
    else:
        with st.spinner("답변을 생성하는 중입니다..."):
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=120,
            )

        if response.ok:
            result = response.json()
            st.subheader("답변")
            st.write(result["answer"])

            st.subheader("참고 문맥")
            for i, context in enumerate(result["contexts"], start=1):
                with st.expander(f"문맥 {i} - {context.get('source')}"):
                    st.write(context["text"])
        else:
            st.error(response.text)
