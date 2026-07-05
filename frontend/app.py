import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("PDF 기반 RAG 챗봇")

if "messages" not in st.session_state:
    st.session_state.messages = []

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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant" and message.get("contexts"):
            with st.expander("참고 문맥"):
                for i, context in enumerate(message["contexts"], start=1):
                    st.markdown(f"**문맥 {i} - {context.get('source')}**")
                    st.write(context["text"])

question = st.chat_input("PDF 내용에 대해 질문하세요")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하는 중입니다..."):
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=120,
            )

        if response.ok:
            result = response.json()
            answer = result["answer"]
            contexts = result["contexts"]

            st.write(answer)
            if contexts:
                with st.expander("참고 문맥"):
                    for i, context in enumerate(contexts, start=1):
                        st.markdown(f"**문맥 {i} - {context.get('source')}**")
                        st.write(context["text"])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "contexts": contexts,
                }
            )
        else:
            error_message = response.text
            st.error(error_message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "contexts": [],
                }
            )
