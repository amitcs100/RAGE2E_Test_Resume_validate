"""Streamlit UI for asking grounded questions about uploaded resumes."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Iterable

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

SUPPORTED_TYPES = ["pdf", "docx", "txt"]
DEFAULT_MODEL = "gpt-4o-mini"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a careful resume analysis assistant.
Answer using only the resume excerpts supplied below.

Rules:
- Never invent a skill, employer, date, qualification, or achievement.
- If the excerpts do not contain the answer, say: "I couldn't find that in the uploaded resumes."
- When several candidates are present, clearly identify which candidate each claim concerns.
- Be concise, but include enough detail to answer the question.
- End factual claims with source markers such as [Resume.pdf, page 2] using the
  source labels in the context. Do not create source markers that are not present.

Resume excerpts:
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def _api_key() -> str:
    """Read the key from Streamlit secrets, the environment, or the sidebar."""
    secret_key = ""
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        # Streamlit raises a dedicated exception when no secrets file exists.
        pass
    return (
        secret_key
        or os.getenv("OPENAI_API_KEY", "")
        or st.session_state.get("openai_api_key", "")
    ).strip()


def _uploaded_files_signature(uploaded_files: Iterable) -> str:
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def load_resume(uploaded_file) -> list[Document]:
    """Load an uploaded PDF, DOCX, or TXT resume and preserve its display name."""
    suffix = Path(uploaded_file.name).suffix.lower()
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": lambda path: TextLoader(path, encoding="utf-8", autodetect_encoding=True),
    }
    if suffix not in loaders:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        documents = loaders[suffix](temp_path).load()
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

    for document in documents:
        document.metadata["source"] = uploaded_file.name
        if isinstance(document.metadata.get("page"), int):
            document.metadata["page"] += 1
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks: list[Document], api_key: str) -> FAISS:
    if not chunks:
        raise ValueError("No readable text was found in the uploaded files.")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )
    return FAISS.from_documents(chunks, embeddings)


def create_rag_system(uploaded_files: list, api_key: str):
    """Build a retriever for all uploaded resumes."""
    documents: list[Document] = []
    for uploaded_file in uploaded_files:
        documents.extend(load_resume(uploaded_file))
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks, api_key)
    return vector_store.as_retriever(search_kwargs={"k": min(6, len(chunks))})


def format_documents(documents: list[Document]) -> str:
    sections = []
    for document in documents:
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page")
        location = f"{source}, page {page}" if page else str(source)
        sections.append(f"[{location}]\n{document.page_content.strip()}")
    return "\n\n---\n\n".join(sections)


def answer_question(question: str, retriever, api_key: str) -> tuple[str, list[Document]]:
    relevant_documents = retriever.invoke(question)
    prompt_value = PROMPT.invoke(
        {"context": format_documents(relevant_documents), "question": question}
    )
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", DEFAULT_MODEL),
        temperature=0,
        max_tokens=800,
        api_key=api_key,
    )
    response = llm.invoke(prompt_value)
    return str(response.content), relevant_documents


def _render_sources(documents: list[Document]) -> None:
    seen = set()
    with st.expander("Retrieved resume excerpts"):
        for document in documents:
            source = document.metadata.get("source", "Unknown")
            page = document.metadata.get("page")
            key = (source, page, document.page_content)
            if key in seen:
                continue
            seen.add(key)
            label = f"{source} — page {page}" if page else str(source)
            st.markdown(f"**{label}**")
            st.write(document.page_content.strip())


def main() -> None:
    st.set_page_config(
        page_title="Resume RAG Assistant",
        page_icon="📄",
        layout="wide",
    )
    st.title("Resume RAG Assistant")
    st.caption(
        "Upload resumes, build a private in-session search index, and ask questions "
        "whose answers stay grounded in the uploaded text."
    )

    with st.sidebar:
        st.header("Setup")
        if not (_api_key()):
            st.text_input(
                "OpenAI API key",
                type="password",
                key="openai_api_key",
                help="Used for this session only. For deployment, use Streamlit Secrets.",
            )
        else:
            st.success("OpenAI API key is configured.")
        st.markdown(
            "Supported formats: PDF, DOCX, TXT. Scanned/image-only PDFs need OCR "
            "before upload."
        )

    uploaded_files = st.file_uploader(
        "Upload one or more resumes",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )

    current_signature = _uploaded_files_signature(uploaded_files) if uploaded_files else ""
    index_is_current = (
        st.session_state.get("file_signature") == current_signature
        and st.session_state.get("retriever") is not None
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) selected.")
        if st.button(
            "Build resume index",
            type="primary",
            disabled=not bool(_api_key()),
        ):
            with st.spinner("Reading resumes and creating the search index..."):
                try:
                    st.session_state.retriever = create_rag_system(
                        uploaded_files, _api_key()
                    )
                    st.session_state.file_signature = current_signature
                    st.session_state.messages = []
                    index_is_current = True
                    st.success("Resume index is ready.")
                except Exception as error:
                    st.session_state.retriever = None
                    st.error(f"Could not build the index: {error}")
    else:
        st.info("Upload at least one resume to begin.")

    if uploaded_files and not _api_key():
        st.warning("Add an OpenAI API key in the sidebar to build the index.")
    elif uploaded_files and not index_is_current:
        st.info("Build the resume index before asking questions.")

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                _render_sources(message["sources"])

    question = st.chat_input(
        "Ask about experience, skills, education, or candidate fit…",
        disabled=not index_is_current,
    )
    if question:
        st.session_state.setdefault("messages", []).append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the resumes..."):
                try:
                    answer, sources = answer_question(
                        question, st.session_state.retriever, _api_key()
                    )
                    st.markdown(answer)
                    _render_sources(sources)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                except Exception as error:
                    st.error(f"Could not answer the question: {error}")


if __name__ == "__main__":
    main()
