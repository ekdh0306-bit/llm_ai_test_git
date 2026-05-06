"""
문제 7. RAG 로 문서 기반 답변 챗봇 (화성시 도시기본계획)

- 대상 PDF: '화성시고시제2018-642호.pdf'
- 파이프라인:
    1) PyPDFLoader 로 PDF 로드
    2) RecursiveCharacterTextSplitter 로 청크 분할
    3) OpenAIEmbeddings 로 임베딩
    4) FAISS 벡터스토어에 저장
    5) as_retriever() 로 리트리버 생성
    6) LCEL 체인 (context + question → prompt → ChatOpenAI → StrOutputParser)
    7) 콘솔에서 질문을 받아 반복 답변

사전 설치:
    pip install langchain langchain-openai langchain-community langchain-core
                python-dotenv pypdf faiss-cpu
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = "화성시고시제2018-642호.pdf"


def build_retriever(pdf_path: str):
    """PDF 를 로드 → 분할 → 임베딩 → FAISS 인덱스 → retriever 반환."""
    # 1) Load
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[INFO] PDF 페이지 수: {len(documents)}")

    # 2) Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] 청크 개수: {len(chunks)}")

    # 3) Embed + 4) Store (FAISS)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5) Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever


def format_docs(docs):
    """검색된 Document 들을 하나의 문자열 컨텍스트로 합친다."""
    return "\n\n".join(d.page_content for d in docs)


def build_rag_chain(retriever):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 화성시 도시기본계획 문서를 근거로 답변하는 챗봇이야.\n"
                "아래 [컨텍스트]에 주어진 내용만을 사용해서 한국어로 답해줘.\n"
                "컨텍스트에 답이 없다면 솔직하게 '문서에서 해당 정보를 찾을 수 없습니다.' 라고 답해.\n\n"
                "[컨텍스트]\n{context}",
            ),
            ("user", "{question}"),
        ]
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    parser = StrOutputParser()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | parser
    )
    return chain


def main():
    print(f"[INFO] PDF 로드 시작: {PDF_PATH}")
    retriever = build_retriever(PDF_PATH)
    chain = build_rag_chain(retriever)

    print("\n=== 화성시 도시기본계획 RAG 챗봇 ===")
    print("질문을 입력하세요. 종료하려면 'exit' 입력.\n")

    while True:
        question = input("Q: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("종료합니다.")
            break

        answer = chain.invoke(question)
        print(f"\nA: {answer}\n")


if __name__ == "__main__":
    main()
