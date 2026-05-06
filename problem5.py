"""
문제 5. LangChain 을 이용해 결과물 도출

- 질문(Q): "내 이름이 뭐야?"
- LangChain 사용을 명시적으로 보여주기 위해 다음 컴포넌트를 모두 사용한다:
    1) ChatPromptTemplate     : 프롬프트 템플릿 (변수 주입)
    2) ChatOpenAI             : LangChain 의 OpenAI Chat 래퍼
    3) StrOutputParser        : 출력 파서
    4) LCEL 파이프(|)          : 체인 구성

사전 설치: pip install langchain langchain-openai langchain-core python-dotenv
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1) 프롬프트 템플릿 정의 (system 에 이름을 주입해 모델이 기억하도록 함)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "사용자의 이름은 {name} 이다. 사용자의 질문에 친절하고 자연스럽게 한국어로 답해줘."),
        ("user", "{question}"),
    ]
)

# 2) LLM (LangChain 의 ChatOpenAI 래퍼 사용)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# 3) 출력 파서
parser = StrOutputParser()

# 4) LCEL 체인 구성: prompt → llm → parser
chain = prompt | llm | parser

# 5) 체인 실행 (이름은 콘솔 입력으로 받음)
name = input("당신의 이름을 입력하세요: ").strip()
if not name:
    name = "익명"

question = "내 이름이 뭐야?"
result = chain.invoke({"name": name, "question": question})

print("=" * 50)
print(f"Q: {question}")
print(f"A: {result}")
print("=" * 50)
