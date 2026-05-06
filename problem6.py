"""
문제 6. LangChain 을 이용해 결과물 도출 (질문: "현재 시간은?")

- 5번 문제와 동일한 LangChain 컴포넌트 구조를 유지하되, 질문을 "현재 시간은?" 으로 변경.
- LLM 은 스스로 현재 시간을 알 수 없으므로, Python datetime 으로 얻은 현재 시각을
  ChatPromptTemplate 의 변수({current_time})로 주입한다.

사용 컴포넌트:
    1) ChatPromptTemplate
    2) ChatOpenAI
    3) StrOutputParser
    4) LCEL 파이프(|)

사전 설치: pip install langchain langchain-openai langchain-core python-dotenv
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1) 프롬프트 템플릿 (system 에 현재 시간을 주입)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "현재 시간은 {current_time} 이다. "
            "사용자의 질문에 한국어로 자연스럽고 친절하게 답해줘.",
        ),
        ("user", "{question}"),
    ]
)

# 2) LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# 3) 출력 파서
parser = StrOutputParser()

# 4) LCEL 체인 구성
chain = prompt | llm | parser

# 5) 체인 실행
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
question = "현재 시간은?"
result = chain.invoke({"current_time": current_time, "question": question})

print("=" * 50)
print(f"[현재 시각 주입값] {current_time}")
print(f"Q: {question}")
print(f"A: {result}")
print("=" * 50)
