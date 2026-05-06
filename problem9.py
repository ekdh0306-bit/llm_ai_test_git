"""
문제 9. LangGraph 로 만드는 기본 챗봇 (메모리 보유)

시나리오:
    1턴) "난 {학생이름}이야~"
    2턴) "내 이름이 뭐라고?"   → 봇이 1턴의 이름을 기억해 답변

LangGraph 사용 능력 시연 포인트:
    1) StateGraph         : 그래프(상태머신) 정의
    2) State + add_messages reducer : 메시지가 누적되도록 합치는 reducer
    3) add_node / add_edge / START / END : 그래프 토폴로지 구성
    4) MemorySaver        : 체크포인터(thread 기반 메모리) → 같은 thread_id 호출 시
                            이전 상태가 자동 복원되어 모델이 이전 발화를 "기억"
    5) compile()          : 실행 가능한 그래프(앱)로 컴파일
    6) invoke(..., config={"configurable": {"thread_id": ...}}) : 세션별 호출

사전 설치: pip install langgraph langchain-openai langchain-core python-dotenv
"""

import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# 1) State 정의 (messages 는 add_messages reducer 로 누적 합쳐짐)
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


# 2) LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# 3) 노드: LLM 을 호출해 응답을 messages 에 append
def chatbot_node(state: ChatState) -> ChatState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# 4) 그래프 구성
builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 5) 체크포인터(메모리)와 함께 컴파일
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


def chat(user_text: str, thread_id: str, system_prompt: str | None = None) -> str:
    """주어진 thread_id 의 세션에 user_text 를 보내고 봇의 응답 텍스트를 반환."""
    new_messages = []
    # 첫 호출에만 system 메시지를 넣는다 (thread 가 비어있는 경우).
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    is_first_turn = (state is None) or (not state.values) or (not state.values.get("messages"))
    if is_first_turn and system_prompt:
        new_messages.append(SystemMessage(content=system_prompt))
    new_messages.append(HumanMessage(content=user_text))

    result = graph.invoke(
        {"messages": new_messages},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


def main():
    name = input("학생 이름을 입력하세요: ").strip() or "익명"
    thread_id = "exam-session-1"

    system_prompt = (
        "너는 학생과 대화하는 친근한 챗봇이야. "
        "학생이 알려준 정보(특히 이름)를 기억하고, 한국어로 자연스럽게 답해줘."
    )

    # 1턴
    q1 = f"난 {name}이야~"
    print(f"\n[1턴] Q: {q1}")
    a1 = chat(q1, thread_id, system_prompt=system_prompt)
    print(f"[1턴] A: {a1}")

    # 2턴 (같은 thread_id → 체크포인터에서 이전 messages 복원)
    q2 = "내 이름이 뭐라고?"
    print(f"\n[2턴] Q: {q2}")
    a2 = chat(q2, thread_id)
    print(f"[2턴] A: {a2}")


if __name__ == "__main__":
    main()
