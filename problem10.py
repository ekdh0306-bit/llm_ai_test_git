"""
문제 10. LangGraph 로 만드는 기본 챗봇 + 실시간 날씨 도구

시나리오:
    1턴) "난 {학생이름}이야~"
    2턴) "현재 수원 날씨는 어때?"   → LLM 이 도구를 호출해 실시간 날씨를 가져와 답변

문제 9 대비 차이점:
    - "현재 날씨"는 모델 자체 지식만으로는 답할 수 없으므로 외부 도구가 필요.
    - LangGraph 에 "도구 호출 노드(ToolNode)" 와 "조건부 엣지" 를 추가했다.

LangGraph 사용 능력 시연 포인트:
    1) StateGraph + add_messages reducer
    2) START / END / add_node / add_edge
    3) add_conditional_edges + tools_condition (조건부 라우팅)
    4) ToolNode (prebuilt) : 도구 실행 노드
    5) ChatOpenAI.bind_tools : LLM 에 도구 바인딩
    6) MemorySaver + thread_id : 1턴 → 2턴 메모리 유지

도구 구현:
    - get_current_weather(location) 는 OpenAI Responses API + web_search_preview 로
      실시간 웹 검색을 수행해 한국어 날씨 요약 텍스트를 반환한다.

사전 설치: pip install langgraph langchain-openai langchain-core python-dotenv openai
"""

import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

_openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── 도구 정의 ──────────────────────────────────────────────
@tool
def get_current_weather(location: str) -> str:
    """주어진 지역의 '현재' 날씨를 인터넷 검색으로 조회해 한국어 요약을 반환한다.

    Args:
        location: 한국어 또는 영문 지역명. 예) "수원", "Suwon"
    """
    response = _openai_client.responses.create(
        model="gpt-4o-mini",
        tools=[
            {
                "type": "web_search_preview",
                "search_context_size": "high",
                "user_location": {"type": "approximate", "country": "KR"},
            }
        ],
        input=(
            f"{location} 의 현재 날씨를 인터넷에서 검색해서 한국어로 간단히 정리해줘. "
            "기온/체감/강수/풍속/하늘상태를 포함하고, 출처 매체 이름도 한 줄 적어줘."
        ),
    )
    return response.output_text


# ── State 정의 ─────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


# ── LLM (도구 바인딩) ──────────────────────────────────────
TOOLS = [get_current_weather]
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
).bind_tools(TOOLS)


# ── 노드 ──────────────────────────────────────────────────
def chatbot_node(state: ChatState) -> ChatState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(TOOLS)


# ── 그래프 ─────────────────────────────────────────────────
builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "chatbot")
# tool_calls 가 있으면 'tools' 로, 없으면 END 로 라우팅
builder.add_conditional_edges("chatbot", tools_condition)
# 도구 실행 후에는 다시 chatbot 으로 돌아가 최종 답변을 생성
builder.add_edge("tools", "chatbot")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ── 헬퍼: 한 턴 호출 ───────────────────────────────────────
def chat(user_text: str, thread_id: str, system_prompt: str | None = None) -> str:
    new_messages = []
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
    thread_id = "exam-session-10"

    system_prompt = (
        "너는 학생과 대화하는 친근한 한국어 챗봇이야. "
        "학생이 알려준 정보를 기억하고 자연스럽게 답해줘. "
        "현재 날씨처럼 실시간 정보가 필요한 질문이면 반드시 get_current_weather 도구를 사용해."
    )

    # 1턴
    q1 = f"난 {name}이야~"
    print(f"\n[1턴] Q: {q1}")
    a1 = chat(q1, thread_id, system_prompt=system_prompt)
    print(f"[1턴] A: {a1}")

    # 2턴 (실시간 날씨 → 도구 호출 발생)
    q2 = "현재 수원 날씨는 어때?"
    print(f"\n[2턴] Q: {q2}")
    a2 = chat(q2, thread_id)
    print(f"[2턴] A: {a2}")


if __name__ == "__main__":
    main()
