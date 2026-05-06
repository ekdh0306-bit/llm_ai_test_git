"""
문제 2. Streamlit 챗봇 UI (세션 활용)

- st.session_state.messages 에 대화 이력을 누적해 모델이 이전 발화를 "기억"하도록 함.
- 시나리오:
    1) 사용자: "안녕? 나는 OOO이야"
    2) 사용자: "내 이름이 뭐야?"  → 모델이 1)의 이름을 회상해서 답변.

실행: streamlit run problem2.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

st.title("💬 이름 기억하는 챗봇")
st.caption("먼저 '안녕? 나는 OOO이야'로 이름을 알려준 뒤, '내 이름이 뭐야?'라고 물어보세요.")

# ── 세션 상태 초기화 ──
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "너는 친근한 챗봇이야. 사용자가 알려준 정보(특히 이름)를 기억하고 자연스럽게 대화해줘.",
        }
    ]

# ── 기존 대화 렌더링 (system 제외) ──
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 사용자 입력 ──
user_input = st.chat_input("메시지를 입력하세요")

if user_input:
    # 1) 사용자 메시지 추가 + 즉시 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) OpenAI 호출 (누적 messages 전체 전달 → 이전 이름 기억 가능)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages,
    )
    assistant_reply = response.choices[0].message.content

    # 3) 어시스턴트 응답 저장 + 표시
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
