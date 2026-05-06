"""
문제 1. 퓨샷 프롬프팅 (Few-shot Prompting)

- 필수 system: "너는 유치원생이야 유치원생처럼 답변해줘"
- 필수 user  : "강아지"
- OpenAI Chat Completions 규격에 따라 system → (예시 user/assistant 쌍) → user("강아지") 순으로 구성한다.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

messages = [
    {
        "role": "system",
        "content": "너는 유치원생이야 유치원생처럼 답변해줘",
    },
    {
        "role": "user",
        "content": "고양이",
    },
    {
        "role": "assistant",
        "content": "야옹야옹",
    },
    {
        "role": "user",
        "content": "사자",
    },
    {
        "role": "assistant",
        "content": "어흥",
    },
    {
        "role": "user",
        "content": "강아지",
    },
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

print(response.choices[0].message.content)
