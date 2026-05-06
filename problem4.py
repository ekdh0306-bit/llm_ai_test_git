"""
문제 4. GPT 비전에게 이미지 설명 요청하기

- 인터넷 교통 이벤트(사고/화재 등) 이미지 URL 1장을 GPT-4o-mini 비전에 전달.
- system 으로 "교통 이벤트 분석가" 역할 부여.
- user 메시지를 (text + image_url) 멀티파트로 구성.
- 모델 응답(이미지 설명)을 출력.

사전 설치: pip install openai python-dotenv
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 인터넷에 공개된 교통 이벤트(사고/화재) 이미지 URL.
# 필요 시 다른 공개 이미지 URL로 교체해 사용해도 됨.
IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/7/71/"
    "Indonesian_fire_fighters_during_a_traffic_accident.jpg"
)


def describe_traffic_event(image_url: str) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": (
                "너는 교통 이벤트(사고, 화재, 차량 고장 등)를 분석하는 AI 분석가야. "
                "사용자가 보낸 이미지를 보고 다음을 한국어로 설명해줘:\n"
                "1) 이미지에서 보이는 장면 요약\n"
                "2) 발생한 교통 이벤트의 종류(예: 추돌사고, 차량 화재 등)\n"
                "3) 식별 가능한 차량/사물/환경 정보\n"
                "4) 위험 요소 및 주의사항\n"
                "확실하지 않은 부분은 단정하지 말고 '추정'으로 표시해."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "이 이미지에서 일어난 교통 이벤트를 분석해서 설명해줘.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return response.choices[0].message.content


def main():
    print(f"[INFO] 분석 대상 이미지 URL: {IMAGE_URL}")
    print("[INFO] GPT 비전 분석 요청 중...\n")

    result = describe_traffic_event(IMAGE_URL)

    print("=" * 60)
    print("🚨 교통 이벤트 분석 결과")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
