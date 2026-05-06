"""
문제 8. 인터넷 검색 후 답변하는 챗봇 (OpenAI Responses API + web_search_preview)

- 질문(Q): "BTS 최신곡에 대한 반응"
- 사용 API: OpenAI Responses API
- 사용 도구: web_search_preview  (모델이 명시적으로 웹 검색 도구를 호출 → 출처 인용 보장)

이전에 시도한 두 가지 방식의 한계:
  · DuckDuckGo: 인덱스 신선도가 낮아 '오늘 기준 최신' 콘텐츠를 잘 못 가져옴.
  · gpt-4o-mini-search-preview (chat completions): 모델이 검색을 비결정적으로
    건너뛰고 자체 지식으로 답해 환각이 섞임.

Responses API + web_search_preview 는 도구 호출이 명시적으로 일어나므로
검색이 안정적으로 발생하고, output_text 안에 [출처](url) 인용이 함께 포함된다.

비용: 검색 1회당 약간의 추가 과금 + 일반 토큰 비용.

사전 설치: pip install openai python-dotenv
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def ask_with_web_search(question: str) -> dict:
    """Responses API + web_search_preview 로 검색 기반 답변 생성."""
    instruction = (
        "너는 인터넷을 직접 검색해서 답변하는 한국어 AI 챗봇이야. "
        "최신성이 중요한 질문이므로 반드시 웹 검색을 사용해서, "
        "최근 기사에 기반한 한국어 답변을 정리해줘. "
        "각 사실 진술 옆에 출처를 인용해줘."
    )

    response = client.responses.create(
        model="gpt-4o-mini",
        tools=[
            {
                "type": "web_search_preview",
                "search_context_size": "high",
                "user_location": {
                    "type": "approximate",
                    "country": "KR",
                },
            }
        ],
        input=f"{instruction}\n\n질문: {question}",
    )

    # 1) 본문 답변
    answer = response.output_text

    # 2) 실제로 web_search_call 이 발생했는지 확인
    search_called = any(
        getattr(item, "type", None) == "web_search_call" for item in response.output
    )

    # 3) message 항목의 url_citation annotations 수집
    citations = []
    for item in response.output:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", []) or []:
            for ann in getattr(part, "annotations", []) or []:
                if getattr(ann, "type", None) == "url_citation":
                    citations.append(
                        {
                            "title": getattr(ann, "title", None),
                            "url": getattr(ann, "url", None),
                        }
                    )

    return {
        "answer": answer,
        "search_called": search_called,
        "citations": citations,
    }


def main():
    question = "BTS 최신곡에 대한 반응"
    print(f"Q: {question}")
    print("[INFO] OpenAI Responses API (web_search_preview) 호출 중...\n")

    result = ask_with_web_search(question)

    print("=" * 60)
    print(f"[검색 도구 호출 여부] {'✅ 검색 수행됨' if result['search_called'] else '❌ 검색 미수행'}")
    print("=" * 60)
    print("A:")
    print(result["answer"])
    print("=" * 60)

    if result["citations"]:
        print("\n[참고 출처]")
        # 중복 URL 제거
        seen = set()
        for c in result["citations"]:
            if c["url"] in seen:
                continue
            seen.add(c["url"])
            print(f"  - {c['title']}\n    {c['url']}")
    else:
        print("\n[참고 출처 없음]")


if __name__ == "__main__":
    main()
