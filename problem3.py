"""
문제 3. PDF 문서를 요약하는 AI 연구원

- 대상 PDF: '농업 현장데이터의 디지털 전환을 위한 영농일지 표준화와 자동화 전략.pdf'
- 흐름:
    1) pypdf 로 PDF 의 모든 페이지 텍스트를 추출해 하나의 문자열로 만든다.
    2) system 으로 "PDF 요약 AI 연구원" 역할을 부여한다.
    3) user 메시지로 추출한 PDF 텍스트를 통째로 보낸다.
    4) 모델 응답(요약)을 출력한다.

사전 설치: pip install openai python-dotenv pypdf
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

PDF_PATH = "농업 현장데이터의 디지털 전환을 위한 영농일지 표준화와 자동화 전략.pdf"


def extract_pdf_text(path: str) -> str:
    """PDF 파일의 모든 페이지에서 텍스트를 뽑아 하나의 문자열로 반환."""
    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def summarize_pdf(pdf_text: str) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": (
                "너는 PDF 문서를 분석하고 요약하는 AI 연구원이야. "
                "사용자가 보낸 문서의 핵심 내용을 정확하고 구조적으로 요약해줘. "
                "다음 형식을 따라줘:\n"
                "1) 문서 주제 한 줄 요약\n"
                "2) 핵심 키워드 5개\n"
                "3) 주요 내용 정리(번호 매긴 목록)\n"
                "4) 결론 및 시사점"
            ),
        },
        {
            "role": "user",
            "content": f"다음 PDF 문서를 요약해줘.\n\n---PDF 본문 시작---\n{pdf_text}\n---PDF 본문 끝---",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return response.choices[0].message.content


def main():
    print(f"[INFO] PDF 로드 중: {PDF_PATH}")
    pdf_text = extract_pdf_text(PDF_PATH)
    print(f"[INFO] 추출 텍스트 길이: {len(pdf_text)}자")

    print("[INFO] OpenAI 요약 요청 중...\n")
    summary = summarize_pdf(pdf_text)

    print("=" * 60)
    print("📄 요약 결과")
    print("=" * 60)
    print(summary)


if __name__ == "__main__":
    main()
