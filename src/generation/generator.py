"""검색된 문서를 근거로 한국어 답변 생성 (출처 인용)."""
from __future__ import annotations

import logging

from src import config
from src.generation import prompts
from src.retrieval.vector_retriever import citation_label

logger = logging.getLogger(__name__)


def format_context(docs: list[dict]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        label = citation_label(d["metadata"])
        blocks.append(f"[{i}] 출처: {label}\n{d['text']}")
    return "\n\n".join(blocks)


def generate_answer(question: str, docs: list[dict]) -> dict:
    """docs(검색 결과)를 근거로 답변 생성. {'answer','sources'} 반환."""
    if not docs:
        logger.warning("참조 문서 없음 — 답변 생성 불가")
        return {"answer": "정보 없음: 관련 참조 문서를 찾지 못했습니다.", "sources": []}

    logger.info("답변 생성 시작 | 참조 문서 %d개 | 모델=%s", len(docs), config.CHAT_MODEL)
    context = format_context(docs)
    client = config.get_openai_client()
    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {"role": "user", "content": prompts.ANSWER_USER.format(
                context=context, question=question)},
        ],
        temperature=0,
    )
    answer = resp.choices[0].message.content.strip()
    logger.info("답변 생성 완료 | %d자", len(answer))
    return {
        "answer": answer,
        "sources": [
            {
                "label": citation_label(d["metadata"]),
                "snippet": d["text"][:240],
                **d["metadata"],
                "score": d["score"],
            }
            for d in docs
        ],
    }


def is_sufficient(question: str, docs: list[dict]) -> bool:
    """수집 문서가 판정에 충분한지 LLM 으로 평가 (LangGraph self-correction 에서 사용)."""
    if not docs:
        return False
    logger.info("충족성 검증 시작 | 문서 %d개", len(docs))
    client = config.get_openai_client()
    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": prompts.VERIFY_SYSTEM},
            {"role": "user", "content": prompts.VERIFY_USER.format(
                question=question, context=format_context(docs))},
        ],
        temperature=0,
        max_tokens=5,
    )
    result = "INSUFFICIENT" not in resp.choices[0].message.content.upper()
    logger.info("충족성 검증 결과 | %s", "SUFFICIENT" if result else "INSUFFICIENT")
    return result
