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


def _fallback_answer(docs: list[dict]) -> str:
    """LLM 호출 실패 시, 검색된 청크 원문을 출처와 함께 그대로 제시한다.

    답변 생성은 못 하더라도 '무엇이 참조되었는지'(출처)와 근거 원문은 보존되어
    질의/응답 루프가 에러 없이 종료되도록 한다(시스템 무중단 보장).
    """
    head = "⚠️ 답변 자동 생성에 실패하여 검색된 근거 원문을 그대로 제시합니다.\n"
    blocks = []
    for i, d in enumerate(docs[:5], start=1):
        label = citation_label(d["metadata"])
        blocks.append(f"[{i}] 출처: {label}\n{d['text'][:500]}")
    return head + "\n\n".join(blocks)


def generate_answer(question: str, docs: list[dict], query_type: str = "C") -> dict:
    """docs(검색 결과)를 근거로 답변 생성. {'answer','sources'} 반환.

    query_type: 라우터(nodes.detect_query_type)가 확정한 유형(A/B/C/D).
                해당 유형의 프롬프트만 LLM 에 전달해 형식 혼동을 차단한다.
    """
    if not docs:
        logger.warning("참조 문서 없음 — 답변 생성 불가")
        return {"answer": "정보 없음: 관련 참조 문서를 찾지 못했습니다.", "sources": []}

    user_template = prompts.ANSWER_USER_BY_TYPE.get(query_type, prompts.ANSWER_USER_C)
    logger.info("답변 생성 시작 | 유형=%s | 참조 문서 %d개 | 모델=%s",
                query_type, len(docs), config.CHAT_MODEL)
    context = format_context(docs)
    client = config.get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": prompts.ANSWER_SYSTEM},
                {"role": "user", "content": user_template.format(
                    context=context, question=question)},
            ],
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip()
        logger.info("답변 생성 완료 | %d자", len(answer))
    except Exception as exc:  # 타임아웃/네트워크 등 — 시스템 중단 없이 청크 원문 폴백
        logger.error("LLM 답변 생성 실패(%s) — 검색 청크 원문으로 폴백", exc)
        answer = _fallback_answer(docs)
    return {
        "answer": answer,
        "sources": [
            {
                # cite_index: 답변 본문의 [N] 인용 번호와 1:1 대응되는 1-base 인덱스.
                # format_context 가 enumerate(docs, start=1) 로 매긴 번호와 동일하므로
                # UI 가 이 값을 표시·정렬하면 본문 인용과 시각적으로 일치한다.
                "cite_index": i,
                "label": citation_label(d["metadata"]),
                "snippet": d["text"][:240],
                **d["metadata"],
                "score": d["score"],
            }
            for i, d in enumerate(docs, start=1)
        ],
    }


def is_sufficient(question: str, docs: list[dict], query_type: str = "C") -> bool:
    """수집 문서가 답변에 충분한지 LLM 으로 평가. 유형별 충족 기준이 다르므로
    질의 유형을 함께 전달한다 (예: D는 절차 설명 문서가 있으면 충분)."""
    if not docs:
        return False
    logger.info("충족성 검증 시작 | 유형=%s | 문서 %d개", query_type, len(docs))
    client = config.get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": prompts.VERIFY_SYSTEM},
                {"role": "user", "content": prompts.VERIFY_USER.format(
                    query_type=query_type,
                    question=question,
                    context=format_context(docs))},
            ],
            temperature=0,
            max_tokens=5,
        )
        result = "INSUFFICIENT" not in resp.choices[0].message.content.upper()
    except Exception as exc:  # 검증 실패 시 무한 재검색 방지: 보유 문서로 진행
        logger.error("충족성 검증 실패(%s) — 보유 문서로 리포트 진행", exc)
        result = True
    logger.info("충족성 검증 결과 | %s", "SUFFICIENT" if result else "INSUFFICIENT")
    return result
