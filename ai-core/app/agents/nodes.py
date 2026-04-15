"""
app/agents/nodes.py
────────────────────
LangGraph 워크플로우를 구성하는 4개의 비동기 노드 함수.

노드 실행 순서:
  requirement_analyzer → hybrid_retriever → cross_verifier
                                              ↓            ↓
                                    (데이터 부족)  (충분)
                                              ↓            ↓
                                    hybrid_retriever   report_generator
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.state import VerificationState
from app.core.config import get_settings
from app.core.db import get_neo4j, get_redis
from app.services.retrieval import HybridRetriever

logger = logging.getLogger(__name__)

# 재검색 최대 허용 횟수 (무한 루프 방지)
MAX_RETRY_COUNT = 2


def _get_llm() -> ChatOpenAI:
    """GPT-4o LLM 인스턴스를 반환한다. (매 노드 호출 시 공유 인스턴스 사용)"""
    settings = get_settings()
    return ChatOpenAI(
        model="gpt-4o",
        api_key=settings.openai_api_key,
        temperature=0,          # 결정론적 출력 (규격 검증은 재현성이 중요)
        max_tokens=4096,
    )


# ── Node 1: requirement_analyzer ─────────────────────────────────────────────

async def requirement_analyzer(state: VerificationState) -> VerificationState:
    """
    사용자 질의에서 검증 대상 부품명과 표준명을 구조화된 형태로 추출한다.

    입력  state["user_query"]
    출력  state["extracted_requirements"] = {
              "components": ["TI TM4C123GH6PM", ...],
              "standards":  ["NASA-STD-5001", "MIL-STD-810H", ...],
              "verification_aspects": ["진동", "온도", ...]
          }
    """
    logger.info("[requirement_analyzer] 요구사항 분석 시작")
    llm = _get_llm()

    system_prompt = """당신은 항공우주/방산 분야 MIL-SPEC 요구사항 분석 전문가입니다.
사용자 질의에서 다음 정보를 JSON 형식으로 정확히 추출하세요:
1. components: 검증 대상 부품명 목록 (모델명/제조사 포함)
2. standards: 검증 기준이 되는 표준명 목록 (예: MIL-STD-810H, NASA-STD-5001)
3. verification_aspects: 검증이 필요한 구체적인 항목 목록 (예: 진동, 열 사이클, 방사선 내성)

반드시 유효한 JSON만 출력하세요. 설명 없이 JSON 객체만 반환하세요."""

    user_prompt = f"다음 질의를 분석하세요:\n\n{state['user_query']}"

    response = await llm.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    # LLM 응답에서 JSON 파싱
    try:
        raw_content = response.content.strip()
        # 마크다운 코드 블록이 있을 경우 제거
        if raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1]
            if raw_content.startswith("json"):
                raw_content = raw_content[4:]
        extracted: dict = json.loads(raw_content)
    except (json.JSONDecodeError, IndexError) as exc:
        logger.warning("[requirement_analyzer] JSON 파싱 실패, 기본값 사용: %s", exc)
        # 파싱 실패 시 전체 질의를 검색 키워드로 사용하는 폴백
        extracted = {
            "components": [],
            "standards": [],
            "verification_aspects": [state["user_query"]],
        }

    logger.info("[requirement_analyzer] 추출 결과: %s", extracted)
    return {**state, "extracted_requirements": extracted}


# ── Node 2: hybrid_retriever ──────────────────────────────────────────────────

async def hybrid_retriever(state: VerificationState) -> VerificationState:
    """
    requirement_analyzer가 추출한 부품/표준 정보를 기반으로
    Neo4j(그래프) + Redis(벡터) Hybrid RAG 검색을 수행한다.

    입력  state["extracted_requirements"], state["user_query"]
    출력  state["retrieved_context"] (구조화된 컨텍스트 문자열)
    """
    logger.info("[hybrid_retriever] Hybrid RAG 검색 시작")

    requirements: dict = state.get("extracted_requirements", {})
    component_names: list[str] = requirements.get("components", [])
    standard_names: list[str] = requirements.get("standards", [])

    # 부품/표준이 추출되지 않은 경우 질의 전체를 검색어로 사용
    if not component_names and not standard_names:
        logger.warning(
            "[hybrid_retriever] 추출된 부품/표준 없음. 원본 질의로 벡터 검색 수행."
        )

    retriever = HybridRetriever(
        driver=get_neo4j(),
        redis_client=get_redis(),
    )
    context = await retriever.retrieve(
        query=state["user_query"],
        component_names=component_names,
        standard_names=standard_names,
    )

    logger.info(
        "[hybrid_retriever] 컨텍스트 수집 완료 (길이: %d자)", len(context)
    )
    return {**state, "retrieved_context": context}


# ── Node 3: cross_verifier ────────────────────────────────────────────────────

async def cross_verifier(state: VerificationState) -> VerificationState:
    """
    검색된 컨텍스트를 기반으로 부품이 MIL-SPEC 요건을 충족하는지 논리적으로 비교한다.

    - 데이터가 충분하면 PASS/FAIL 판단과 근거를 state["verification_result"]에 저장
    - 데이터가 불충분하면 needs_more_info=True를 설정하여 재검색을 트리거한다
    - MAX_RETRY_COUNT 초과 시 강제로 INSUFFICIENT 판정하여 루프를 종료한다

    입력  state["user_query"], state["extracted_requirements"], state["retrieved_context"]
    출력  state["verification_result"], state["needs_more_info"], state["retry_count"]
    """
    logger.info("[cross_verifier] 규격 교차 검증 시작")

    current_retry: int = state.get("retry_count", 0)

    # 재검색 횟수 초과 시 강제 종료
    if current_retry >= MAX_RETRY_COUNT:
        logger.warning(
            "[cross_verifier] 재검색 횟수(%d) 초과. INSUFFICIENT로 강제 종료.",
            current_retry,
        )
        return {
            **state,
            "verification_result": "INSUFFICIENT: 최대 재검색 횟수를 초과하였습니다. "
                                   "수집된 데이터만으로 최종 보고서를 생성합니다.",
            "needs_more_info": False,
            "retry_count": current_retry,
        }

    llm = _get_llm()
    requirements = state.get("extracted_requirements", {})
    context = state.get("retrieved_context", "컨텍스트 없음")

    system_prompt = """당신은 항공우주/방산 규격 교차 검증 전문가입니다.
제공된 부품 사양과 표준 요구사항을 비교 분석하세요.

판정 기준:
- PASS: 부품이 모든 표준 요건을 명확히 충족하는 증거가 존재
- FAIL: 부품이 하나 이상의 표준 요건을 충족하지 못하는 증거가 존재
- INSUFFICIENT: 판정에 필요한 데이터가 부족하여 재검색이 필요

응답 형식 (JSON):
{
  "verdict": "PASS | FAIL | INSUFFICIENT",
  "confidence": 0.0~1.0,
  "met_requirements": ["충족된 요건 목록"],
  "unmet_requirements": ["미충족 요건 목록"],
  "missing_data": ["부족한 데이터 목록"],
  "reasoning": "판정 근거 상세 설명"
}

반드시 유효한 JSON만 출력하세요."""

    user_prompt = f"""
검증 대상:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

수집된 컨텍스트:
{context}

원본 질의: {state['user_query']}
"""

    response = await llm.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    try:
        raw_content = response.content.strip()
        if raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1]
            if raw_content.startswith("json"):
                raw_content = raw_content[4:]
        verdict_data: dict = json.loads(raw_content)
    except (json.JSONDecodeError, IndexError) as exc:
        logger.error("[cross_verifier] JSON 파싱 실패: %s", exc)
        verdict_data = {
            "verdict": "INSUFFICIENT",
            "reasoning": "LLM 응답 파싱 오류",
        }

    verdict: str = verdict_data.get("verdict", "INSUFFICIENT")
    needs_more: bool = verdict == "INSUFFICIENT"

    if needs_more:
        logger.info(
            "[cross_verifier] 데이터 부족 감지. 재검색 트리거 (시도 %d/%d)",
            current_retry + 1, MAX_RETRY_COUNT,
        )

    return {
        **state,
        "verification_result": json.dumps(verdict_data, ensure_ascii=False),
        "needs_more_info": needs_more,
        "retry_count": current_retry + 1 if needs_more else current_retry,
    }


# ── Node 4: report_generator ──────────────────────────────────────────────────

async def report_generator(state: VerificationState) -> VerificationState:
    """
    검증 결과를 바탕으로 한국어 엔지니어링 보고서를 생성한다.

    보고서 구성:
    1. 검증 개요 (부품, 표준, 질의)
    2. 수집 근거 (컨텍스트 요약)
    3. 검증 결과 (PASS/FAIL/INSUFFICIENT + 상세 판단)
    4. 권고사항 및 다음 단계

    입력  state 전체 (user_query, extracted_requirements, retrieved_context, verification_result)
    출력  state["final_report"]
    """
    logger.info("[report_generator] 최종 보고서 생성 시작")
    llm = _get_llm()

    # verification_result JSON 파싱 (가능한 경우)
    try:
        verdict_data: dict = json.loads(state.get("verification_result", "{}"))
    except json.JSONDecodeError:
        verdict_data = {"reasoning": state.get("verification_result", "알 수 없음")}

    system_prompt = """당신은 항공우주/방산 분야 기술 문서 작성 전문가입니다.
주어진 검증 결과를 바탕으로 전문적이고 체계적인 한국어 엔지니어링 보고서를 작성하세요.

보고서는 반드시 다음 구조를 따라야 합니다:

# MIL-SPEC 부품 적합성 검증 보고서

## 1. 검증 개요
- 검증 요청 질의
- 검증 대상 부품 목록
- 적용 표준 목록

## 2. 수집 근거 요약
- 그래프 DB 조회 결과 요약
- 벡터 검색 결과 요약

## 3. 검증 결과
- 최종 판정: PASS / FAIL / INSUFFICIENT
- 신뢰도 (%)
- 충족된 요건
- 미충족 요건 (해당 시)
- 판정 근거

## 4. 권고사항
- 추가 시험 필요 항목 (해당 시)
- 대체 부품 제안 (해당 시)
- 다음 단계 조치사항

보고서는 전문적이고 객관적인 어조로 작성하며, 항공우주/방산 도메인 전문 용어를 사용하세요."""

    user_prompt = f"""
원본 질의: {state.get('user_query', '')}

추출된 요구사항:
{json.dumps(state.get('extracted_requirements', {}), ensure_ascii=False, indent=2)}

검증 판정 결과:
{json.dumps(verdict_data, ensure_ascii=False, indent=2)}

수집된 컨텍스트 (요약용):
{state.get('retrieved_context', '컨텍스트 없음')[:3000]}
"""

    response = await llm.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    final_report: str = response.content.strip()
    logger.info(
        "[report_generator] 보고서 생성 완료 (길이: %d자)", len(final_report)
    )
    return {**state, "final_report": final_report}
