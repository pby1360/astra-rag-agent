"""
app/agents/graph.py
────────────────────
LangGraph StateGraph 구성 및 컴파일.

워크플로우 토폴로지:
  [START]
     ↓
  requirement_analyzer   ← 요구사항 추출
     ↓
  hybrid_retriever       ← 그래프+벡터 Hybrid RAG 검색
     ↓
  cross_verifier         ← 규격 교차 검증
     ↓ (조건부 엣지)
  ┌──────────────────────────┐
  │ needs_more_info == True  │ → hybrid_retriever (재검색)
  │ needs_more_info == False │ → report_generator
  └──────────────────────────┘
     ↓
  report_generator       ← 한국어 보고서 생성
     ↓
  [END]
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import (
    cross_verifier,
    hybrid_retriever,
    report_generator,
    requirement_analyzer,
)
from app.agents.state import VerificationState

logger = logging.getLogger(__name__)


# ── 조건부 라우팅 함수 ────────────────────────────────────────────────────────

def route_after_verification(
    state: VerificationState,
) -> Literal["hybrid_retriever", "report_generator"]:
    """
    cross_verifier 실행 후 다음 노드를 결정하는 라우팅 함수.

    - needs_more_info == True  → hybrid_retriever (재검색)
    - needs_more_info == False → report_generator (보고서 생성)
    """
    if state.get("needs_more_info", False):
        logger.info(
            "[라우터] 데이터 부족 감지 → hybrid_retriever로 재검색 (시도 %d)",
            state.get("retry_count", 0),
        )
        return "hybrid_retriever"

    logger.info("[라우터] 검증 완료 → report_generator로 이동")
    return "report_generator"


# ── 그래프 빌드 ───────────────────────────────────────────────────────────────

def build_verification_graph() -> StateGraph:
    """
    MIL-SPEC 검증 워크플로우 StateGraph를 구성하고 반환한다.
    컴파일 전 그래프 객체를 반환하므로, 필요 시 추가 수정이 가능하다.
    """
    graph = StateGraph(VerificationState)

    # ── 노드 등록 ─────────────────────────────────────────────────────────────
    graph.add_node("requirement_analyzer", requirement_analyzer)
    graph.add_node("hybrid_retriever", hybrid_retriever)
    graph.add_node("cross_verifier", cross_verifier)
    graph.add_node("report_generator", report_generator)

    # ── 엣지 연결 (고정 경로) ─────────────────────────────────────────────────
    graph.add_edge(START, "requirement_analyzer")
    graph.add_edge("requirement_analyzer", "hybrid_retriever")
    graph.add_edge("hybrid_retriever", "cross_verifier")

    # ── 조건부 엣지 (cross_verifier → 재검색 또는 보고서 생성) ───────────────
    graph.add_conditional_edges(
        source="cross_verifier",
        path=route_after_verification,
        path_map={
            "hybrid_retriever": "hybrid_retriever",
            "report_generator": "report_generator",
        },
    )

    # ── 종료 엣지 ─────────────────────────────────────────────────────────────
    graph.add_edge("report_generator", END)

    return graph


# ── 컴파일된 그래프 싱글턴 ───────────────────────────────────────────────────

# 애플리케이션 시작 시 단 한 번 컴파일하여 재사용한다.
# LangGraph CompiledGraph는 thread-safe하므로 전역 인스턴스를 공유해도 안전하다.
verification_graph = build_verification_graph().compile()

logger.info("MIL-SPEC 검증 LangGraph 컴파일 완료")
