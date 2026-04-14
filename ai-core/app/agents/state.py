"""
app/agents/state.py
────────────────────
LangGraph 워크플로우의 공유 상태(State) 정의.

TypedDict를 사용하여 각 노드 간에 전달되는 데이터 구조를 명확히 정의한다.
모든 필드는 선택적(Optional) 또는 기본값을 가지므로,
그래프 초기 호출 시 user_query만 제공해도 동작한다.
"""

from __future__ import annotations

from typing import TypedDict


class VerificationState(TypedDict, total=False):
    """
    MIL-SPEC 검증 워크플로우의 전체 상태를 나타내는 TypedDict.

    total=False 로 선언하여 모든 키를 선택적으로 만든다.
    각 노드는 자신이 담당하는 필드만 갱신하고 나머지는 유지한다.

    Fields:
        user_query:             사용자가 입력한 원본 검증 질의
        extracted_requirements: requirement_analyzer 노드가 추출한 구조화 정보
                                예: {"components": ["TI MCU"], "standards": ["NASA-STD-5001"]}
        retrieved_context:      hybrid_retriever 노드가 조합한 그래프+벡터 컨텍스트 문자열
        verification_result:    cross_verifier 노드의 규격 비교 판단 결과 (PASS/FAIL/INSUFFICIENT)
        needs_more_info:        cross_verifier가 데이터 부족으로 재검색이 필요하다고 판단한 경우 True
        retry_count:            재검색 횟수 (무한 루프 방지용 가드)
        final_report:           report_generator 노드가 생성한 최종 한국어 엔지니어링 보고서
    """

    # ── 입력 ──────────────────────────────────────────────────────────────────
    user_query: str

    # ── requirement_analyzer 노드 출력 ────────────────────────────────────────
    extracted_requirements: dict

    # ── hybrid_retriever 노드 출력 ────────────────────────────────────────────
    retrieved_context: str

    # ── cross_verifier 노드 출력 ──────────────────────────────────────────────
    verification_result: str
    needs_more_info: bool
    retry_count: int

    # ── report_generator 노드 출력 ────────────────────────────────────────────
    final_report: str
