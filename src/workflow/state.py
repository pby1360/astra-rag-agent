"""LangGraph 공유 상태 객체.

에이전트들이 진행 상황을 실시간 공유하는 전역 상태. needs_more_info 판단(is_sufficient)과
무한루프 방지(retry_count)를 캡슐화한다.
"""
from __future__ import annotations

from typing import Optional, TypedDict


class RAGState(TypedDict, total=False):
    question: str               # 사용자 질의
    expanded_query: str         # 한영 확장된 질의
    known_components: list      # 질의에서 인식된 등록 부품 번호
    category: Optional[str]     # 판정 카테고리 (vibration/radiation/corrosion)
    query_type: str             # 결정론적 라우팅 결과: "A"(기준조회) / "B"(판정) / "C"(설명) / "D"(절차)
    docs: list                  # 누적 검색 문서 (그래프 + 하이브리드)
    is_sufficient: bool         # verify 결과: 판정에 충분한가
    retry_count: int            # retrieve 진입 횟수 (무한루프 방지)
    answer: Optional[str]       # 최종 한국어 리포트
    sources: list               # 출처 메타데이터
    trace: list                 # 노드 실행 로그 (시연/UI 용)
