"""
app/api/routes.py
──────────────────
FastAPI 라우터: MIL-SPEC 부품 적합성 검증 엔드포인트.

POST /api/v1/verification/analyze
  - 요청: 사용자 자연어 질의
  - 처리: LangGraph 워크플로우 비동기 실행
  - 응답: 한국어 엔지니어링 보고서
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.agents.graph import verification_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/verification", tags=["MIL-SPEC 검증"])


# ── 요청/응답 스키마 (Pydantic v2) ────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """검증 요청 바디 스키마."""

    query: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="검증 대상 부품과 표준을 포함한 자연어 질의",
        examples=["TI MCU(TM4C123GH6PM)가 NASA-STD-5001 진동 규격을 만족하는가?"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "TI MCU(TM4C123GH6PM)가 NASA-STD-5001 진동 규격을 만족하는가?"
                }
            ]
        }
    }


class AnalysisResponse(BaseModel):
    """검증 응답 바디 스키마."""

    success: bool = Field(description="요청 처리 성공 여부")
    query: str = Field(description="원본 사용자 질의")
    final_report: str = Field(description="한국어 MIL-SPEC 검증 엔지니어링 보고서")
    verdict: str = Field(description="최종 판정 (PASS / FAIL / INSUFFICIENT)")
    elapsed_seconds: float = Field(description="총 처리 시간 (초)")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="추출된 요구사항 등 부가 정보",
    )


class ErrorResponse(BaseModel):
    """에러 응답 바디 스키마."""

    success: bool = False
    error_code: str
    message: str


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="MIL-SPEC 부품 적합성 검증",
    description=(
        "자연어 질의를 입력받아 LangGraph 멀티에이전트 워크플로우를 실행하고 "
        "한국어 엔지니어링 검증 보고서를 반환한다.\n\n"
        "내부적으로 Neo4j(부품 의존성 그래프)와 Redis Stack(MIL-SPEC 벡터 검색)을 "
        "결합한 Hybrid RAG를 사용한다."
    ),
    responses={
        200: {"description": "검증 보고서 반환 성공"},
        422: {"description": "요청 바디 유효성 검사 실패"},
        500: {"description": "워크플로우 실행 중 서버 오류"},
    },
)
async def analyze_verification(request: AnalysisRequest) -> AnalysisResponse:
    """
    MIL-SPEC 부품 적합성 검증 요청을 처리한다.

    1. LangGraph `verification_graph`를 비동기로 호출한다.
    2. 그래프 실행이 완료되면 `final_report`를 추출한다.
    3. 처리 시간, 판정 결과, 메타데이터와 함께 응답을 반환한다.
    """
    start_time = time.monotonic()
    logger.info("검증 요청 수신: %s", request.query[:100])

    try:
        # LangGraph 비동기 실행 (ainvoke는 전체 그래프를 실행하고 최종 상태를 반환)
        initial_state = {
            "user_query": request.query,
            "retry_count": 0,
            "needs_more_info": False,
        }
        final_state = await verification_graph.ainvoke(initial_state)

    except Exception as exc:
        logger.exception("LangGraph 워크플로우 실행 실패: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"워크플로우 실행 중 오류가 발생했습니다: {exc!s}",
        ) from exc

    # 최종 보고서 추출
    final_report: str = final_state.get(
        "final_report",
        "보고서 생성에 실패했습니다. 관리자에게 문의하세요.",
    )

    # verification_result JSON에서 판정 추출
    import json
    verdict = "UNKNOWN"
    try:
        vr_raw: str = final_state.get("verification_result", "{}")
        verdict_data: dict = json.loads(vr_raw)
        verdict = verdict_data.get("verdict", "UNKNOWN")
    except (json.JSONDecodeError, AttributeError):
        verdict = final_state.get("verification_result", "UNKNOWN")

    elapsed = round(time.monotonic() - start_time, 2)
    logger.info("검증 완료: 판정=%s, 처리시간=%.2f초", verdict, elapsed)

    return AnalysisResponse(
        success=True,
        query=request.query,
        final_report=final_report,
        verdict=verdict,
        elapsed_seconds=elapsed,
        metadata={
            "extracted_requirements": final_state.get("extracted_requirements", {}),
            "retry_count": final_state.get("retry_count", 0),
        },
    )


# ── 헬스체크 엔드포인트 ───────────────────────────────────────────────────────

@router.get(
    "/health",
    summary="라우터 헬스체크",
    description="검증 라우터가 정상 동작 중인지 확인한다.",
    include_in_schema=False,
)
async def health_check() -> dict[str, str]:
    """검증 라우터 헬스체크."""
    return {"status": "ok", "router": "verification"}
