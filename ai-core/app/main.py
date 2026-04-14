"""
app/main.py
────────────
FastAPI 애플리케이션 진입점.

- lifespan 이벤트 핸들러로 Redis Stack / Neo4j 연결을 관리한다.
- 전역 예외 핸들러와 CORS 미들웨어를 등록한다.
- /api/v1/verification 라우터를 마운트한다.
"""

from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as verification_router
from app.core.config import get_settings
from app.core.db import lifespan_db

# ── 로깅 설정 ─────────────────────────────────────────────────────────────────
settings = get_settings()

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            }
        },
        "root": {
            "level": settings.log_level.upper(),
            "handlers": ["console"],
        },
    }
)

logger = logging.getLogger(__name__)


# ── FastAPI Lifespan (시작/종료 이벤트) ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI 애플리케이션 수명주기 관리자.

    Startup:  Redis Stack / Neo4j 연결 초기화
    Shutdown: 모든 DB 연결 안전하게 종료
    """
    logger.info("=== MIL-SPEC AI Inference Core 시작 ===")
    logger.info("환경: %s | 로그 레벨: %s", settings.app_env, settings.log_level)

    async with lifespan_db(app):
        yield

    logger.info("=== MIL-SPEC AI Inference Core 종료 완료 ===")


# ── FastAPI 애플리케이션 생성 ─────────────────────────────────────────────────

app = FastAPI(
    title="MIL-SPEC AI Inference Core",
    description=(
        "항공우주/방산 COTS 부품의 MIL-SPEC 적합성을 검증하는 "
        "멀티에이전트 AI 추론 코어 API.\n\n"
        "**주요 기능:**\n"
        "- Neo4j 그래프 DB 기반 부품 의존성 분석\n"
        "- Redis Stack 벡터 검색 기반 MIL-SPEC 규격 검색\n"
        "- LangGraph 오케스트레이션 기반 자동 검증 워크플로우\n"
        "- GPT-4o 기반 한국어 엔지니어링 보고서 생성"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── CORS 미들웨어 ─────────────────────────────────────────────────────────────

# 개발 환경에서는 모든 출처 허용, 프로덕션에서는 허용 도메인을 명시할 것
_allowed_origins = (
    ["*"] if settings.app_env == "development" else ["https://your-domain.com"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 전역 예외 핸들러 ──────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """처리되지 않은 예외를 잡아 일관된 JSON 에러 응답을 반환한다."""
    logger.exception("처리되지 않은 예외 발생: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "서버 내부 오류가 발생했습니다. 관리자에게 문의하세요.",
        },
    )


# ── 라우터 마운트 ─────────────────────────────────────────────────────────────

app.include_router(verification_router)


# ── 루트 엔드포인트 (헬스체크) ────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> dict:
    """루트 경로 헬스체크. 로드밸런서 Health Probe용."""
    return {
        "service": "MIL-SPEC AI Inference Core",
        "version": "1.0.0",
        "status": "running",
        "env": settings.app_env,
    }


@app.get("/health", include_in_schema=False)
async def health() -> dict:
    """상세 헬스체크 엔드포인트."""
    from app.core.db import get_neo4j, get_redis

    db_status: dict = {}

    # Redis 연결 상태 확인
    try:
        redis = get_redis()
        await redis.ping()
        db_status["redis"] = "ok"
    except Exception as exc:
        db_status["redis"] = f"error: {exc!s}"

    # Neo4j 연결 상태 확인
    try:
        neo4j = get_neo4j()
        await neo4j.verify_connectivity()
        db_status["neo4j"] = "ok"
    except Exception as exc:
        db_status["neo4j"] = f"error: {exc!s}"

    all_ok = all(v == "ok" for v in db_status.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "databases": db_status,
    }


# ── 개발 서버 진입점 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )
