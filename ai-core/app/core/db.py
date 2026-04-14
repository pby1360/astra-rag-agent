"""
app/core/db.py
──────────────
Redis Stack(벡터 DB) 및 Neo4j(그래프 DB) 비동기 연결 관리자.

FastAPI lifespan 이벤트에서 연결을 초기화하고,
애플리케이션 종료 시 안전하게 정리(graceful shutdown)한다.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from neo4j import AsyncDriver, AsyncGraphDatabase

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ── 전역 연결 객체 (앱 수명 동안 단일 인스턴스 유지) ─────────────────────────
_redis_client: aioredis.Redis | None = None
_neo4j_driver: AsyncDriver | None = None


# ── Redis Stack ───────────────────────────────────────────────────────────────

async def init_redis() -> aioredis.Redis:
    """
    Redis Stack 비동기 클라이언트를 초기화하고 연결을 검증한다.
    이미 초기화된 경우 기존 클라이언트를 반환한다.
    """
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    settings = get_settings()
    _redis_client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False,   # 벡터 바이너리 데이터 처리를 위해 False 유지
        max_connections=20,       # 커넥션 풀 최대 크기
    )
    # 연결 유효성 확인
    await _redis_client.ping()
    logger.info("Redis Stack 연결 성공: %s", settings.redis_url)
    return _redis_client


async def close_redis() -> None:
    """Redis 클라이언트 연결을 안전하게 종료한다."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis Stack 연결 종료 완료")


def get_redis() -> aioredis.Redis:
    """
    초기화된 Redis 클라이언트를 반환한다.
    FastAPI 의존성 주입(Depends) 또는 직접 호출로 사용한다.
    """
    if _redis_client is None:
        raise RuntimeError(
            "Redis 클라이언트가 초기화되지 않았습니다. "
            "FastAPI lifespan 이벤트를 확인하세요."
        )
    return _redis_client


# ── Neo4j (그래프 DB) ─────────────────────────────────────────────────────────

async def init_neo4j() -> AsyncDriver:
    """
    Neo4j 비동기 드라이버를 초기화하고 연결 가능 여부를 검증한다.
    이미 초기화된 경우 기존 드라이버를 반환한다.
    """
    global _neo4j_driver
    if _neo4j_driver is not None:
        return _neo4j_driver

    settings = get_settings()
    _neo4j_driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_pool_size=20,
    )
    # 연결 유효성 확인
    await _neo4j_driver.verify_connectivity()
    logger.info("Neo4j 연결 성공: %s", settings.neo4j_uri)
    return _neo4j_driver


async def close_neo4j() -> None:
    """Neo4j 드라이버 연결을 안전하게 종료한다."""
    global _neo4j_driver
    if _neo4j_driver is not None:
        await _neo4j_driver.close()
        _neo4j_driver = None
        logger.info("Neo4j 연결 종료 완료")


def get_neo4j() -> AsyncDriver:
    """
    초기화된 Neo4j 드라이버를 반환한다.
    FastAPI 의존성 주입(Depends) 또는 직접 호출로 사용한다.
    """
    if _neo4j_driver is None:
        raise RuntimeError(
            "Neo4j 드라이버가 초기화되지 않았습니다. "
            "FastAPI lifespan 이벤트를 확인하세요."
        )
    return _neo4j_driver


# ── FastAPI Lifespan 컨텍스트 매니저 ─────────────────────────────────────────

@asynccontextmanager
async def lifespan_db(app) -> AsyncGenerator[None, None]:  # noqa: ANN001
    """
    FastAPI lifespan 핸들러에서 사용하는 DB 연결 컨텍스트 매니저.
    애플리케이션 시작 시 두 DB를 초기화하고,
    종료 시 안전하게 연결을 닫는다.

    사용 예시 (main.py):
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with lifespan_db(app):
                yield
    """
    # ── Startup ──
    logger.info("데이터베이스 연결 초기화 시작...")
    await init_redis()
    await init_neo4j()
    logger.info("모든 데이터베이스 연결 초기화 완료")

    yield  # 애플리케이션 실행 중

    # ── Shutdown ──
    logger.info("데이터베이스 연결 종료 시작...")
    await close_redis()
    await close_neo4j()
    logger.info("모든 데이터베이스 연결 종료 완료")
