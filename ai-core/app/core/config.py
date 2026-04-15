"""
app/core/config.py
──────────────────
애플리케이션 전역 설정 모듈.
pydantic-settings 를 사용해 .env 파일 및 환경변수를 타입 안전하게 로드한다.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 전체 설정값을 담는 Pydantic v2 모델."""

    model_config = SettingsConfigDict(
        env_file=".env",          # .env 파일에서 값 로드
        env_file_encoding="utf-8",
        case_sensitive=False,     # 환경변수 대소문자 무시
        extra="ignore",           # 정의되지 않은 환경변수 무시
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="GPT-4o 호출용 OpenAI API 키")

    # ── Redis Stack (벡터 DB) ─────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis Stack 연결 URL",
    )
    redis_index_name: str = Field(
        default="mil_spec_index",
        description="MIL-SPEC 규격 벡터 인덱스 이름",
    )
    vector_top_k: int = Field(
        default=5,
        description="벡터 유사도 검색 시 반환할 최대 결과 수",
    )

    # ── Neo4j (그래프 DB) ─────────────────────────────────────────────────────
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j 연결 URI",
    )
    neo4j_user: str = Field(default="neo4j", description="Neo4j 사용자명")
    neo4j_password: str = Field(..., description="Neo4j 비밀번호")

    # ── 임베딩 모델 ───────────────────────────────────────────────────────────
    embedding_model_name: str = Field(
        default="BAAI/bge-m3",
        description="BGE-M3 임베딩 모델 이름 또는 로컬 경로",
    )

    # ── 애플리케이션 ──────────────────────────────────────────────────────────
    app_env: str = Field(default="development", description="실행 환경")
    log_level: str = Field(default="INFO", description="로그 레벨")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Settings 인스턴스를 싱글턴으로 반환한다.
    lru_cache 덕분에 앱 수명 동안 단 한 번만 파싱된다.
    """
    return Settings()  # type: ignore[call-arg]
