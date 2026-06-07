"""중앙 설정 및 클라이언트 싱글톤.

모든 모듈은 경로/모델명/외부 클라이언트를 여기서 가져온다.
환경 변수는 프로젝트 루트의 .env 에서 로드된다 (.env.example 참고).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트 기준 .env 로드
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- 경로 ---
DATA_DIR = ROOT / "data"
STANDARDS_DIR = DATA_DIR / "standards"
COMPONENTS_DIR = DATA_DIR / "components"
GLOSSARY_PATH = DATA_DIR / "glossary" / "ko_en_terms.yaml"
RAW_PDF_DIR = ROOT / "raw_pdfs"
CHROMA_DIR = ROOT / "chroma_db"
CHROMA_COLLECTION = "astra_specs"

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
REPORT_MODEL = os.getenv("OPENAI_REPORT_MODEL", "gpt-4o")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
# LLM 호출 타임아웃(초) — 무한 대기 방지 (generator 에서 예외 시 청크 원문 폴백)
LLM_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "30"))

# --- LlamaParse ---
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "astra-rag-password")


@lru_cache(maxsize=1)
def get_openai_client():
    """OpenAI 클라이언트 (프로세스당 1회 생성)."""
    from openai import OpenAI

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY 가 설정되지 않았습니다. .env 파일을 확인하세요."
        )
    return OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT_SEC, max_retries=2)


@lru_cache(maxsize=1)
def get_neo4j_driver():
    """Neo4j 드라이버 (프로세스당 1회 생성). 사용 후 close()는 호출 측 책임."""
    from neo4j import GraphDatabase

    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


@lru_cache(maxsize=1)
def get_chroma_collection():
    """Chroma persistent 컬렉션 (코사인 유사도, HNSW)."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
