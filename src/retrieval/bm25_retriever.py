"""BM25 희소 검색 (Sparse).

파트번호(TPS7H1101-SP)나 식별자처럼 의미매칭이 아닌 정확매칭이 필요한 질의를
보완한다. 로더와 동일한 청크 코퍼스 위에 인메모리 인덱스를 구축한다 (1회 캐시).
"""
from __future__ import annotations

import logging
import re

from rank_bm25 import BM25Okapi

from src.ingestion.loader import load_all

logger = logging.getLogger(__name__)

# 라틴/숫자(하이픈 포함, 예: tps7h1101-sp) + 한글 연속 토큰
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*|[가-힣]+")

_bm25: BM25Okapi | None = None
_chunks: list[dict] | None = None


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _ensure_index() -> None:
    global _bm25, _chunks
    if _bm25 is None:
        logger.info("BM25 인덱스 구축 중...")
        _chunks = load_all()
        _bm25 = BM25Okapi([_tokenize(c["text"]) for c in _chunks])
        logger.info("BM25 인덱스 완료 | %d개 청크", len(_chunks))


def retrieve(query: str, k: int = 5) -> list[dict]:
    _ensure_index()
    assert _bm25 is not None and _chunks is not None
    logger.info("BM25 검색 시작 | k=%d | query='%s'", k, query[:60])
    scores = _bm25.get_scores(_tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    results = [
        {
            "id": _chunks[i]["id"],
            "text": _chunks[i]["text"],
            "metadata": _chunks[i]["metadata"],
            "score": float(scores[i]),
        }
        for i in top
        if scores[i] > 0
    ]
    logger.info("BM25 검색 완료 | %d건 반환 | 상위 출처: %s",
                len(results),
                [r["metadata"].get("source", "") for r in results[:3]])
    return results
