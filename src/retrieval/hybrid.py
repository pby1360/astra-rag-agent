"""하이브리드 검색 = Dense(Chroma) + Sparse(BM25), RRF 결합 + 한영 쿼리 확장.

- 한영 확장: 용어집으로 국문/은어 질의에 영문 규격 용어를 덧붙여 영문 원천 문서와 연결.
- RRF(Reciprocal Rank Fusion): 점수 스케일이 다른 두 검색을 순위 기반으로 안정 결합.
"""
from __future__ import annotations

import logging

import yaml

from src import config
from src.retrieval import bm25_retriever, vector_retriever

logger = logging.getLogger(__name__)

_glossary_terms: list[dict] | None = None


def _load_glossary_terms() -> list[dict]:
    global _glossary_terms
    if _glossary_terms is None:
        if config.GLOSSARY_PATH.exists():
            data = yaml.safe_load(config.GLOSSARY_PATH.read_text(encoding="utf-8")) or {}
            _glossary_terms = data.get("terms", [])
        else:
            _glossary_terms = []
    return _glossary_terms


def expand_query_ko_en(query: str) -> str:
    """질의에 국문 용어가 있으면 대응 영문 용어를 덧붙인다.

    예: "소금물 테스트 코팅" -> "소금물 테스트 코팅 Salt Fog Salt Spray"
    """
    additions: list[str] = []
    for t in _load_glossary_terms():
        ko_forms = [t.get("ko", "")] + (t.get("aliases") or [])
        if any(ko and ko in query for ko in ko_forms):
            additions += [t.get("en", "")] + (t.get("en_aliases") or [])
    extra = " ".join(dict.fromkeys(w for w in additions if w))  # 순서 유지 중복 제거
    expanded = f"{query} {extra}".strip() if extra else query
    if extra:
        logger.info("쿼리 한영 확장 | 추가된 용어: %s", extra)
    return expanded


def _rrf(result_lists: list[list[dict]], rrf_k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    store: dict[str, dict] = {}
    for results in result_lists:
        for rank, d in enumerate(results):
            scores[d["id"]] = scores.get(d["id"], 0.0) + 1.0 / (rrf_k + rank + 1)
            store[d["id"]] = d
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    out = []
    for cid, s in ranked:
        d = dict(store[cid])
        d["score"] = s
        out.append(d)
    return out


def hybrid_retrieve(query: str, k: int = 5, pool: int = 10) -> list[dict]:
    logger.info("하이브리드 검색 시작 | k=%d | query='%s'", k, query[:60])
    expanded = expand_query_ko_en(query)
    dense = vector_retriever.retrieve(expanded, k=pool)
    sparse = bm25_retriever.retrieve(expanded, k=pool)
    fused = _rrf([dense, sparse])
    results = fused[:k]
    logger.info("하이브리드 검색 완료 | RRF 결합 후 %d건 | 상위 출처: %s",
                len(results),
                [r["metadata"].get("source", "") for r in results[:3]])
    return results
