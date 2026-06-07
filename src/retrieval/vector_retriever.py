"""Chroma 벡터 검색 (Dense).

질의를 OpenAI 로 임베딩한 뒤 코사인 유사도 KNN 으로 상위 청크를 반환한다.
메타데이터 필터(where)로 검색 공간을 좁힐 수 있다 (예: {"standard": "MIL-STD-810H"}).
"""
from __future__ import annotations

import logging

from src import config
from src.retrieval.embeddings import embed_query

logger = logging.getLogger(__name__)


def retrieve(query: str, k: int = 5, where: dict | None = None) -> list[dict]:
    logger.info("Dense 검색 시작 | k=%d | query='%s'", k, query[:60])
    coll = config.get_chroma_collection()
    qvec = embed_query(query)
    res = coll.query(
        query_embeddings=[qvec],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    results = [
        {"id": cid, "text": doc, "metadata": meta, "score": 1.0 - dist}
        for cid, doc, meta, dist in zip(ids, docs, metas, dists)
    ]
    logger.info("Dense 검색 완료 | %d건 반환 | 상위 출처: %s",
                len(results),
                [r["metadata"].get("source", "") for r in results[:3]])
    return results


def citation_label(meta: dict) -> str:
    """검색 결과 메타데이터를 사람이 읽는 출처 라벨로."""
    if meta.get("doc_type") == "component":
        return f"부품 {meta.get('part_number','')} ({meta.get('source','')})"
    if meta.get("doc_type") == "graph":
        pn = meta.get("part_number", "")
        std = meta.get("standard", "")
        asm = meta.get("assembly", "")
        base = f"부품 {pn}" if pn else "지식그래프"
        if asm and asm != pn:
            base = f"{base} ⊂ {asm}"  # 하위 부품임을 명시
        return f"{base} · {std} (neo4j)" if std else f"{base} (neo4j)"
    if meta.get("doc_type") == "requirement":
        return f"규격 요구치 {meta.get('requirement_id','')} ({meta.get('source','')})"
    if meta.get("doc_type") == "glossary":
        return "용어집 (ko_en_terms.yaml)"
    std = meta.get("standard", "")
    method = meta.get("method", "")
    label = std + (f" Method {method}" if method else "")
    return f"{label} ({meta.get('source','')})".strip()
