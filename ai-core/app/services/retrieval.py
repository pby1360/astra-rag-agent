"""
app/services/retrieval.py
──────────────────────────
Hybrid RAG 검색 서비스.

1. GraphRetriever  : Neo4j 그래프 DB에서 부품 의존성 체인을 조회한다.
2. VectorRetriever : Redis Stack에 저장된 MIL-SPEC 규격 임베딩을 벡터 유사도로 검색한다.
3. HybridRetriever : 두 결과를 병합하여 단일 컨텍스트 문자열로 반환한다.
"""

from __future__ import annotations

import json
import logging
import struct
from typing import Any

import numpy as np
import redis.asyncio as aioredis
from FlagEmbedding import BGEM3FlagModel
from neo4j import AsyncDriver

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# BGE-M3 모델은 로딩 비용이 크므로 모듈 수준 싱글턴으로 유지한다.
# (실제 배포 시 별도의 임베딩 서버 분리를 권장)
_embedding_model: BGEM3FlagModel | None = None


def _get_embedding_model() -> BGEM3FlagModel:
    """BGE-M3 임베딩 모델을 싱글턴으로 반환한다."""
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        logger.info("BGE-M3 모델 로딩 중: %s", settings.embedding_model_name)
        _embedding_model = BGEM3FlagModel(
            settings.embedding_model_name,
            use_fp16=True,   # FP16으로 메모리 절감 및 추론 속도 향상
        )
        logger.info("BGE-M3 모델 로딩 완료")
    return _embedding_model


def _embed_query(query: str) -> bytes:
    """
    쿼리 문자열을 BGE-M3 dense 벡터로 변환하고,
    Redis 저장 형식인 float32 바이트 배열로 직렬화한다.
    """
    model = _get_embedding_model()
    # encode 결과에서 dense_vecs만 사용 (BGE-M3는 dense/sparse/colbert 지원)
    output = model.encode(
        [query],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vector: np.ndarray = output["dense_vecs"][0]
    # Redis Vector Search는 float32 little-endian 바이트 배열을 요구한다.
    return vector.astype(np.float32).tobytes()


# ── Graph Retriever (Neo4j) ───────────────────────────────────────────────────

class GraphRetriever:
    """
    Neo4j 그래프 DB에서 부품 의존성 정보를 조회하는 클래스.
    Cypher 쿼리로 지정된 부품과 연결된 하위 부품 체인을 최대 3단계까지 탐색한다.
    """

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    async def get_component_dependencies(
        self, component_name: str
    ) -> list[dict[str, Any]]:
        """
        주어진 부품명의 의존성 체인을 Neo4j에서 조회한다.

        반환 형식:
            [{"part": "TI-TM4C123GH6PM", "depends_on": "ARM Cortex-M4", "standard": "MIL-STD-810H"}, ...]
        """
        # APOC 없이 순수 Cypher로 가변 깊이 경로 탐색
        cypher = """
        MATCH path = (c:Component {name: $name})-[:DEPENDS_ON*1..3]->(dep:Component)
        RETURN
            c.name           AS part,
            dep.name         AS depends_on,
            dep.standard     AS standard,
            dep.description  AS description,
            length(path)     AS depth
        ORDER BY depth
        LIMIT 50
        """
        async with self._driver.session() as session:
            result = await session.run(cypher, name=component_name)
            records = await result.data()

        if not records:
            logger.warning("Neo4j: '%s' 부품 의존성 정보 없음", component_name)
        else:
            logger.info(
                "Neo4j: '%s' 의존성 %d건 조회 완료", component_name, len(records)
            )
        return records

    async def get_standard_requirements(self, standard_name: str) -> list[dict[str, Any]]:
        """
        특정 MIL-SPEC 표준에 연결된 요구사항 노드를 조회한다.

        반환 형식:
            [{"requirement_id": "MIL-STD-810H-514.8", "description": "...", "category": "Vibration"}, ...]
        """
        cypher = """
        MATCH (s:Standard {name: $standard})-[:HAS_REQUIREMENT]->(r:Requirement)
        RETURN
            r.id          AS requirement_id,
            r.description AS description,
            r.category    AS category,
            r.threshold   AS threshold
        ORDER BY r.category, r.id
        LIMIT 30
        """
        async with self._driver.session() as session:
            result = await session.run(cypher, standard=standard_name)
            records = await result.data()

        logger.info(
            "Neo4j: '%s' 표준 요구사항 %d건 조회 완료", standard_name, len(records)
        )
        return records


# ── Vector Retriever (Redis Stack) ────────────────────────────────────────────

class VectorRetriever:
    """
    Redis Stack에 저장된 MIL-SPEC 규격 문서를 벡터 유사도로 검색하는 클래스.
    RediSearch(FT.SEARCH) 명령어와 KNN 알고리즘을 사용한다.
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client
        self._settings = get_settings()

    async def search_specs(self, query: str) -> list[dict[str, Any]]:
        """
        쿼리를 BGE-M3로 임베딩한 뒤 Redis KNN 벡터 검색을 수행한다.

        반환 형식:
            [{"doc_id": "mil_doc_001", "content": "...", "score": 0.92, "standard": "MIL-STD-810H"}, ...]
        """
        query_vector = _embed_query(query)
        index_name = self._settings.redis_index_name
        top_k = self._settings.vector_top_k

        # RediSearch KNN 쿼리 구문:
        # *=>[KNN {K} @embedding $query_vec AS score]
        search_query = f"*=>[KNN {top_k} @embedding $query_vec AS score]"

        try:
            raw = await self._redis.execute_command(
                "FT.SEARCH",
                index_name,
                search_query,
                "PARAMS", "2", "query_vec", query_vector,
                "RETURN", "4", "doc_id", "content", "standard", "score",
                "SORTBY", "score",
                "DIALECT", "2",
            )
        except Exception as exc:
            logger.error("Redis 벡터 검색 실패: %s", exc)
            return []

        results = self._parse_redis_results(raw)
        logger.info("Redis 벡터 검색: '%s' → %d건 반환", query[:50], len(results))
        return results

    @staticmethod
    def _parse_redis_results(raw: list[Any]) -> list[dict[str, Any]]:
        """
        FT.SEARCH 응답(리스트 형식)을 딕셔너리 목록으로 변환한다.
        raw[0] = 총 결과 수, raw[1], raw[3], ... = 키, raw[2], raw[4], ... = 필드 목록
        """
        if not raw or len(raw) < 2:
            return []

        results: list[dict[str, Any]] = []
        # FT.SEARCH 결과는 [count, key1, [field, val, ...], key2, ...] 형태
        items = raw[1:]  # 첫 번째 요소(count) 제거
        for i in range(0, len(items), 2):
            if i + 1 >= len(items):
                break
            fields: list[bytes] = items[i + 1]
            doc: dict[str, Any] = {}
            for j in range(0, len(fields), 2):
                if j + 1 >= len(fields):
                    break
                key = fields[j].decode("utf-8") if isinstance(fields[j], bytes) else fields[j]
                val = fields[j + 1].decode("utf-8") if isinstance(fields[j + 1], bytes) else fields[j + 1]
                doc[key] = val
            results.append(doc)
        return results


# ── Hybrid Retriever (통합) ───────────────────────────────────────────────────

class HybridRetriever:
    """
    GraphRetriever와 VectorRetriever를 결합한 Hybrid RAG 검색기.
    두 검색 결과를 병합하여 단일 컨텍스트 문자열로 구조화한다.
    """

    def __init__(self, driver: AsyncDriver, redis_client: aioredis.Redis) -> None:
        self._graph = GraphRetriever(driver)
        self._vector = VectorRetriever(redis_client)

    async def retrieve(
        self,
        query: str,
        component_names: list[str],
        standard_names: list[str],
    ) -> str:
        """
        그래프(부품 의존성) + 벡터(규격 문서) 검색을 병렬로 수행하고
        결과를 구조화된 컨텍스트 문자열로 반환한다.

        Args:
            query:           원본 사용자 질의
            component_names: 분석 대상 부품명 목록 (requirement_analyzer 출력)
            standard_names:  검증 대상 표준명 목록 (requirement_analyzer 출력)

        Returns:
            str: LLM 프롬프트에 삽입될 컨텍스트 문자열
        """
        import asyncio

        # ── 병렬 검색 실행 ─────────────────────────────────────────────────
        graph_tasks = [
            self._graph.get_component_dependencies(c) for c in component_names
        ] + [
            self._graph.get_standard_requirements(s) for s in standard_names
        ]
        vector_task = self._vector.search_specs(query)

        all_tasks = graph_tasks + [vector_task]
        all_results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # 그래프 결과 분리
        num_graph = len(graph_tasks)
        graph_results = all_results[:num_graph]
        vector_results = all_results[num_graph]

        # 예외 처리: 실패한 태스크는 빈 리스트로 대체
        graph_data: list[dict] = []
        for r in graph_results:
            if isinstance(r, Exception):
                logger.error("그래프 검색 태스크 실패: %s", r)
            elif isinstance(r, list):
                graph_data.extend(r)

        vector_data: list[dict] = []
        if isinstance(vector_results, Exception):
            logger.error("벡터 검색 태스크 실패: %s", vector_results)
        elif isinstance(vector_results, list):
            vector_data = vector_results

        # ── 컨텍스트 문자열 조합 ───────────────────────────────────────────
        context_parts: list[str] = []

        if graph_data:
            context_parts.append(
                "=== [그래프 DB: 부품 의존성 및 표준 요구사항] ===\n"
                + json.dumps(graph_data, ensure_ascii=False, indent=2)
            )
        else:
            context_parts.append(
                "=== [그래프 DB] ===\n해당 부품/표준 정보 없음"
            )

        if vector_data:
            context_parts.append(
                "\n=== [벡터 DB: 관련 MIL-SPEC 규격 문서] ===\n"
                + json.dumps(vector_data, ensure_ascii=False, indent=2)
            )
        else:
            context_parts.append(
                "\n=== [벡터 DB] ===\n유사 규격 문서 없음"
            )

        return "\n".join(context_parts)
