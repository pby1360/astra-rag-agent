"""OpenAI 임베딩 공용 헬퍼 (적재·검색 양쪽에서 사용)."""
from __future__ import annotations

from src import config


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """텍스트 리스트를 배치로 임베딩."""
    client = config.get_openai_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=config.EMBED_MODEL, input=batch)
        vectors.extend(d.embedding for d in resp.data)
    return vectors


def embed_query(text: str) -> list[float]:
    """단일 질의 임베딩."""
    return embed_texts([text])[0]
