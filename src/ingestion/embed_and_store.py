"""청크를 OpenAI 임베딩 후 Chroma 에 적재.

재실행 시 컬렉션을 초기화(삭제 후 재생성)하여 중복을 방지한다.

실행:  python -m src.ingestion.embed_and_store
"""
from __future__ import annotations

import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.ingestion.loader import load_all  # noqa: E402
from src.retrieval.embeddings import embed_texts  # noqa: E402
from src.utils import enable_utf8_stdout  # noqa: E402


def reset_collection():
    """Chroma 컬렉션을 비우고 새로 만든다 (코사인 공간)."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass
    return client.create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def main() -> None:
    enable_utf8_stdout()
    print("1) 청크 로딩...")
    chunks = load_all()
    print(f"   {len(chunks)}개 청크")

    print("2) OpenAI 임베딩 생성...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    print(f"   {len(embeddings)}개 벡터 (차원 {len(embeddings[0])})")

    print("3) Chroma 적재...")
    coll = reset_collection()
    add_batch = 500
    for i in range(0, len(chunks), add_batch):
        sl = slice(i, i + add_batch)
        coll.add(
            ids=[c["id"] for c in chunks[sl]],
            documents=[c["text"] for c in chunks[sl]],
            embeddings=embeddings[sl],
            metadatas=[c["metadata"] for c in chunks[sl]],
        )
    print(f"   적재 완료. 컬렉션 '{config.CHROMA_COLLECTION}' 총 {coll.count()}개")


if __name__ == "__main__":
    main()
