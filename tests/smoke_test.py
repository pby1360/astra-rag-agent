"""Phase 1 검증 스크립트.

OpenAI 챗 / OpenAI 임베딩 / Neo4j 연결이 각각 1회씩 성공하는지 확인한다.
이게 통과해야 이후 모든 단계가 동작한다.

실행:  python tests/smoke_test.py
"""
import sys
from pathlib import Path

# src 패키지 import 를 위해 루트를 path 에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.utils import enable_utf8_stdout  # noqa: E402

enable_utf8_stdout()


def check_openai_chat() -> None:
    client = config.get_openai_client()
    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[{"role": "user", "content": "한 단어로만 답하세요: 정상이면 OK"}],
        max_tokens=5,
    )
    print(f"  [OpenAI chat ] {config.CHAT_MODEL} -> {resp.choices[0].message.content.strip()}")


def check_openai_embed() -> None:
    client = config.get_openai_client()
    resp = client.embeddings.create(model=config.EMBED_MODEL, input="염수 분무 시험")
    dim = len(resp.data[0].embedding)
    print(f"  [OpenAI embed] {config.EMBED_MODEL} -> 벡터 차원 {dim}")


def check_neo4j() -> None:
    driver = config.get_neo4j_driver()
    driver.verify_connectivity()
    with driver.session() as session:
        val = session.run("RETURN 1 AS ok").single()["ok"]
    print(f"  [Neo4j      ] {config.NEO4J_URI} -> RETURN 1 = {val}")


def main() -> None:
    checks = [
        ("OpenAI Chat", check_openai_chat),
        ("OpenAI Embedding", check_openai_embed),
        ("Neo4j 연결", check_neo4j),
    ]
    failed = 0
    print("Astra-RAG-Agent smoke test")
    print("-" * 40)
    for name, fn in checks:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {name}: {exc}")
    print("-" * 40)
    if failed:
        print(f"실패 {failed}건. .env 키 / Docker(Neo4j) 기동 상태를 확인하세요.")
        sys.exit(1)
    print("OK: Phase 1 smoke test 통과.")


if __name__ == "__main__":
    main()
