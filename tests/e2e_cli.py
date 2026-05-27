"""End-to-end CLI: 질문 -> 벡터 검색 -> 답변 + 출처 출력.

실행:  python -m tests.e2e_cli
       python -m tests.e2e_cli "MIL-STD-810H Method 514 진동 카테고리 24 기준은?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.generator import generate_answer  # noqa: E402
from src.retrieval.hybrid import hybrid_retrieve  # noqa: E402
from src.utils import enable_utf8_stdout  # noqa: E402


def answer_once(question: str, k: int = 5) -> None:
    docs = hybrid_retrieve(question, k=k)
    result = generate_answer(question, docs)
    print("\n" + "=" * 70)
    print(result["answer"])
    print("-" * 70)
    print("참조 문서:")
    for i, s in enumerate(result["sources"], start=1):
        print(f"  [{i}] {s['label']}  (관련도 {s['score']:.4f})")
    print("=" * 70 + "\n")


def main() -> None:
    enable_utf8_stdout()
    if len(sys.argv) > 1:
        answer_once(" ".join(sys.argv[1:]))
        return
    print("Astra-RAG CLI (종료: exit)")
    while True:
        try:
            q = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", ""}:
            break
        answer_once(q)


if __name__ == "__main__":
    main()
