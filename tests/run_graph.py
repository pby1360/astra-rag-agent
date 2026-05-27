"""LangGraph 워크플로우 실행 테스트 (trace + 답변 + 출처).

실행:  python -m tests.run_graph "질문"
       python -m tests.run_graph        # 기본 시나리오 일괄 실행
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_config import setup_logging  # noqa: E402
from src.utils import enable_utf8_stdout  # noqa: E402
from src.workflow.builder import run  # noqa: E402

DEFAULT_QUESTIONS = [
    "LV-ENGINE-X 엔진의 진동 규격이 MIL-STD-810H Method 514 Category 24로 상향되면, 하위 부품 중 기준 미달인 것을 모두 찾아줘.",
    "TI TPS7H1101-SP를 GEO 15년 임무에 쓰려는데 방사선(TID) 기준을 충족해?",
    "AVIONICS-MOD-E의 진동 내구(g rms)가 Category 24 기준을 충족하는지 판정해줘.",
]


def show(question: str) -> None:
    state = run(question)
    print("\n" + "#" * 72)
    print("질문:", question)
    print("-" * 72)
    print("[실행 trace]")
    for t in state.get("trace", []):
        print("  ", t)
    print("-" * 72)
    print(state.get("answer", "(답변 없음)"))
    print("-" * 72)
    print("출처:")
    for i, s in enumerate(state.get("sources", []), 1):
        print(f"  [{i}] {s.get('label','')}")
    print("#" * 72)


def main() -> None:
    enable_utf8_stdout()
    setup_logging()
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
    else:
        for q in DEFAULT_QUESTIONS:
            show(q)


if __name__ == "__main__":
    main()
