"""회귀 테스트 — demo_questions.yaml 기반 자동 검증.

실행:  python -m tests.regression
       python -m tests.regression S2_radiation   # 특정 시나리오만
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from src.utils import enable_utf8_stdout  # noqa: E402
from src.workflow.builder import run  # noqa: E402

DEMO_YAML = Path(__file__).parent / "demo_questions.yaml"


def _check(state: dict, scenario: dict) -> list[str]:
    failures = []
    answer = (state.get("answer") or "").lower()
    source_labels = " ".join(
        s.get("label", "") for s in state.get("sources", [])
    ).lower()

    for kw in scenario.get("expect_keywords", []):
        if kw.lower() not in answer:
            failures.append(f"답변에 '{kw}' 없음")

    for src in scenario.get("expect_sources", []):
        if src.lower() not in source_labels:
            failures.append(f"출처에 '{src}' 없음")

    for bad in scenario.get("expect_not", []):
        if bad.lower() in answer:
            failures.append(f"답변에 '{bad}' 있어선 안 됨")

    return failures


def run_scenario(scenario: dict, verbose: bool = True) -> bool:
    sid = scenario["id"]
    name = scenario["name"]
    question = scenario["question"]

    print(f"\n{'─'*68}")
    print(f"[{sid}] {name}")
    print(f"질문: {question[:80]}{'…' if len(question)>80 else ''}")

    t0 = time.time()
    state = run(question)
    dt = time.time() - t0

    if verbose:
        print(f"trace: {' → '.join(t.split(']')[0].strip('[') for t in state.get('trace', []))}")
        print(f"답변: {(state.get('answer','') or '')[:200]}…")
        sources = state.get("sources", [])
        if sources:
            print(f"출처({len(sources)}): {', '.join(s.get('label','') for s in sources[:3])}")

    failures = _check(state, scenario)
    elapsed = f"{dt:.1f}s"
    if failures:
        print(f"❌ FAIL ({elapsed}): {'; '.join(failures)}")
        return False
    print(f"✅ PASS ({elapsed})")
    return True


def main() -> None:
    enable_utf8_stdout()
    data = yaml.safe_load(DEMO_YAML.read_text(encoding="utf-8"))
    scenarios = data["scenarios"]

    filter_id = sys.argv[1] if len(sys.argv) > 1 else None
    if filter_id:
        scenarios = [s for s in scenarios if s["id"] == filter_id]
        if not scenarios:
            print(f"시나리오 ID '{filter_id}' 없음. 가능한 ID: {[s['id'] for s in data['scenarios']]}")
            sys.exit(1)

    print(f"Astra-RAG-Agent 회귀 테스트 ({len(scenarios)}건)")
    passed = sum(run_scenario(s) for s in scenarios)
    total = len(scenarios)

    print(f"\n{'='*68}")
    print(f"결과: {passed}/{total} 통과" + (" ✅ 전체 통과" if passed == total else " ❌ 일부 실패"))
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
