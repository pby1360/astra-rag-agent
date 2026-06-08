"""정량평가 — demo_questions.yaml 기반 지표 산출.

regression.py 가 이진 pass/fail 만 본다면, 이 스크립트는 점수·지표를 계산한다:
  · 키워드 recall   : 답변에 포함된 기대 키워드 비율 (부분 점수)
  · 출처 recall     : 기대 출처가 sources.label 에 잡힌 비율
  · 부정 제약 준수  : expect_not 단어가 답변에 없는 비율
  · 라우팅 정확도   : detect_query_type 결과가 expect_type(GT)과 일치하는 비율
  · latency 분포    : mean / p50 / p95 / max
  · 재검색 / 충족   : self-correction 재시도 횟수, verify 충족 여부, verdict 유무

실행:
  python -m tests.evaluate                 # 전체 1회
  python -m tests.evaluate --repeat 3      # 각 시나리오 3회 → 안정성(표준편차) 측정
  python -m tests.evaluate --only S2_radiation
  python -m tests.evaluate --no-report     # JSON 저장 생략
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as stats
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from src.utils import enable_utf8_stdout  # noqa: E402
from src.workflow.builder import run  # noqa: E402

DEMO_YAML = Path(__file__).parent / "demo_questions.yaml"
REPORT_DIR = Path(__file__).parent / "reports"


# ───────────────────────── 매칭 유틸 ─────────────────────────

def _present(text: str, term: str) -> bool:
    """term 이 text 안에 '토큰 경계' 기준으로 존재하는지.

    단순 substring 은 "100"이 "1000"에 매칭되는 거짓양성을 낳는다. 영숫자
    경계(lookaround)를 적용해 숫자/단어가 더 긴 토큰의 일부일 때를 배제한다.
    (한글은 ASCII 영숫자 경계만 적용 — substring 허용으로 충분)
    """
    if not term:
        return False
    left = r"(?<![0-9A-Za-z])" if term[:1].isalnum() and term[:1].isascii() else ""
    right = r"(?![0-9A-Za-z])" if term[-1:].isalnum() and term[-1:].isascii() else ""
    return re.search(left + re.escape(term) + right, text, re.IGNORECASE) is not None


def _ratio(hit: int, total: int) -> float | None:
    """total 이 0이면 해당 지표 없음(None) — 매크로 평균에서 제외된다."""
    return None if total == 0 else hit / total


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(stats.mean(xs), 4) if xs else None


def _pct(xs: list[float], p: float) -> float:
    """간단 백분위수 (nearest-rank)."""
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return round(s[idx], 3)


# ───────────────────────── 단일 실행 채점 ─────────────────────────

def evaluate_once(scenario: dict) -> dict:
    """시나리오 1회 실행 → 지표 딕셔너리. 예외도 결과로 흡수한다."""
    question = scenario["question"]
    t0 = time.time()
    try:
        state = run(question)
        error = None
    except Exception as exc:  # 평가 자체는 멈추지 않고 실패로 기록
        state = {}
        error = f"{type(exc).__name__}: {exc}"
    latency = round(time.time() - t0, 3)

    answer = (state.get("answer") or "")
    labels = " ".join(s.get("label", "") for s in state.get("sources", []))

    kw = scenario.get("expect_keywords", [])
    kw_hit = sum(_present(answer, k) for k in kw)

    src = scenario.get("expect_sources", [])
    src_hit = sum(_present(labels, s) for s in src)

    neg = scenario.get("expect_not", [])
    neg_violations = sum(_present(answer, n) for n in neg)

    detected = state.get("query_type")
    expect_type = scenario.get("expect_type")
    routing_ok = None if expect_type is None else (detected == expect_type)

    kw_recall = _ratio(kw_hit, len(kw))
    src_recall = _ratio(src_hit, len(src))
    neg_ok = neg_violations == 0

    # strict pass = regression.py 와 동일한 전 항목 충족
    strict_pass = (
        error is None
        and (kw_recall in (None, 1.0))
        and (src_recall in (None, 1.0))
        and neg_ok
    )

    return {
        "latency_s": latency,
        "error": error,
        "query_type": detected,
        "expect_type": expect_type,
        "routing_ok": routing_ok,
        "keyword_recall": kw_recall,
        "keyword_hit": kw_hit,
        "keyword_total": len(kw),
        "source_recall": src_recall,
        "source_hit": src_hit,
        "source_total": len(src),
        "negative_ok": neg_ok,
        "negative_violations": neg_violations,
        "strict_pass": strict_pass,
        "is_sufficient": state.get("is_sufficient"),
        "retrieve_attempts": state.get("retry_count"),
        "has_verdict": any(
            s.get("verdict") in ("PASS", "FAIL") for s in state.get("sources", [])
        ),
        "n_sources": len(state.get("sources", [])),
        "answer_len": len(answer),
    }


def evaluate_scenario(scenario: dict, repeat: int, verbose: bool) -> dict:
    """시나리오를 repeat 회 실행 → 회차 평균 + 안정성 집계."""
    sid, name = scenario["id"], scenario["name"]
    runs = [evaluate_once(scenario) for _ in range(repeat)]

    kw = [r["keyword_recall"] for r in runs]
    sr = [r["source_recall"] for r in runs]
    lat = [r["latency_s"] for r in runs]
    passes = [r["strict_pass"] for r in runs]

    agg = {
        "id": sid,
        "name": name,
        "runs": repeat,
        "pass_rate": round(sum(passes) / repeat, 3),
        "keyword_recall": _mean(kw),
        "source_recall": _mean(sr),
        "negative_ok_rate": round(sum(r["negative_ok"] for r in runs) / repeat, 3),
        "routing_ok": runs[0]["routing_ok"],  # 결정론적이라 회차 무관
        "expect_type": runs[0]["expect_type"],
        "query_type": runs[0]["query_type"],
        "latency_mean": round(stats.mean(lat), 3),
        "latency_stdev": round(stats.pstdev(lat), 3) if repeat > 1 else 0.0,
        "errors": [r["error"] for r in runs if r["error"]],
        "raw": runs,
    }

    if verbose:
        flag = "OK " if agg["pass_rate"] == 1.0 else ("~~ " if agg["pass_rate"] > 0 else "XX ")
        rline = (f" route={agg['query_type']}/{agg['expect_type']}"
                 if agg["expect_type"] else "")
        print(
            f"  [{flag}] {sid:<26} "
            f"kw={_fmt(agg['keyword_recall'])} src={_fmt(agg['source_recall'])} "
            f"neg={agg['negative_ok_rate']:.0%} pass={agg['pass_rate']:.0%} "
            f"{agg['latency_mean']:.1f}s{rline}"
        )
        for e in agg["errors"][:1]:
            print(f"        ! {e}")
    return agg


def _fmt(x: float | None) -> str:
    return " -- " if x is None else f"{x:.0%}"


# ───────────────────────── 집계 ─────────────────────────

def aggregate(results: list[dict]) -> dict:
    all_latency = [r for res in results for r in
                   [run_["latency_s"] for run_ in res["raw"]]]
    routed = [res for res in results if res["routing_ok"] is not None]

    return {
        "scenarios": len(results),
        "strict_pass_rate": round(_mean([r["pass_rate"] for r in results]) or 0, 4),
        "keyword_recall_macro": _mean([r["keyword_recall"] for r in results]),
        "source_recall_macro": _mean([r["source_recall"] for r in results]),
        "negative_compliance": round(
            _mean([r["negative_ok_rate"] for r in results]) or 0, 4),
        "routing_accuracy": (
            round(sum(r["routing_ok"] for r in routed) / len(routed), 4)
            if routed else None),
        "routing_evaluated": len(routed),
        "latency_mean": round(stats.mean(all_latency), 3) if all_latency else 0,
        "latency_p50": _pct(all_latency, 50),
        "latency_p95": _pct(all_latency, 95),
        "latency_max": round(max(all_latency), 3) if all_latency else 0,
        "error_count": sum(len(r["errors"]) for r in results),
    }


def print_summary(summary: dict) -> None:
    print(f"\n{'='*68}")
    print("정량평가 요약")
    print(f"{'-'*68}")
    print(f"  시나리오               : {summary['scenarios']}건")
    print(f"  엄격 통과율(strict)    : {summary['strict_pass_rate']:.1%}")
    print(f"  키워드 recall (macro)  : {_fmt(summary['keyword_recall_macro'])}")
    print(f"  출처 recall  (macro)   : {_fmt(summary['source_recall_macro'])}")
    print(f"  부정 제약 준수율       : {summary['negative_compliance']:.1%}")
    if summary["routing_accuracy"] is not None:
        print(f"  라우팅 정확도          : {summary['routing_accuracy']:.1%} "
              f"({summary['routing_evaluated']}건 평가)")
    print(f"  latency mean/p50/p95   : "
          f"{summary['latency_mean']:.1f} / {summary['latency_p50']:.1f} / "
          f"{summary['latency_p95']:.1f}s")
    print(f"  오류 발생              : {summary['error_count']}건")
    print(f"{'='*68}")


# ───────────────────────── 엔트리포인트 ─────────────────────────

def main() -> None:
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="Astra-RAG-Agent 정량평가")
    ap.add_argument("--repeat", type=int, default=1, help="시나리오당 실행 횟수")
    ap.add_argument("--only", help="특정 시나리오 ID만")
    ap.add_argument("--no-report", action="store_true", help="JSON 리포트 저장 생략")
    ap.add_argument("--quiet", action="store_true", help="시나리오별 출력 생략")
    args = ap.parse_args()

    data = yaml.safe_load(DEMO_YAML.read_text(encoding="utf-8"))
    scenarios = data["scenarios"]
    if args.only:
        scenarios = [s for s in scenarios if s["id"] == args.only]
        if not scenarios:
            print(f"시나리오 ID '{args.only}' 없음")
            sys.exit(1)

    print(f"Astra-RAG-Agent 정량평가 | {len(scenarios)}개 시나리오 × {args.repeat}회")
    print(f"{'-'*68}")

    t0 = time.time()
    results = [evaluate_scenario(s, args.repeat, not args.quiet) for s in scenarios]
    wall = round(time.time() - t0, 1)

    summary = aggregate(results)
    summary["wall_time_s"] = wall
    print_summary(summary)

    if not args.no_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORT_DIR / f"eval_{ts}.json"
        path.write_text(
            json.dumps(
                {"summary": summary, "scenarios": results,
                 "meta": {"repeat": args.repeat, "timestamp": ts}},
                ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"리포트 저장: {path}")

    # CI 용 종료 코드: 오류가 있거나 엄격 통과율이 100% 미만이면 1
    sys.exit(0 if summary["error_count"] == 0 and summary["strict_pass_rate"] == 1.0 else 1)


if __name__ == "__main__":
    main()
