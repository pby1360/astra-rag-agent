"""30종 검색 케이스 일괄 검증 (Phase 1 라우팅 + Phase 2 풀 실행).

Phase 1 — 드라이런: detect_query_type/detect_category/find_known_components 만 호출
                    (LLM/Chroma/Neo4j 깊은 호출 없음, 초당 수십 케이스 검증 가능)
Phase 2 — 풀 실행: graph.invoke 로 실제 RAG 파이프라인 통과
                    응답 키워드 / 출처 / 금지어 검증

실행:
    python -m tests.coverage_check           # 둘 다
    python -m tests.coverage_check phase1    # 라우팅만
    python -m tests.coverage_check phase2    # 풀 실행만
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import enable_utf8_stdout  # noqa: E402
from src.workflow.nodes import detect_category, detect_query_type  # noqa: E402


@dataclass
class Case:
    cid: str
    question: str
    # 라우팅 기대치 (Phase 1)
    exp_type: str                     # A/B/C/D
    exp_category: str | None          # vibration/radiation/corrosion/None
    exp_known: list[str] = field(default_factory=list)  # 부분 일치 기대 부품번호
    # 응답 기대치 (Phase 2)
    exp_source_any: list[str] = field(default_factory=list)  # 출처 라벨 부분일치(1개 이상)
    exp_answer_any: list[str] = field(default_factory=list)  # 답변 부분일치(전부 포함)
    exp_not: list[str] = field(default_factory=list)        # 답변/출처에 없어야 함
    skip_phase2: bool = False         # 환경 의존이 큰 케이스는 Phase 2 생략


# -------------------------------------------------------------------------
# 30종 케이스 정의 (위 정리한 A1~G4 + F2~F4 = 30종)
# -------------------------------------------------------------------------
CASES: list[Case] = [
    # ── A. 규격 기준값 조회 ─────────────────────────────────────
    Case("A1", "Falcon 9 PAF 진동 기준 얼마야?",
         "A", "vibration",
         exp_source_any=["falcon", "spacex"], exp_answer_any=["8.8"],
         exp_not=["pass", "fail"]),
    # A2: 현 시스템 한계로 기록되는 케이스 (질의 변경 금지).
    # "PSD 값" 은 Method 514 본문의 PSD 곡선(g²/Hz)을 의미할 수 있고,
    # requirements.json 의 14.1 g rms 는 Cat 24 누적 진동 강도. 두 값은 별개.
    # Dense 가 PSD 키워드에 끌려 본문 청크를 우선시하므로 임계치 청크가 RRF 에서 밀린다.
    # → 데이터에 "Cat 24 = 14.1 g rms" 가 PSD 라벨로 적힌 청크가 추가되어야 자연 해결.
    Case("A2", "MIL-STD-810H Method 514 Category 24의 PSD 값은 얼마?",
         "A", "vibration",
         exp_source_any=["514"], exp_answer_any=["14.1"],
         exp_not=["pass", "fail"]),
    Case("A3", "GEO 15년 임무 TID 허용치는 얼마야?",
         "A", "radiation",
         exp_source_any=["geo", "tid", "radiation"], exp_answer_any=["50"],
         exp_not=["pass", "fail"]),
    Case("A4", "염수분무 코팅 두께 기준이 얼마야?",
         "A", "corrosion",
         exp_source_any=["509", "corrosion", "saltfog"], exp_answer_any=["20"],
         exp_not=["pass", "fail"]),
    # A5: 데이터 미수록 한계로 기록되는 케이스 (질의 변경 금지).
    # requirements.json 은 단일 임계치만 보유하므로 "등급별 차이" 정보가 없다.
    # LLM 이 가장 가까운 REQ-TID-GEO15(≥50 krad) 를 인용하지만 답변에 "1019" 라벨은
    # 누락된다. → Method 1019 의 Class A/B/C 별 임계치를 standards 청크에 보강해야 해결.
    Case("A5", "MIL-STD-883 Method 1019 TID 등급별 차이가 얼마야?",
         "A", "radiation",
         exp_source_any=["1019"], exp_answer_any=["1019"],
         exp_not=[]),
    Case("A6", "습도 시험 권장 RH 조건은 몇 %?",
         "A", None,
         exp_source_any=["507", "humidity"], exp_answer_any=[],  # 수치 다양
         exp_not=["pass", "fail"]),

    # ── B. 부품 적합성 판정 ─────────────────────────────────────
    Case("B1", "TPS7H1101-SP가 GEO 15년 TID 기준을 충족해?",
         "B", "radiation", exp_known=["TPS7H1101-SP"],
         exp_source_any=["tps7h1101"], exp_answer_any=["pass"],
         exp_not=["정보 없음"]),
    Case("B2", "SAMV71Q21RT의 동작 온도 -55~125 범위를 만족해?",
         "B", None, exp_known=["SAMV71Q21RT"],
         exp_source_any=["samv71q21rt"], exp_answer_any=[],
         exp_not=[]),
    Case("B3", "LV-ENGINE-X 진동 Category 24 기준 미달 부품을 찾아줘.",
         "B", "vibration", exp_known=["LV-ENGINE-X"],
         exp_source_any=["lv-engine-x"], exp_answer_any=["fail"],
         exp_not=[]),  # 일부 하위 부품에 스펙 미기재 → 행에 "정보 없음" 등장은 정상
    Case("B4", "STARLINK-SAT-GEN2 TID 기준을 모두 통과해?",
         "B", "radiation", exp_known=["STARLINK-SAT-GEN2"],
         exp_source_any=["starlink-sat-gen2"], exp_answer_any=["pass"],
         exp_not=["정보 없음"]),
    Case("B5", "XQR5VFX130과 RAD750 중 TID 내성이 더 우수한 부품이 적합한지 판정해줘.",
         "B", "radiation", exp_known=["XQR5VFX130", "RAD750"],
         exp_source_any=["xqr5vfx130", "rad750"], exp_answer_any=["1000"],
         exp_not=["정보 없음"]),
    Case("B6", "FUEL-VALVE-C가 Falcon 9 진동 기준을 통과해?",
         "B", "vibration", exp_known=["FUEL-VALVE-C"],
         exp_source_any=["fuel-valve-c"], exp_answer_any=[],
         exp_not=["정보 없음"]),
    Case("B7", "가상 부품 ABC-9999-XYZ가 기준을 통과해?",
         "B", None,
         exp_source_any=[], exp_answer_any=["정보 없음"],
         exp_not=["pass", "fail"]),

    # ── C. 부품/규격 설명 ─────────────────────────────────────
    Case("C1", "RAD750은 어떤 부품이야?",
         "C", None, exp_known=["RAD750"],
         exp_source_any=["rad750"], exp_answer_any=["rad750"],
         exp_not=["pass", "fail"]),
    Case("C2", "TPS7A4501-SP를 설명해줘.",
         "C", None, exp_known=["TPS7A4501-SP"],
         exp_source_any=["tps7a4501"], exp_answer_any=["tps7a4501"],
         exp_not=["pass", "fail"]),
    Case("C3", "MIL-STD-461G가 뭐야?",
         "C", None,
         exp_source_any=["461"], exp_answer_any=["461"],
         exp_not=["pass", "fail"]),
    Case("C4", "SMC-S-016은 어떤 규격이지?",
         "C", None,
         exp_source_any=["smc"], exp_answer_any=["smc"],
         exp_not=["pass", "fail"]),
    Case("C5", "등록된 우주용 LDO(전원 IC) 부품에는 어떤 것들이 있어?",
         "C", None,
         exp_source_any=["tps7h1101", "tps7a4501"], exp_answer_any=["tps7"],
         exp_not=["pass", "fail"]),
    Case("C6", "AVIONICS-MOD-E는 어떤 구성이야?",
         "C", None, exp_known=["AVIONICS-MOD-E"],
         exp_source_any=["avionics-mod-e"], exp_answer_any=["avionics-mod-e"],
         exp_not=["pass", "fail"]),

    # ── D. 시험 절차/방법 설명 ─────────────────────────────────
    Case("D1", "MIL-STD-810H Method 514의 진동 시험 절차는 어떻게 진행해?",
         "D", None,
         exp_source_any=["514"], exp_answer_any=["514"],
         exp_not=["pass", "fail", "8.8"]),  # Falcon 9 임계치 혼입 차단
    Case("D2", "염수 분무 시험은 어떻게 진행해?",
         "D", None,
         exp_source_any=["509", "salt"], exp_answer_any=[],
         exp_not=["pass", "fail", "20 ", "20μ"]),  # 코팅 임계치 혼입 차단
    Case("D3", "TID 시험 단계가 어떻게 돼?",
         "D", None,
         exp_source_any=["1019", "tid", "radiation"], exp_answer_any=[],
         exp_not=["pass", "fail"]),
    Case("D4", "온도 충격 시험 방법이 뭐야?",
         "D", None,
         exp_source_any=["503"], exp_answer_any=[],
         exp_not=["pass", "fail"]),
    Case("D5", "습도 시험 순서가 어떻게 돼?",
         "D", None,
         exp_source_any=["507", "humidity"], exp_answer_any=[],
         exp_not=["pass", "fail"]),
    Case("D6", "EMC RS103 시험은 어떻게 해?",
         "D", None,
         exp_source_any=["461"], exp_answer_any=[],
         exp_not=["pass", "fail"]),

    # ── E. 한영 혼용 / 글로서리 확장 ────────────────────────────
    # E1/E2/E3 는 다른 유형과 중복되므로 글로서리 확장 효과만 별도 확인.
    Case("E4", "충격 g 값 한계가 얼마야?",
         "A", None,
         exp_source_any=["shock", "516"], exp_answer_any=[],
         exp_not=["pass", "fail"]),

    # ── F. 검색 경로 / 메타데이터 ───────────────────────────────
    Case("F1", "MIL-STD-883 Method 1004는 뭐야?",
         "C", None,
         exp_source_any=["1004"], exp_answer_any=["1004"],
         exp_not=["pass", "fail"]),
    Case("F3", "MIL-STD-810H Method 514와 Method 516의 차이점이 뭐야?",
         "C", None,
         exp_source_any=["514", "516"], exp_answer_any=["진동"],
         exp_not=["pass", "fail"]),

    # ── G. 안전성 / Self-Correction ─────────────────────────────
    Case("G1", "MIL-STD-9999는 뭐야?",
         "C", None,
         exp_source_any=[], exp_answer_any=["정보 없음"],
         exp_not=[]),
    Case("G2", "UNKNOWN-PROC-ABC123이 GEO TID 기준을 통과해?",
         "B", "radiation",
         exp_source_any=[], exp_answer_any=["정보 없음"],
         exp_not=["pass", "fail"]),
    Case("G4", "What is the salt fog test procedure?",
         "D", None,
         exp_source_any=["509", "salt"], exp_answer_any=[],
         exp_not=["pass", "fail"]),
]


# -------------------------------------------------------------------------
# Phase 1 — 라우팅 드라이런
# -------------------------------------------------------------------------
def phase1() -> list[tuple[Case, list[str]]]:
    """라우터(detect_query_type / detect_category / find_known_components)만 호출."""
    from src.retrieval import graph_retriever as gr

    print(f"\n{'='*72}\nPhase 1 — 라우팅 드라이런 (30 cases)\n{'='*72}")
    all_known = gr.list_part_numbers()  # 1회만 조회
    failures: list[tuple[Case, list[str]]] = []

    for c in CASES:
        actual_type = detect_query_type(c.question)
        actual_cat = detect_category(c.question)
        if actual_type == "D":
            actual_cat = None  # analyze_node 와 동일 규칙
        actual_known = [pn for pn in all_known if pn.lower() in c.question.lower()]

        errs: list[str] = []
        if actual_type != c.exp_type:
            errs.append(f"type: 기대={c.exp_type} 실제={actual_type}")
        if actual_cat != c.exp_category:
            errs.append(f"category: 기대={c.exp_category} 실제={actual_cat}")
        if set(actual_known) != set(c.exp_known):
            errs.append(f"known: 기대={c.exp_known} 실제={actual_known}")

        mark = "✅" if not errs else "❌"
        print(f"{mark} [{c.cid:4}] type={actual_type} cat={str(actual_cat):10} "
              f"known={actual_known}")
        if errs:
            for e in errs:
                print(f"     └─ {e}")
            failures.append((c, errs))

    print(f"\nPhase 1 결과: {len(CASES)-len(failures)}/{len(CASES)} 통과")
    return failures


# -------------------------------------------------------------------------
# Phase 2 — 풀 그래프 실행
# -------------------------------------------------------------------------
def phase2(filter_ids: set[str] | None = None) -> list[tuple[Case, list[str]]]:
    from src.workflow.builder import run

    cases = [c for c in CASES if not c.skip_phase2 and
             (filter_ids is None or c.cid in filter_ids)]
    print(f"\n{'='*72}\nPhase 2 — 풀 그래프 실행 ({len(cases)} cases)\n{'='*72}")
    failures: list[tuple[Case, list[str]]] = []

    for c in cases:
        t0 = time.time()
        try:
            state = run(c.question)
        except Exception as exc:
            print(f"❌ [{c.cid:4}] 실행 예외: {exc}")
            failures.append((c, [f"실행 예외: {exc}"]))
            continue
        dt = time.time() - t0

        answer = (state.get("answer") or "").lower()
        source_blob = " ".join(
            s.get("label", "") for s in state.get("sources", [])
        ).lower()
        qtype = state.get("query_type", "?")

        errs: list[str] = []
        # 출처 부분일치 (any-of: 하나라도 만족)
        if c.exp_source_any and not any(s.lower() in source_blob
                                        for s in c.exp_source_any):
            errs.append(f"출처에 {c.exp_source_any} 중 하나도 없음 "
                        f"(실제: {source_blob[:80]}…)")
        # 답변 키워드 (all: 전부 포함)
        for kw in c.exp_answer_any:
            if kw.lower() not in answer:
                errs.append(f"답변에 '{kw}' 없음")
        # 금지어
        for bad in c.exp_not:
            if bad.lower() in answer:
                errs.append(f"답변에 금지어 '{bad}' 있음")

        mark = "✅" if not errs else "❌"
        print(f"{mark} [{c.cid:4}] {dt:.1f}s type={qtype} "
              f"src={len(state.get('sources', []))}건 "
              f"answer={answer[:60]}…")
        if errs:
            for e in errs:
                print(f"     └─ {e}")
            failures.append((c, errs))

    print(f"\nPhase 2 결과: {len(cases)-len(failures)}/{len(cases)} 통과")
    return failures


def main() -> None:
    enable_utf8_stdout()
    args = sys.argv[1:]
    mode = args[0] if args else "all"

    if mode in ("phase1", "all"):
        p1_fails = phase1()
    else:
        p1_fails = []

    if mode in ("phase2", "all"):
        # Phase 1 실패한 케이스도 Phase 2 진행 (응답까지 어떻게 나오는지 확인)
        p2_fails = phase2()
    else:
        p2_fails = []

    print(f"\n{'='*72}\n전체 요약")
    print(f"  Phase 1 실패: {len(p1_fails)} / Phase 2 실패: {len(p2_fails)}")
    if p1_fails or p2_fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
