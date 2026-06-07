"""Astra RAG — Streamlit 시연 UI."""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src.logging_config import setup_logging  # noqa: E402
from src.workflow.builder import build_graph  # noqa: E402

setup_logging()

EXAMPLES = {
    "🛰️ S1 · 진동 규격 연쇄 위반": (
        "LV-ENGINE-X 엔진의 진동 규격이 MIL-STD-810H Method 514 "
        "Category 24로 상향되면, 하위 부품 중 기준 미달인 것을 모두 찾아줘."
    ),
    "☢️ S2 · COTS 방사선 교차검증": (
        "TI TPS7H1101-SP를 GEO 15년 임무에 쓰려는데 방사선(TID) 기준을 충족해?"
    ),
    "🌊 S3 · 염수분무 코팅 (한영)": (
        "K-방산 수출용 안테나 전개 장치, 염수 분무(소금물) 테스트 "
        "통과하려면 내부식성 코팅 두께를 얼마로 해야 해?"
    ),
    "🔁 자기수정(self-correction) 데모": (
        "AVIONICS-MOD-E의 진동 내구(g rms)가 Category 24 기준을 충족해?"
    ),
}

DOC_TYPE_ICON = {
    "graph": "🔗", "standard": "📄", "component": "🔩",
    "requirement": "📊", "glossary": "📖",
}
VERDICT_COLOR = {"PASS": "normal", "FAIL": "inverse", "INSUFFICIENT": "off"}


@st.cache_resource(show_spinner="LangGraph 워크플로우 초기화 중...")
def get_graph():
    return build_graph()


def _extract_verdict(answer: str) -> str | None:
    """답변 텍스트에서 판정 키워드를 추출."""
    m = re.search(r"판정[:\s]*([A-Z가-힣\s]+)", answer)
    if not m:
        return None
    parts = m.group(1).strip().split()
    if not parts:
        return None
    token = parts[0]
    if "PASS" in token:
        return "PASS"
    if "FAIL" in token:
        return "FAIL"
    if "정보" in token or "없음" in token:
        return "정보 없음"
    return None


def _verdict_from_sources(sources: list) -> str | None:
    """출처 목록에서 전체 판정을 추론 (FAIL 하나라도 있으면 FAIL)."""
    verdicts = [s.get("verdict", "") for s in sources if s.get("verdict")]
    if not verdicts:
        return None
    if "FAIL" in verdicts:
        return "FAIL"
    if all(v == "PASS" for v in verdicts):
        return "PASS"
    return "INSUFFICIENT"


def render_verdict_banner(answer: str, sources: list) -> None:
    verdict = _extract_verdict(answer) or _verdict_from_sources(sources)
    if verdict == "PASS":
        st.success("🟢 **PASS** — 규격 기준 충족", icon="✅")
    elif verdict == "FAIL":
        st.error("🔴 **FAIL** — 규격 기준 미달", icon="❌")
    elif verdict == "정보 없음":
        st.warning("🟡 **정보 없음** — 판정에 필요한 수치를 찾지 못했습니다.", icon="⚠️")


def render_sources(sources: list) -> None:
    """답변 본문의 [N] 인용 번호와 시각적으로 일치하도록 출처를 렌더링한다.

    - 각 항목 라벨 앞에 `[N]` 을 표기 (N = generator 가 매긴 cite_index)
    - doc_type 그룹은 유지하되, 그룹 내에서는 cite_index 오름차순으로 정렬
    - cite_index 누락 시(역호환) 입력 순서를 1-base 폴백으로 사용
    """
    if not sources:
        return

    # cite_index 폴백: 구버전 메시지에도 안전하게 동작
    for i, s in enumerate(sources, start=1):
        s.setdefault("cite_index", i)

    def _grp(t: str) -> list:
        return sorted(
            [s for s in sources if s.get("doc_type") == t],
            key=lambda x: x["cite_index"],
        )

    graph_docs = _grp("graph")
    std_docs   = _grp("standard")
    comp_docs  = _grp("component")
    req_docs   = _grp("requirement")
    other_docs = sorted(
        [s for s in sources if s.get("doc_type") not in
         {"graph", "standard", "component", "requirement"}],
        key=lambda x: x["cite_index"],
    )

    with st.expander(f"📚 참조 문서 {len(sources)}개 (본문 [N] 과 동일 번호)", expanded=False):
        first_group = True

        # 그래프 판정 결과
        if graph_docs:
            st.markdown("**🔗 지식그래프 판정**")
            cols = st.columns(min(len(graph_docs), 3))
            for i, s in enumerate(graph_docs):
                badge = {"PASS": "🟢", "FAIL": "🔴", "INSUFFICIENT": "🟡"}.get(
                    s.get("verdict", ""), "⚪"
                )
                pn = s.get("part_number") or s.get("label", "")
                with cols[i % 3]:
                    st.markdown(
                        f"{badge} **[{s['cite_index']}] {pn}**  \n"
                        f"<small>{s.get('verdict', '-')}</small>",
                        unsafe_allow_html=True,
                    )
            first_group = False

        # 규격 요구치 (REQ-*)
        if req_docs:
            if not first_group:
                st.divider()
            st.markdown("**📊 규격 요구치**")
            for s in req_docs:
                st.markdown(f"`[{s['cite_index']}] {s.get('label', '')}`")
                if s.get("snippet"):
                    st.caption(s["snippet"].replace("\n", " ")[:160] + "…")
            first_group = False

        # 규격 문서
        if std_docs:
            if not first_group:
                st.divider()
            st.markdown("**📄 규격 문서**")
            for s in std_docs:
                relevance = min(int(s.get("score", 0) * 5), 5)
                bar = "█" * relevance + "░" * (5 - relevance)
                st.markdown(
                    f"`[{s['cite_index']}] {s.get('label', '')}` &nbsp; `{bar}`",
                    unsafe_allow_html=True,
                )
                if s.get("snippet"):
                    st.caption(s["snippet"].replace("\n", " ")[:160] + "…")
            first_group = False

        # 부품 문서
        if comp_docs:
            if not first_group:
                st.divider()
            st.markdown("**🔩 부품 데이터**")
            for s in comp_docs:
                st.markdown(f"`[{s['cite_index']}] {s.get('label', '')}`")
                if s.get("snippet"):
                    st.caption(s["snippet"].replace("\n", " ")[:160] + "…")
            first_group = False

        # 기타(glossary 등)
        if other_docs:
            if not first_group:
                st.divider()
            st.markdown("**📖 기타**")
            for s in other_docs:
                st.markdown(f"`[{s['cite_index']}] {s.get('label', '')}`")
                if s.get("snippet"):
                    st.caption(s["snippet"].replace("\n", " ")[:160] + "…")


def render_trace(trace: list, dt: float) -> None:
    node_icon = {
        "analyze": "🔎", "retrieve": "📥", "verify": "⚖️", "report": "📝"
    }
    retries = sum(1 for t in trace if "[retrieve]" in t) - 1

    with st.expander("🔍 실행 흐름 보기", expanded=False):
        # 파이프라인 단계 표시
        cols = st.columns(len(trace))
        for i, t in enumerate(trace):
            node = t.split("]")[0].strip("[")
            icon = node_icon.get(node, "▸")
            with cols[i]:
                st.markdown(
                    f"<div style='text-align:center'>{icon}<br>"
                    f"<small><b>{node}</b></small></div>",
                    unsafe_allow_html=True,
                )

        st.divider()

        # 세부 내용
        for t in trace:
            st.markdown(f"`{t}`")

        # 요약 메트릭
        c1, c2, c3 = st.columns(3)
        c1.metric("총 소요 시간", f"{dt:.1f}s")
        c2.metric("실행 단계", len(trace))
        c3.metric("재검색 횟수", max(retries, 0))


def main() -> None:
    st.set_page_config(page_title="Astra RAG", page_icon="🛰️", layout="wide")
    st.title("🛰️ Astra RAG — 항공우주·방산 규격 교차검증")
    st.caption(
        "Dense(Chroma) + BM25 하이브리드 검색 · Neo4j 지식그래프 의존성 추론 · "
        "LangGraph 자기수정 워크플로우"
    )

    with st.sidebar:
        st.header("예시 질문")
        for label, q in EXAMPLES.items():
            if st.button(label, use_container_width=True):
                st.session_state.pending = q
        st.divider()
        st.markdown("**판정 범례**")
        st.markdown("🟢 PASS &nbsp;&nbsp; 🔴 FAIL &nbsp;&nbsp; 🟡 정보 없음")
        st.divider()
        st.markdown("**데이터셋**")
        st.markdown("📄 규격 문서 91개  \n🔩 부품 9개  \n🔗 그래프 노드/엣지")
        st.divider()
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                render_verdict_banner(m["content"], m.get("sources", []))
            st.markdown(m["content"])
            if m.get("sources"):
                render_sources(m["sources"])
            if m.get("trace"):
                render_trace(m["trace"], m.get("elapsed", 0))

    prompt = st.chat_input("질문을 입력하세요 (예: 부품의 규격 적합성 검증)")
    if not prompt and "pending" in st.session_state:
        prompt = st.session_state.pop("pending")

    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        graph = get_graph()
        init = {"question": prompt, "docs": [], "retry_count": 0, "trace": []}
        acc: dict = {}
        t0 = time.time()
        with st.status("⚙️ 에이전트 워크플로우 실행 중...", expanded=True) as status:
            for step in graph.stream(init, config={"recursion_limit": 12}):
                for node, update in step.items():
                    acc.update(update)
                    if update.get("trace"):
                        node_name = update["trace"][-1].split("]")[0].strip("[")
                        icon = {"analyze": "🔎", "retrieve": "📥",
                                "verify": "⚖️", "report": "📝"}.get(node_name, "▸")
                        status.write(f"{icon} {update['trace'][-1]}")
            dt = time.time() - t0
            status.update(label=f"✅ 완료 ({dt:.1f}s)", state="complete", expanded=False)

        answer = acc.get("answer", "(답변을 생성하지 못했습니다.)")
        sources = acc.get("sources", [])
        trace = acc.get("trace", [])

        render_verdict_banner(answer, sources)
        st.markdown(answer)
        render_sources(sources)
        render_trace(trace, dt)

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "sources": sources, "trace": trace, "elapsed": dt,
    })


main()
