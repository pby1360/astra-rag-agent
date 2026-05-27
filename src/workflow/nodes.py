"""LangGraph 노드 함수.

기존 검색/생성 함수를 그대로 감싼다 (새 로직을 짜지 않음).
각 노드는 State 의 일부만 반환한다 (전체 덮어쓰기 방지).
"""
from __future__ import annotations

from src.generation.generator import generate_answer, is_sufficient
from src.retrieval import graph_retriever as gr
from src.retrieval.hybrid import expand_query_ko_en, hybrid_retrieve
from src.workflow.state import RAGState

# 질의 -> 판정 카테고리 (그래프 의존성 판정용)
CATEGORY_KEYWORDS = {
    "vibration": ["진동", "vibration", "category 24", "cat 24", "cat24", "psd", "g rms", "grms"],
    "radiation": ["방사", "tid", "총이온화", "ionizing", "radiation", "krad"],
    "corrosion": ["염수", "소금물", "salt fog", "부식", "코팅", "corrosion", "coating"],
}


def detect_category(query: str) -> str | None:
    q = query.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return cat
    return None


def _merge(existing: list, new: list) -> list:
    seen = {d["id"] for d in existing}
    return existing + [d for d in new if d["id"] not in seen]


def analyze_node(state: RAGState) -> dict:
    q = state["question"]
    expanded = expand_query_ko_en(q)
    known = gr.find_known_components(q)
    category = detect_category(q)
    return {
        "expanded_query": expanded,
        "known_components": known,
        "category": category,
        "trace": state.get("trace", []) + [
            f"[analyze] 부품인식={known or '없음'}, 카테고리={category or '없음'}"
        ],
    }


def retrieve_node(state: RAGState) -> dict:
    attempt = state.get("retry_count", 0)
    k = 5 if attempt == 0 else 10  # 재시도 시 검색 폭 확대
    q = state["question"]
    category = state.get("category")

    # 1) 그래프: 조립체면 의존성 순회, 리프 부품이면 직접 노드 스펙 조회
    graph_docs: list = []
    for pn in state.get("known_components", []):
        subs = gr.get_subcomponents(pn)
        if subs and category:
            assessment = gr.assess_assembly(pn, category)
            graph_docs += gr.assessment_to_documents(assessment)
        else:
            doc = gr.get_component_as_document(pn)
            if doc:
                graph_docs.append(doc)

    # 2) 하이브리드 (Dense + BM25 + 한영확장)
    hybrid_docs = hybrid_retrieve(q, k=k)

    docs = _merge(state.get("docs", []), graph_docs)
    docs = _merge(docs, hybrid_docs)
    return {
        "docs": docs,
        "retry_count": attempt + 1,
        "trace": state.get("trace", []) + [
            f"[retrieve] 시도{attempt + 1}: 그래프 {len(graph_docs)} + 하이브리드 {len(hybrid_docs)} → 누적 {len(docs)}"
        ],
    }


def verify_node(state: RAGState) -> dict:
    docs = state.get("docs", [])
    # 그래프 판정(PASS/FAIL)이 이미 있으면 LLM 호출 없이 SUFFICIENT 확정
    has_verdict = any(
        d.get("metadata", {}).get("verdict") in ("PASS", "FAIL")
        for d in docs
    )
    if has_verdict:
        sufficient = True
    else:
        sufficient = is_sufficient(state["question"], docs)
    return {
        "is_sufficient": sufficient,
        "trace": state.get("trace", []) + [
            f"[verify] 정보 충족={sufficient}" + ("" if sufficient else " → 재검색(self-correction)")
        ],
    }


def report_node(state: RAGState) -> dict:
    result = generate_answer(state["question"], state.get("docs", []))
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "trace": state.get("trace", []) + ["[report] 한국어 판정 리포트 생성 완료"],
    }
