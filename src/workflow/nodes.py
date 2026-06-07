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

# 질의 유형(A/B/C/D) 결정론적 라우팅 키워드.
# 우선순위: D(절차) > B(판정) > A(기준) > C(설명). 더 구체적인 의도가 먼저 매칭되어야
# 카테고리 청크 혼입으로 인한 응답 형식 오선택을 막을 수 있다.
TYPE_D_KEYWORDS = [  # 시험 절차/방법
    "절차", "방법", "순서", "어떻게", "단계", "시험 방법", "시험방법",
    "procedure", "how to", "steps", "step ",
]
TYPE_B_KEYWORDS = [  # 적합성 판정
    "충족", "통과", "미달", "적합", "부적합", "합격", "불합격", "만족",
    "기준 미달", "위반", "넘는지", "넘어", "pass", "fail",
]
TYPE_A_KEYWORDS = [  # 규격 기준값 조회
    "얼마", "몇 ", "기준이 뭐", "기준치", "기준값", "허용치", "허용값", "threshold",
    "맞춰야", "되어야", "이상이어야", "이하이어야",
]


def detect_category(query: str) -> str | None:
    """진동/방사선/부식 카테고리 감지. 절차 질의에서는 임계치 청크 혼입을
    막기 위해 호출 측(analyze_node)에서 query_type=='D' 일 때 None 으로 덮어쓴다."""
    q = query.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return cat
    return None


def detect_query_type(query: str) -> str:
    """질의 유형(A/B/C/D) 결정론적 분류.

    LLM 에게 유형 선택을 맡기지 않고 라우터가 먼저 확정한다. 우선순위는
    의도가 더 구체적인 D(절차) → B(판정) → A(기준) 순. 어디에도 매칭되지
    않으면 C(설명)로 폴백한다.
    """
    q = query.lower()
    if any(kw in q for kw in TYPE_D_KEYWORDS):
        return "D"
    if any(kw in q for kw in TYPE_B_KEYWORDS):
        return "B"
    if any(kw in q for kw in TYPE_A_KEYWORDS):
        return "A"
    return "C"


def _merge(existing: list, new: list) -> list:
    seen = {d["id"] for d in existing}
    return existing + [d for d in new if d["id"] not in seen]


def analyze_node(state: RAGState) -> dict:
    q = state["question"]
    expanded = expand_query_ko_en(q)
    known = gr.find_known_components(q)
    query_type = detect_query_type(q)
    category = detect_category(q)
    # 절차/방법 질의(D)에서는 그래프 임계치 청크(REQ-*) 혼입을 차단해
    # "절차를 물었는데 기준값을 답하는" 라우팅 오류를 방지한다.
    if query_type == "D":
        category = None
    return {
        "expanded_query": expanded,
        "known_components": known,
        "category": category,
        "query_type": query_type,
        "trace": state.get("trace", []) + [
            f"[analyze] 유형={query_type}, 부품인식={known or '없음'}, 카테고리={category or '없음'}"
        ],
    }


def retrieve_node(state: RAGState) -> dict:
    attempt = state.get("retry_count", 0)
    k = 5 if attempt == 0 else 10  # 재시도 시 검색 폭 확대
    q = state["question"]
    category = state.get("category")

    # 1) 그래프: 조립체면 의존성 순회, 리프 부품이면 단일 결정론 판정
    graph_docs: list = []
    for pn in state.get("known_components", []):
        subs = gr.get_subcomponents(pn)
        if subs and category:
            assessment = gr.assess_assembly(pn, category)
            graph_docs += gr.assessment_to_documents(assessment)
        elif category:
            # 리프 부품 + 카테고리: 단일 부품을 규격 임계치와 직접 대조해 결정론적 판정
            single_docs = gr.single_assessment_to_document(
                gr.assess_single_component(pn, category)
            )
            if single_docs:
                graph_docs += single_docs
            else:
                doc = gr.get_component_as_document(pn)
                if doc:
                    graph_docs.append(doc)
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
    query_type = state.get("query_type", "C")
    # 그래프 판정(PASS/FAIL)이 이미 있으면 LLM 호출 없이 SUFFICIENT 확정
    has_verdict = any(
        d.get("metadata", {}).get("verdict") in ("PASS", "FAIL")
        for d in docs
    )
    if has_verdict:
        sufficient = True
    else:
        sufficient = is_sufficient(state["question"], docs, query_type)
    return {
        "is_sufficient": sufficient,
        "trace": state.get("trace", []) + [
            f"[verify] 정보 충족={sufficient}" + ("" if sufficient else " → 재검색(self-correction)")
        ],
    }


def report_node(state: RAGState) -> dict:
    result = generate_answer(
        state["question"], state.get("docs", []), state.get("query_type", "C")
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "trace": state.get("trace", []) + ["[report] 한국어 판정 리포트 생성 완료"],
    }
