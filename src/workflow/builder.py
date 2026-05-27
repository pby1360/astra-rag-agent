"""LangGraph 그래프 구성 + 조건 분기 (self-correction 루프).

흐름:  START → analyze → retrieve → verify ─(충분)→ report → END
                                      └─(부족 & 재시도 여유)→ retrieve (재검색)
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.workflow import nodes
from src.workflow.state import RAGState

MAX_RETRIES = 2  # retrieve 최대 진입 횟수 (1회 재검색 허용)


def should_retry(state: RAGState) -> str:
    """verify 이후 분기: 충분하거나 재시도 소진 시 report, 아니면 재검색."""
    if state.get("is_sufficient"):
        return "report"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "report"
    return "retrieve"


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("analyze", nodes.analyze_node)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("verify", nodes.verify_node)
    g.add_node("report", nodes.report_node)

    g.add_edge(START, "analyze")
    g.add_edge("analyze", "retrieve")
    g.add_edge("retrieve", "verify")
    g.add_conditional_edges(
        "verify", should_retry, {"report": "report", "retrieve": "retrieve"}
    )
    g.add_edge("report", END)
    return g.compile()


def run(question: str) -> dict:
    """질의를 그래프로 실행하고 최종 상태를 반환."""
    graph = build_graph()
    init: RAGState = {"question": question, "docs": [], "retry_count": 0, "trace": []}
    return graph.invoke(init, config={"recursion_limit": 12})
