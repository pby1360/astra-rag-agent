"""Neo4j 지식그래프 검색 (GraphRAG).

벡터/BM25 로는 불가능한 BOM 의존성 추론을 담당한다.
  - get_subcomponents: 부품의 하위 부품을 DEPENDS_ON 으로 최대 N단계 재귀 순회
  - assess_assembly:   하위 부품 각각을 규격 요구치와 대조해 PASS/FAIL/INSUFFICIENT 판정
  - find_known_components: 질의에 등장하는 등록 부품/조립체 식별 (그래프 진입점)
"""
from __future__ import annotations

import logging
import re

from src import config

logger = logging.getLogger(__name__)


def _run(cypher: str, **params) -> list[dict]:
    driver = config.get_neo4j_driver()
    with driver.session() as session:
        return [r.data() for r in session.run(cypher, **params)]


def list_part_numbers() -> list[str]:
    rows = _run("MATCH (c:Component) RETURN c.part_number AS pn")
    return [r["pn"] for r in rows]


def find_known_components(query: str) -> list[str]:
    """질의 문자열에 등장하는 등록 부품 번호를 반환 (대소문자 무시)."""
    q = query.lower()
    found = [pn for pn in list_part_numbers() if pn.lower() in q]
    if found:
        logger.info("질의에서 부품 식별 | %s", found)
    else:
        logger.info("질의에서 식별된 등록 부품 없음")
    return found


def get_component(part_number: str) -> dict | None:
    rows = _run(
        "MATCH (c:Component {part_number:$pn}) RETURN c AS c", pn=part_number
    )
    return rows[0]["c"] if rows else None


def get_subcomponents(part_number: str, max_depth: int = 3) -> list[dict]:
    """DEPENDS_ON 을 최대 max_depth 단계 순회해 하위 부품 + 준수 규격 반환."""
    depth = max(1, min(int(max_depth), 5))
    cypher = (
        f"MATCH (c:Component {{part_number:$pn}})-[:DEPENDS_ON*1..{depth}]->(dep:Component) "
        "OPTIONAL MATCH (dep)-[:TESTED_PER]->(s:Standard) "
        "RETURN dep AS dep, collect(DISTINCT s.name) AS standards"
    )
    out = []
    for r in _run(cypher, pn=part_number):
        node = dict(r["dep"])
        node["standards"] = r["standards"]
        out.append(node)
    if out:
        logger.info("하위 부품 순회 | %s → %d개: %s",
                    part_number, len(out), [s.get("part_number") for s in out])
    return out


def get_component_as_document(part_number: str) -> dict | None:
    """Neo4j 에서 부품 노드를 직접 조회해 generator 가 인용할 수 있는 문서로 반환."""
    rows = _run(
        "MATCH (c:Component {part_number:$pn}) "
        "OPTIONAL MATCH (c)-[:TESTED_PER]->(s:Standard) "
        "RETURN c AS c, collect(DISTINCT s.name) AS standards",
        pn=part_number,
    )
    if not rows:
        return None
    node = dict(rows[0]["c"])
    standards = rows[0]["standards"]
    props = {k: v for k, v in node.items() if k not in ("part_number", "name", "manufacturer", "category", "is_real_part")}
    lines = [
        f"[그래프] 부품 {node.get('part_number', '')} ({node.get('name', '')})",
        f"  - 제조사: {node.get('manufacturer', '')}",
        f"  - 분류: {node.get('category', '')}",
    ]
    if props:
        lines.append("  - 스펙(Neo4j 노드 속성):")
        lines += [f"    {k}: {v}" for k, v in props.items()]
    if standards:
        lines.append("  - 준수 규격: " + ", ".join(standards))
    text = "\n".join(lines)
    return {
        "id": f"graph::component::{part_number}",
        "text": text,
        "metadata": {
            "doc_type": "graph",
            "source": "neo4j",
            "part_number": part_number,
            "standard": standards[0] if standards else "",
            "verdict": "",
        },
        "score": 1.0,
    }


def _select_requirement(query: str | None, candidates: list[dict]) -> dict | None:
    """동일 카테고리에 요구치가 여러 개일 때, 질의 키워드로 가장 적합한 요구치를 고른다.

    각 후보의 (standard+description+id)에서 '다른 후보에는 없는' 식별 토큰을 뽑아,
    질의에 등장하는 식별 토큰 수가 가장 많은 후보를 선택한다. 매칭이 없으면
    MIL-STD 기준(없으면 id 사전순)을 결정론적 기본값으로 사용한다.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    q = (query or "").lower()

    def toks(r: dict) -> set[str]:
        text = f"{r.get('standard','')} {r.get('description','')} {r.get('id','')}".lower()
        return set(re.findall(r"[a-z0-9]+|[가-힣]+", text))

    token_sets = [toks(r) for r in candidates]
    best, best_score = None, -1
    for i, r in enumerate(candidates):
        others: set[str] = set().union(
            *[s for j, s in enumerate(token_sets) if j != i]
        )
        discriminating = token_sets[i] - others
        score = sum(1 for t in discriminating if len(t) >= 2 and t in q)
        if score > best_score:
            best, best_score = r, score
    if best_score <= 0:
        mil = [r for r in candidates
               if str(r.get("standard", "")).upper().startswith("MIL-STD")]
        return sorted(mil or candidates, key=lambda r: str(r.get("id", "")))[0]
    return best


def get_requirement_by_category(category: str, query: str | None = None) -> dict | None:
    """카테고리 요구치를 반환한다. 동일 카테고리에 여러 요구치가 있으면
    질의 키워드(예: Category 24 / Falcon / GEO / Starlink)로 정확히 선택한다."""
    rows = _run(
        "MATCH (st:Standard)-[:HAS_REQUIREMENT]->(r:Requirement {category:$cat}) "
        "RETURN r AS r, st.name AS standard",
        cat=category,
    )
    if not rows:
        return None
    candidates = []
    for row in rows:
        req = dict(row["r"])
        req["standard"] = row["standard"]
        candidates.append(req)
    chosen = _select_requirement(query, candidates)
    if chosen and len(candidates) > 1:
        logger.info("요구치 선택 | 카테고리=%s | 후보 %d개 → %s",
                    category, len(candidates), chosen.get("id"))
    return chosen


def _judge(value, operator: str, threshold) -> str:
    if value is None:
        return "INSUFFICIENT"
    if operator == ">=":
        return "PASS" if value >= threshold else "FAIL"
    if operator == "<=":
        return "PASS" if value <= threshold else "FAIL"
    return "INSUFFICIENT"


def assess_assembly(assembly_pn: str, category: str, query: str | None = None, max_depth: int = 3) -> dict | None:
    """조립체 하위 부품을 해당 카테고리 요구치와 대조해 판정한다 (시나리오 1).

    반환: {requirement, results:[{part_number,name,value,verdict,standards}, ...]}
    """
    logger.info("조립체 평가 시작 | %s | 카테고리=%s", assembly_pn, category)
    req = get_requirement_by_category(category, query)
    if req is None:
        logger.warning("요구치 없음 | 카테고리=%s", category)
        return None
    param, op, thr = req["parameter"], req["operator"], req["threshold"]
    results = []
    for sub in get_subcomponents(assembly_pn, max_depth):
        value = sub.get(param)
        verdict = _judge(value, op, thr)
        results.append({
            "part_number": sub.get("part_number"),
            "name": sub.get("name"),
            "value": value,
            "parameter": param,
            "verdict": verdict,
            "standards": sub.get("standards", []),
        })
        logger.info("  판정 | %s | %s=%s | 기준 %s%s | → %s",
                    sub.get("part_number"), param, value, op, thr, verdict)
    return {"requirement": req, "assembly": assembly_pn, "results": results}


def assess_single_component(part_number: str, category: str, query: str | None = None) -> dict | None:
    """리프(하위 부품 없는) 부품 1개를 카테고리 요구치와 직접 대조해 판정한다 (시나리오 S1b).

    조립체가 아닌 단일 부품 질의에서도 그래프 노드의 스펙과 규격 임계치를
    결정론적으로 비교해 PASS/FAIL/INSUFFICIENT 를 산출한다. 값이 없으면(스펙 누락)
    INSUFFICIENT 를 반환하여 self-correction 루프가 정상 동작하도록 한다.

    반환: {component, requirement, value, verdict} 또는 None
    """
    node = get_component(part_number)
    if node is None:
        return None
    req = get_requirement_by_category(category, query)
    if req is None:
        return None
    node = dict(node)
    param, op, thr = req["parameter"], req["operator"], req["threshold"]
    value = node.get(param)
    verdict = _judge(value, op, thr)
    logger.info("단일 부품 판정 | %s | %s=%s | 기준 %s%s | → %s",
                part_number, param, value, op, thr, verdict)
    return {"component": node, "requirement": req, "value": value, "verdict": verdict}


def single_assessment_to_document(assessment: dict | None) -> list[dict]:
    """단일 부품 판정 결과를 generator 가 인용할 수 있는 텍스트 문서로 변환."""
    if not assessment:
        return []
    req = assessment["requirement"]
    node = assessment["component"]
    value = assessment["value"]
    verdict = assessment["verdict"]
    pn = node.get("part_number", "")
    val = "정보 없음" if value is None else f"{value} {req.get('unit', '')}"
    text = (
        f"[그래프] 부품 {pn}({node.get('name', '')}) 단일 적합성 판정.\n"
        f"  - {req['parameter']} = {val}\n"
        f"  - 요구치({req['id']}, {req['standard']}): "
        f"{req['operator']} {req['threshold']} {req.get('unit', '')} "
        f"— {req.get('description', '')}\n"
        f"  - 판정: {verdict}"
    )
    return [{
        "id": f"graph::single::{pn}",
        "text": text,
        "metadata": {
            "doc_type": "graph",
            "source": "neo4j",
            "part_number": pn,
            "standard": req["standard"],
            "verdict": verdict,
        },
        "score": 1.0,
    }]


def assessment_to_documents(assessment: dict) -> list[dict]:
    """그래프 판정 결과를 generator 가 인용할 수 있는 텍스트 문서로 변환."""
    if not assessment:
        return []
    req = assessment["requirement"]
    docs = []
    for r in assessment["results"]:
        val = "정보 없음" if r["value"] is None else f"{r['value']} {req.get('unit','')}"
        text = (
            f"[그래프] 조립체 {assessment['assembly']} 의 하위 부품 {r['part_number']}"
            f"({r['name']}).\n"
            f"  - {req['parameter']} = {val}\n"
            f"  - 요구치({req['id']}, {req['standard']}): "
            f"{req['operator']} {req['threshold']} {req.get('unit','')} "
            f"— {req.get('description','')}\n"
            f"  - 판정: {r['verdict']}"
        )
        docs.append({
            "id": f"graph::{assessment['assembly']}::{r['part_number']}",
            "text": text,
            "metadata": {
                "doc_type": "graph",
                "source": "neo4j",
                "part_number": r["part_number"],
                "assembly": assessment["assembly"],  # 어셈블리 출처 추적용
                "standard": req["standard"],
                "verdict": r["verdict"],
            },
            "score": 1.0,
        })
    return docs
