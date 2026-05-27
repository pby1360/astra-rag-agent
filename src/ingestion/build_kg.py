"""부품/규격/요구사항을 Neo4j 지식그래프로 적재 (GraphRAG).

노드:  Component(part_number, name, specs...), Standard(name), Requirement(id, parameter, threshold...)
엣지:  (Component)-[:DEPENDS_ON]->(Component)      # BOM 하위 의존
       (Component)-[:TESTED_PER]->(Standard)       # 준수 규격
       (Standard)-[:HAS_REQUIREMENT]->(Requirement) # 규격 세부 요구치

재실행 시 전체 그래프를 초기화(DETACH DELETE)한 뒤 멱등(MERGE) 적재한다.

실행:  python -m src.ingestion.build_kg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.utils import enable_utf8_stdout  # noqa: E402


def _load_components() -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(config.COMPONENTS_DIR.glob("*.json"))
    ]


def _load_requirements() -> list[dict]:
    path = config.DATA_DIR / "requirements.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("requirements", [])


def build() -> dict:
    driver = config.get_neo4j_driver()
    components = _load_components()
    requirements = _load_requirements()

    with driver.session() as s:
        # 초기화 + 제약
        s.run("MATCH (n) DETACH DELETE n")
        s.run("CREATE CONSTRAINT comp_pn IF NOT EXISTS "
              "FOR (c:Component) REQUIRE c.part_number IS UNIQUE")
        s.run("CREATE CONSTRAINT std_name IF NOT EXISTS "
              "FOR (st:Standard) REQUIRE st.name IS UNIQUE")
        s.run("CREATE CONSTRAINT req_id IF NOT EXISTS "
              "FOR (r:Requirement) REQUIRE r.id IS UNIQUE")

        # 부품 노드 + 엣지
        for d in components:
            specs = {k: v for k, v in (d.get("specs") or {}).items() if v is not None}
            s.run(
                """
                MERGE (c:Component {part_number: $pn})
                SET c.name = $name, c.manufacturer = $mfr,
                    c.category = $cat, c.is_real_part = $real
                SET c += $specs
                """,
                pn=d["part_number"], name=d.get("name", ""),
                mfr=d.get("manufacturer", ""), cat=d.get("category", ""),
                real=bool(d.get("is_real_part", False)), specs=specs,
            )
            for dep in d.get("depends_on", []):
                s.run(
                    """
                    MATCH (c:Component {part_number: $pn})
                    MERGE (d:Component {part_number: $dep})
                    MERGE (c)-[:DEPENDS_ON]->(d)
                    """,
                    pn=d["part_number"], dep=dep,
                )
            for std in d.get("compliance", []):
                s.run(
                    """
                    MATCH (c:Component {part_number: $pn})
                    MERGE (st:Standard {name: $std})
                    MERGE (c)-[:TESTED_PER]->(st)
                    """,
                    pn=d["part_number"], std=std,
                )

        # 요구사항 노드 + 엣지
        for r in requirements:
            s.run(
                """
                MERGE (st:Standard {name: $std})
                MERGE (req:Requirement {id: $id})
                SET req.parameter = $param, req.operator = $op,
                    req.threshold = $thr, req.unit = $unit,
                    req.category = $cat, req.description = $desc
                MERGE (st)-[:HAS_REQUIREMENT]->(req)
                """,
                std=r["standard"], id=r["id"], param=r["parameter"],
                op=r["operator"], thr=r["threshold"], unit=r.get("unit", ""),
                cat=r.get("category", ""), desc=r.get("description", ""),
            )

        counts = {
            "Component": s.run("MATCH (c:Component) RETURN count(c) AS n").single()["n"],
            "Standard": s.run("MATCH (st:Standard) RETURN count(st) AS n").single()["n"],
            "Requirement": s.run("MATCH (r:Requirement) RETURN count(r) AS n").single()["n"],
            "DEPENDS_ON": s.run("MATCH ()-[r:DEPENDS_ON]->() RETURN count(r) AS n").single()["n"],
            "TESTED_PER": s.run("MATCH ()-[r:TESTED_PER]->() RETURN count(r) AS n").single()["n"],
            "HAS_REQUIREMENT": s.run("MATCH ()-[r:HAS_REQUIREMENT]->() RETURN count(r) AS n").single()["n"],
        }
    return counts


def main() -> None:
    enable_utf8_stdout()
    print("Neo4j 지식그래프 적재 중...")
    counts = build()
    print("적재 완료:")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
