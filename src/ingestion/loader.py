"""데이터 로딩 + 청킹.

세 종류의 소스를 통일된 청크 dict 리스트로 변환한다.
  1) data/standards/*.md   — frontmatter 분리 후 본문을 청킹 (doc_type=standard)
  2) data/components/*.json — 부품 1개 = 청크 1개로 직렬화 (doc_type=component)
  3) data/glossary/*.yaml   — 용어 1개 = mini-doc 청크 (doc_type=glossary)

각 청크: {"id": str, "text": str, "metadata": dict}
Chroma 메타데이터는 스칼라(str/int/float/bool)만 허용하므로 리스트/None 은 문자열화한다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, text[m.end():]


def _load_standards() -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(config.STANDARDS_DIR.glob("*.md")):
        meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        for i, piece in enumerate(_splitter.split_text(body)):
            piece = piece.strip()
            if len(piece) < 30:
                continue
            chunks.append({
                "id": f"{path.stem}::{i}",
                "text": piece,
                "metadata": {
                    "doc_type": "standard",
                    "source": path.name,
                    "standard": str(meta.get("standard", "")),
                    "method": str(meta.get("method", "")),
                    "category": str(meta.get("category", "general")),
                    "language": str(meta.get("language", "en")),
                    "chunk_index": i,
                },
            })
    return chunks


def _serialize_component(d: dict) -> str:
    lines = [
        f"부품번호(Part Number): {d.get('part_number','')}",
        f"명칭(Name): {d.get('name','')}",
        f"제조사(Manufacturer): {d.get('manufacturer','')}",
        f"분류(Category): {d.get('category','')}",
    ]
    specs = d.get("specs") or {}
    if specs:
        lines.append("스펙(Specs):")
        lines += [f"  - {k}: {v}" for k, v in specs.items()]
    if d.get("compliance"):
        lines.append("준수 규격(Compliance): " + ", ".join(d["compliance"]))
    if d.get("depends_on"):
        lines.append("하위 부품(Depends on): " + ", ".join(d["depends_on"]))
    if d.get("notes"):
        lines.append(f"비고(Notes): {d['notes']}")
    return "\n".join(lines)


def _load_components() -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(config.COMPONENTS_DIR.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        chunks.append({
            "id": f"component::{d['part_number']}",
            "text": _serialize_component(d),
            "metadata": {
                "doc_type": "component",
                "source": path.name,
                "part_number": str(d.get("part_number", "")),
                "manufacturer": str(d.get("manufacturer", "")),
                "category": str(d.get("category", "")),
                "compliance": ", ".join(d.get("compliance", [])),
                "is_real_part": bool(d.get("is_real_part", False)),
            },
        })
    return chunks


def _load_requirements() -> list[dict]:
    """data/requirements.json 의 임계치를 청크로 변환.

    그래프 분기를 타지 않는 자유 질의(known_components=[])에서도
    임계치(예: Category 24 = 14.1 g rms)가 검색되도록 Chroma 에 적재한다.
    각 임계치 = 청크 1개. ID 는 'requirement::REQ-*'.
    """
    req_path = config.DATA_DIR / "requirements.json"
    if not req_path.exists():
        return []
    data = json.loads(req_path.read_text(encoding="utf-8"))
    chunks: list[dict] = []
    for r in data.get("requirements", []):
        unit = r.get("unit", "")
        text = (
            f"규격 요구치 {r['id']} — {r['standard']}\n"
            f"- 카테고리(Category): {r.get('category', '')}\n"
            f"- 파라미터(Parameter): {r['parameter']}\n"
            f"- 기준(Threshold): {r['operator']} {r['threshold']} {unit}\n"
            f"- 설명: {r.get('description', '')}"
        )
        chunks.append({
            "id": f"requirement::{r['id']}",
            "text": text,
            "metadata": {
                "doc_type": "requirement",
                "source": "requirements.json",
                "standard": str(r.get("standard", "")),
                "category": str(r.get("category", "")),
                "parameter": str(r.get("parameter", "")),
                "threshold": float(r.get("threshold", 0)),
                "unit": str(unit),
                "requirement_id": str(r.get("id", "")),
            },
        })
    return chunks


def _load_glossary() -> list[dict]:
    if not config.GLOSSARY_PATH.exists():
        return []
    data = yaml.safe_load(config.GLOSSARY_PATH.read_text(encoding="utf-8")) or {}
    terms = data.get("terms", [])
    chunks: list[dict] = []
    for i, t in enumerate(terms):
        ko_all = [t.get("ko", "")] + (t.get("aliases") or [])
        en_all = [t.get("en", "")] + (t.get("en_aliases") or [])
        text = (
            f"한국어 용어: {', '.join(filter(None, ko_all))}\n"
            f"영문 용어(English): {', '.join(filter(None, en_all))}\n"
            f"관련 규격: {t.get('standard', '')}"
        )
        chunks.append({
            "id": f"glossary::{i}",
            "text": text,
            "metadata": {
                "doc_type": "glossary",
                "source": "ko_en_terms.yaml",
                "category": str(t.get("category", "general")),
                "standard": str(t.get("standard", "")),
            },
        })
    return chunks


def load_all() -> list[dict]:
    """전체 소스를 청크 리스트로 로딩."""
    return _load_standards() + _load_components() + _load_requirements()


if __name__ == "__main__":
    from collections import Counter

    from src.utils import enable_utf8_stdout

    enable_utf8_stdout()
    docs = load_all()
    by_type = Counter(c["metadata"]["doc_type"] for c in docs)
    print(f"총 청크: {len(docs)}")
    for t, n in by_type.items():
        print(f"  {t}: {n}")
    print("\n예시 청크:")
    for c in docs[:2]:
        print(f"  [{c['id']}] {c['metadata']}")
        print(f"    {c['text'][:120]}...")
