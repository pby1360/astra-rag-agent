"""규격 raw 마크다운을 METHOD 단위로 분할하고 frontmatter 를 붙인다.

입력: data/standards/_raw/<STD>.md  (pdf_to_md.py 가 생성, 페이지 마커 `<!-- page N -->` 포함)
출력: data/standards/<std-slug>_method_<id>_<category>.md  (frontmatter + 본문)

분할 원리
  MIL-STD-810H / 883 은 모든 페이지 머리말·꼬리말에 `METHOD <번호>` 가 반복된다.
  각 페이지의 지배적 메서드 번호로 연속 페이지를 그룹핑해 메서드 1개 = 파일 1개로 만든다.
  목차(TOC) 페이지처럼 서로 다른 메서드 번호가 여럿 등장하는 페이지는 front-matter 로 보고 건너뛴다.

실행:
  python -m src.ingestion.split_standards --standard MIL-STD-883
  python -m src.ingestion.split_standards --all
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.utils import enable_utf8_stdout  # noqa: E402

RAW_MD_DIR = config.STANDARDS_DIR / "_raw"

METHOD_RE = re.compile(r"(?:METHOD|Method)\s+(\d{3,4}(?:\.\d+)?)")
PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+\d+\s*-->")

# 제목 키워드 -> 카테고리
CATEGORY_KEYWORDS = {
    "vibration": ["VIBRATION"],
    "shock": ["SHOCK"],
    "temperature": ["TEMPERATURE", "HIGH TEMPERATURE", "LOW TEMPERATURE", "THERMAL"],
    "humidity": ["HUMIDITY", "MOISTURE"],
    "pressure": ["PRESSURE", "ALTITUDE", "BAROMETRIC"],
    "corrosion": ["SALT", "FOG", "CORROSION", "IMMERSION"],
    "radiation": ["RADIATION", "IONIZING", "IONISING", "DOSE", "NEUTRON", "GAMMA"],
    "sand_dust": ["SAND", "DUST"],
    "acceleration": ["ACCELERATION"],
    "sealing": ["SEAL", "FINE LEAK", "GROSS LEAK", "HERMETIC"],
}

# 제거할 반복 노이즈 (정확히 일치하는 줄)
NOISE_EXACT = {
    "Downloaded from https://www.everyspec.com",
    "This page intentionally left blank",
}

# pdfplumber 로 변환된, METHOD 구조를 갖는 규격
METHOD_STANDARDS = {
    "MIL-STD-883": "MIL-STD-883.md",
    "MIL-STD-810H": "MIL-STD-810H.md",
}

# 도메인(환경·기계·방사선) 관련 메서드 번호 범위만 유지.
# 883 의 3000(디지털)·4000(선형)·5000(시험절차) 전기시험 계열은 교차검증과 무관해 제외.
KEEP_RANGES = {
    "MIL-STD-883": [(1000, 2999)],   # 1000=환경, 2000=기계(진동·충격·가속)
    "MIL-STD-810H": [(500, 599)],    # 전 환경시험 메서드
}


def _in_keep_range(standard: str, method_id: str) -> bool:
    ranges = KEEP_RANGES.get(standard)
    if not ranges:
        return True
    base = int(method_id.split(".")[0])
    return any(lo <= base <= hi for lo, hi in ranges)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _split_pages(raw: str) -> list[str]:
    """raw 마크다운을 페이지 단위 텍스트 리스트로 분할."""
    parts = PAGE_MARKER_RE.split(raw)
    return [p for p in parts if p.strip()]


def _page_method(page: str) -> str | None:
    """페이지의 지배적 메서드 번호. 서로 다른 번호가 4개 이상이면 TOC 로 보고 None."""
    ids = METHOD_RE.findall(page)
    if not ids:
        return None
    counts = Counter(ids)
    if len(counts) >= 4:  # 목차/색인 페이지
        return None
    return counts.most_common(1)[0][0]


_TITLE_DENY = ("CONTENTS", "INTENTIONALLY", "PARAGRAPH", "TABLE", "FIGURE", "PAGE")


def _extract_title(text: str, method_id: str) -> str:
    """`METHOD <id>` 다음 줄의 실제 제목을 추출 (줄 단위 스캔).

    810H 는 `METHOD 514.8 / METHOD 514.8 / VIBRATION` 처럼 METHOD 줄이 반복되고
    각 메서드가 목차 페이지로 시작하므로, METHOD 줄마다 다음 줄을 보며
    목차/빈페이지/표준ID/날짜가 아닌 첫 실제 제목을 찾는다.
    """
    mre = re.compile(rf"^(?:METHOD|Method)\s+{re.escape(method_id)}\b")
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        if not mre.match(lines[i].strip()):
            continue
        cand = lines[i + 1].strip()
        if not cand or not re.search(r"[A-Za-z]{3,}", cand):
            continue
        if re.match(r"^\d", cand) or re.match(r"^MIL-STD", cand):
            continue
        if re.match(r"^(METHOD|Method)\b", cand):
            continue
        if any(w in cand.upper() for w in _TITLE_DENY):
            continue
        return cand
    return ""


def _category(title: str) -> str:
    up = title.upper()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in up for kw in kws):
            return cat
    return "general"


def _clean(text: str, standard: str) -> str:
    """반복 노이즈(everyspec 안내, 표준ID 머리말, 페이지번호 꼬리말)를 제거."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s in NOISE_EXACT:
            continue
        if re.fullmatch(rf"{re.escape(standard)}[A-Z]?", s):  # "MIL-STD-883F" 머리말
            continue
        out.append(line)
    # 3줄 이상 연속 빈 줄 축약
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _group_methods(pages: list[str]) -> list[tuple[str, list[str]]]:
    """연속 페이지를 메서드 번호로 그룹핑. [(method_id, [page,...]), ...]"""
    groups: list[tuple[str, list[str]]] = []
    for page in pages:
        mid = _page_method(page)
        if mid is None:
            continue
        if groups and groups[-1][0] == mid:
            groups[-1][1].append(page)
        else:
            groups.append((mid, [page]))
    return groups


def split_standard(standard: str, raw_path: Path, min_chars: int = 400) -> list[Path]:
    raw = raw_path.read_text(encoding="utf-8")
    pages = _split_pages(raw)
    groups = _group_methods(pages)

    std_slug = _slug(standard)
    written: list[Path] = []
    seen_base: set[str] = set()
    for mid, group_pages in groups:
        if not _in_keep_range(standard, mid):
            continue
        base = mid.split(".")[0]  # 개정번호 무시 (514.8 와 514 를 동일 취급)
        if base in seen_base:  # 첫 그룹만 (교차참조로 생기는 중복/비연속 재등장 제거)
            continue
        body = _clean("\n".join(group_pages), standard)
        if len(body) < min_chars:
            continue
        seen_base.add(base)
        title = _extract_title("\n".join(group_pages), mid)
        category = _category(title)
        heading = f"# {standard} Method {mid}" + (f" — {title}" if title else "")
        frontmatter = (
            "---\n"
            f"standard: {standard}\n"
            f'method: "{mid}"\n'
            f"category: {category}\n"
            "language: en\n"
            "---\n\n"
        )
        out_name = f"{std_slug}_method_{mid.replace('.', '_')}_{category}.md"
        out_path = config.STANDARDS_DIR / out_name
        out_path.write_text(frontmatter + heading + "\n\n" + body + "\n", encoding="utf-8")
        written.append(out_path)
    return written


def main() -> None:
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="규격 raw 마크다운을 METHOD 단위로 분할")
    ap.add_argument("--standard", help="표준명 (예: MIL-STD-883)")
    ap.add_argument("--input", help="raw 마크다운 경로 (생략 시 _raw/<표준>.md)")
    ap.add_argument("--all", action="store_true", help="METHOD_STANDARDS 전체 분할")
    args = ap.parse_args()

    targets: list[tuple[str, Path]] = []
    if args.all:
        for std, fname in METHOD_STANDARDS.items():
            targets.append((std, RAW_MD_DIR / fname))
    elif args.standard:
        raw = Path(args.input) if args.input else RAW_MD_DIR / f"{args.standard}.md"
        targets.append((args.standard, raw))
    else:
        ap.print_help()
        return

    for std, raw in targets:
        if not raw.exists():
            print(f"[건너뜀] {std}: {raw} 없음")
            continue
        written = split_standard(std, raw)
        print(f"[{std}] {len(written)}개 메서드 파일 생성")
        for p in written:
            print(f"  - {p.name}")


if __name__ == "__main__":
    main()
