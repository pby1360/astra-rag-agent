"""PDF -> 마크다운 변환 (하이브리드).

- pdfplumber 경로 (무료, 로컬): 표가 적은 규격/데이터시트. API 키 불필요.
- LlamaParse 경로 (LLAMA_CLOUD_API_KEY 필요): 표가 많은 핵심 문서(MIL-STD-810H).

출력은 data/standards/_raw/<name>.md 로 저장된다. 이후 split_standards.py 가
이 raw 마크다운을 METHOD 단위로 분할하고 frontmatter 를 붙인다.

실행:
  # pdfplumber 로 표 적은 규격 일괄 변환 (키 불필요)
  python -m src.ingestion.pdf_to_md --plumber

  # LlamaParse 로 810H 변환 (키 필요)
  python -m src.ingestion.pdf_to_md --llama MIL-STD-810H.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.utils import enable_utf8_stdout  # noqa: E402

RAW_MD_DIR = config.STANDARDS_DIR / "_raw"

# pdfplumber 로 처리할 문서 (표가 적거나 LlamaParse 크레딧을 아낄 문서)
PLUMBER_DOCS = [
    "MIL-STD-883.pdf",
    "MIL-STD-461G.pdf",
    "SMC-S-016.pdf",
    "TPS7H1101-SP.pdf",
    "SAMV71Q21RT.pdf",
]
# LlamaParse 권장 문서 (다단 표 다수)
LLAMA_DOCS = ["MIL-STD-810H.pdf"]


def _table_to_md(table: list[list[str | None]]) -> str:
    """pdfplumber 가 추출한 표(행 리스트)를 마크다운 표로 변환."""
    rows = [
        [(cell or "").replace("\n", " ").strip() for cell in row]
        for row in table
        if row is not None
    ]
    rows = [r for r in rows if any(c for c in r)]  # 빈 행 제거
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]  # 길이 통일
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def convert_with_pdfplumber(pdf_path: Path, out_path: Path) -> int:
    """pdfplumber 로 PDF 를 페이지별 텍스트 + 표 마크다운으로 변환. 페이지 수 반환."""
    import pdfplumber

    parts: list[str] = [f"# {pdf_path.stem}\n"]
    with pdfplumber.open(pdf_path) as pdf:
        n = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            parts.append(f"\n<!-- page {i} -->\n")
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)
            for table in page.extract_tables():
                md = _table_to_md(table)
                if md:
                    parts.append("\n" + md + "\n")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return n


def convert_with_llamaparse(
    pdf_path: Path, out_path: Path, target_pages: str | None = None
) -> None:
    """LlamaParse 로 PDF 를 의미론적 마크다운으로 변환. LLAMA_CLOUD_API_KEY 필요.

    target_pages: "0-4" 또는 "0,5,10" 형식. None 이면 전체. 크레딧 절약·검증용.
    """
    if not config.LLAMA_CLOUD_API_KEY:
        raise RuntimeError("LLAMA_CLOUD_API_KEY 가 설정되지 않았습니다. .env 를 확인하세요.")
    from llama_parse import LlamaParse

    kwargs = {"api_key": config.LLAMA_CLOUD_API_KEY, "result_type": "markdown"}
    if target_pages:
        # "0-4" -> "0,1,2,3,4" (LlamaParse 는 콤마 구분 페이지 목록을 받음)
        if "-" in target_pages:
            lo, hi = (int(x) for x in target_pages.split("-"))
            kwargs["target_pages"] = ",".join(str(p) for p in range(lo, hi + 1))
        else:
            kwargs["target_pages"] = target_pages

    parser = LlamaParse(**kwargs)
    docs = parser.load_data(str(pdf_path))
    md = "\n\n".join(d.text for d in docs)
    out_path.write_text(md, encoding="utf-8")


def main() -> None:
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="PDF -> 마크다운 변환")
    ap.add_argument("--plumber", action="store_true", help="pdfplumber 로 표 적은 문서 일괄 변환")
    ap.add_argument("--llama", metavar="FILE", help="LlamaParse 로 지정 PDF 변환 (키 필요)")
    ap.add_argument("--pages", metavar="RANGE", help="LlamaParse 페이지 범위 (예: 0-4). 검증·크레딧 절약용")
    ap.add_argument("--suffix", metavar="S", default="", help="출력 파일명 접미사 (검증 결과 분리용)")
    args = ap.parse_args()

    RAW_MD_DIR.mkdir(parents=True, exist_ok=True)

    if args.llama:
        src = config.RAW_PDF_DIR / args.llama
        out = RAW_MD_DIR / (src.stem + args.suffix + ".md")
        print(f"[LlamaParse] {src.name} -> {out.name} (pages={args.pages or 'all'})")
        convert_with_llamaparse(src, out, target_pages=args.pages)
        print(f"  완료: {out.stat().st_size/1024:.0f} KB")
        return

    if args.plumber:
        for name in PLUMBER_DOCS:
            src = config.RAW_PDF_DIR / name
            if not src.exists():
                print(f"[건너뜀] {name} 없음")
                continue
            out = RAW_MD_DIR / (src.stem + ".md")
            print(f"[pdfplumber] {name} 변환 중...")
            pages = convert_with_pdfplumber(src, out)
            print(f"  완료: {pages}p -> {out.name} ({out.stat().st_size/1024:.0f} KB)")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
