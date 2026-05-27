"""중앙 로깅 설정.

모든 모듈은 logging.getLogger(__name__) 으로 로거를 가져다 쓴다.
앱 진입점(app.py, run_graph.py 등)에서 setup_logging() 을 1회 호출한다.

Streamlit 환경에서는 stdout/stderr 가 가로채지므로 파일 로그를 사용한다.
  - 로그 파일: astra_rag.log (프로젝트 루트)
  - 실시간 확인: PowerShell > Get-Content -Wait astra_rag.log
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parents[1] / "astra_rag.log"


def setup_logging(level: int = logging.INFO) -> None:
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger("src")
    # 기존 핸들러 제거 후 재설정 (Streamlit 재로드 시 중복 방지)
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    root.setLevel(level)

    # 파일 핸들러 (Streamlit 환경에서도 동작)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # CLI 실행 시 stdout 에도 출력
    stderr_handler = logging.StreamHandler(sys.stdout)
    stderr_handler.setFormatter(fmt)
    root.addHandler(stderr_handler)
