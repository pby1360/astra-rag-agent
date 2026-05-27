"""공용 유틸리티."""
from __future__ import annotations

import sys


def enable_utf8_stdout() -> None:
    """Windows 콘솔에서 한글이 깨지지 않도록 stdout/stderr 를 UTF-8 로 강제한다.

    Python 은 Windows 에서 기본적으로 로케일 인코딩(예: cp949)으로 출력하므로,
    UTF-8 로 디코딩하는 환경에서 한글이 mojibake 로 깨진다. 모든 CLI 진입점에서
    가장 먼저 호출한다. (Streamlit UI 는 브라우저가 UTF-8 을 처리하므로 불필요.)
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass
