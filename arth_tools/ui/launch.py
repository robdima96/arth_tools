# -*- coding: utf-8 -*-
"""Launch Streamlit on arth_tools.ui.app."""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    extra = list(sys.argv[1:] if argv is None else argv)
    try:
        from streamlit.web.cli import main as st_main
    except ImportError:
        print('Streamlit is not installed. Run: pip install -e ".[ui]"')
        return 1
    app = Path(__file__).resolve().parent / "app.py"
    sys.argv = ["streamlit", "run", str(app), *extra]
    st_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
