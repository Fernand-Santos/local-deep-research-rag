"""Allow running as python -m app (delegates to streamlit run main.py at project root)."""
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import streamlit.web.cli as stcli

    sys.argv = ["streamlit", "run", str(root / "main.py"), "--", *sys.argv[1:]]
    stcli.main()
