import py_compile
from pathlib import Path


def test_streamlit_entrypoints_compile():
    paths = [
        Path("app.py"),
        Path("streamlit_app.py"),
        *sorted(Path("pages").glob("*.py")),
    ]
    for path in paths:
        py_compile.compile(str(path), doraise=True)
