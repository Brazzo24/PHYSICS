"""Regenerate the notebooks from _build/nbXX.py (percent format).

    python build_notebooks.py              # write .ipynb files
    python build_notebooks.py --execute    # ... and run them
    python build_notebooks.py 03 --execute # only notebook 03
"""
import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = Path(__file__).parent


def parse(src: str):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if kind == "md":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            cells.append(new_markdown_cell(text))
        elif text.strip():
            cells.append(new_code_cell(text))

    for line in src.splitlines():
        if line.startswith("# %% [markdown]"):
            flush(); kind, buf = "md", []
        elif line.startswith("# %%"):
            flush(); kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def build(path: Path, execute=False):
    nb = new_notebook(cells=parse(path.read_text()))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = HERE / (path.stem.split("_", 1)[1] + ".ipynb")
    if execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, out)
    print("wrote", out.name)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ex = "--execute" in sys.argv
    for p in sorted((HERE / "_build").glob("nb*.py")):
        if not args or any(p.stem[2:4] == a for a in args):
            build(p, ex)
