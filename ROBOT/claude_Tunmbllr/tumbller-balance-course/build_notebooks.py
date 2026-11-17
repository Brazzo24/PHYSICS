"""Regenerate (and optionally execute) all notebooks from the sources in _build/.

    python build_notebooks.py            # write notebooks
    python build_notebooks.py --execute  # write and run them (outputs stored)

Edit the _build/nbXX.py files rather than the .ipynb files.
"""
import importlib
import pathlib
import sys

import nbformat as nbf

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE / "_build"))

MODULES = ["nb01", "nb02", "nb03", "nb04"]


def main(execute=False, only=None):
    for mod in MODULES:
        if only and mod not in only:
            continue
        nb = importlib.import_module(mod).nb
        path = HERE / f"{nb.name}.ipynb"
        book = nb.notebook()
        if execute:
            from nbclient import NotebookClient
            NotebookClient(book, timeout=900, kernel_name="python3",
                           resources={"metadata": {"path": str(HERE)}}).execute()
            for cell in book.cells:          # sliders/animations: run them live
                if {"interactive", "animation"} & set(cell.metadata.get("tags", [])):
                    cell.outputs = []
                    cell.execution_count = None
        nbf.write(book, path)
        print("wrote", path.name)


if __name__ == "__main__":
    main(execute="--execute" in sys.argv,
         only=[a for a in sys.argv[1:] if a.startswith("nb")])
