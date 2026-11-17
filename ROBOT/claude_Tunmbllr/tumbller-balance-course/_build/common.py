"""Tiny helpers shared by the notebook source files in _build/."""
import nbformat as nbf


class NB:
    def __init__(self, name):
        self.name = name
        self.cells = []

    def md(self, s):
        self.cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))

    def code(self, s, tags=None):
        c = nbf.v4.new_code_cell(s.strip("\n"))
        if tags:
            c.metadata["tags"] = list(tags)
        self.cells.append(c)

    def notebook(self):
        nb = nbf.v4.new_notebook()
        nb.cells = self.cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                     "language": "python"}
        return nb


HEAD = '''
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
import tumbller as tb
from tumbller import SERIES, INK, INK2, GRID, tidy
tb.use_style()
p = tb.Params()
'''
