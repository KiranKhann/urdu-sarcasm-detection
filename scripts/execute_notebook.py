# -*- coding: utf-8 -*-
"""Execute the assignment notebook in place with per-cell progress logging and a save after every
cell (so a crash or timeout keeps the outputs produced so far).

    python scripts/execute_notebook.py [notebook_path] [--timeout SECONDS]
"""
import argparse
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("path", nargs="?", default=os.path.join(ROOT, "notebooks", "Urdu_Sarcasm_Detection.ipynb"))
ap.add_argument("--timeout", type=int, default=5400)
a = ap.parse_args()

nb = nbformat.read(a.path, as_version=4)
nb_dir = os.path.dirname(os.path.abspath(a.path))
t_start = time.time()
state = {"t_cell": None}


def on_start(cell=None, cell_index=None, **kw):
    state["t_cell"] = time.time()
    if cell is not None and cell.cell_type == "code":
        first = next((l for l in cell.source.split("\n") if l.strip() and not l.startswith("#")), "")[:90]
        print(f"[{time.time() - t_start:7.0f}s] cell {cell_index:2d} start : {first}", flush=True)


def on_done(cell=None, cell_index=None, **kw):
    if cell is not None and cell.cell_type == "code":
        print(f"[{time.time() - t_start:7.0f}s] cell {cell_index:2d} done  ({time.time() - state['t_cell']:.0f}s)", flush=True)
        nbformat.write(nb, a.path)      # checkpoint


client = NotebookClient(nb, timeout=a.timeout, kernel_name="python3", resources={"metadata": {"path": nb_dir}},
                        on_cell_start=on_start, on_cell_executed=on_done, allow_errors=False)
try:
    client.execute()
finally:
    nbformat.write(nb, a.path)
errs = [(i, o.get("ename")) for i, c in enumerate(nb.cells) if c.cell_type == "code"
        for o in c.get("outputs", []) if o.output_type == "error"]
print(f"finished in {time.time() - t_start:.0f}s; errors: {errs}", flush=True)
sys.exit(1 if errs else 0)
