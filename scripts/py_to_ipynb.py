"""
Conversor minimalista de un script "estilo Jupytext" (celdas separadas por
`# %%` y `# %% [markdown]`) a un notebook .ipynb valido (nbformat 4), sin
depender del paquete `nbformat` (no disponible en este entorno).

Uso:
    python scripts/py_to_ipynb.py notebooks_src/00_descarga_datos.py notebooks/00_descarga_datos.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def split_cells(source: str) -> list[dict]:
    lines = source.splitlines()
    cells = []
    current_type = "code"
    current_lines: list[str] = []

    def flush():
        text = "\n".join(current_lines).strip("\n")
        if text.strip() == "":
            return
        cells.append({"cell_type": current_type, "source": text})

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# %% [markdown]"):
            flush()
            current_lines = []
            current_type = "markdown"
            continue
        if stripped.startswith("# %%"):
            flush()
            current_lines = []
            current_type = "code"
            continue
        current_lines.append(line)
    flush()
    return cells


def to_notebook(cells: list[dict]) -> dict:
    nb_cells = []
    for c in cells:
        source = c["source"]
        if c["cell_type"] == "markdown":
            # quita el prefijo "# " de comentario de cada linea markdown
            text_lines = []
            for ln in source.split("\n"):
                if ln.startswith("# "):
                    text_lines.append(ln[2:])
                elif ln == "#":
                    text_lines.append("")
                else:
                    text_lines.append(ln)
            src_lines = [ln + "\n" for ln in text_lines]
            if src_lines:
                src_lines[-1] = src_lines[-1].rstrip("\n")
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src_lines})
        else:
            src_lines = [ln + "\n" for ln in source.split("\n")]
            if src_lines:
                src_lines[-1] = src_lines[-1].rstrip("\n")
            nb_cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": src_lines,
                }
            )

    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    source = src_path.read_text(encoding="utf-8")
    cells = split_cells(source)
    nb = to_notebook(cells)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {len(nb['cells'])} celdas -> {out_path}")


if __name__ == "__main__":
    main()
