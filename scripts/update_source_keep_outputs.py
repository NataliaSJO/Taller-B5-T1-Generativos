"""
Actualiza el 'source' de las celdas de un notebook YA EJECUTADO a partir de
su fuente .py (notebooks_src/), sin tocar 'outputs'/'execution_count' —
para cuando el cambio en el .py es solo de comentarios/documentacion (el
comportamiento no cambia, así que no hace falta re-ejecutar). Asume que el
.py no ha ganado ni perdido celdas respecto a la version ya ejecutada
(empareja por POSICION).

Uso:
    python scripts/update_source_keep_outputs.py notebooks_src/X.py notebooks/X.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py_to_ipynb import split_cells, to_notebook  # noqa: E402


def main():
    src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    source = src_path.read_text(encoding="utf-8")
    new_cells = to_notebook(split_cells(source))["cells"]

    old_nb = json.loads(out_path.read_text(encoding="utf-8"))
    old_cells = old_nb["cells"]

    if len(new_cells) != len(old_cells):
        print(
            f"ERROR: {src_path} tiene {len(new_cells)} celdas pero "
            f"{out_path} tiene {len(old_cells)} — la fuente gano/perdio celdas, "
            "no se puede emparejar por posicion. Reejecuta el notebook entero."
        )
        sys.exit(1)

    for old_c, new_c in zip(old_cells, new_cells):
        if old_c["cell_type"] != new_c["cell_type"]:
            print("ERROR: tipo de celda distinto en la misma posicion, aborta.")
            sys.exit(1)
        old_c["source"] = new_c["source"]

    out_path.write_text(json.dumps(old_nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK: source actualizado en {out_path} ({len(old_cells)} celdas), outputs intactos")


if __name__ == "__main__":
    main()
