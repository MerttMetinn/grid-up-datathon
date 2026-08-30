# -*- coding: utf-8 -*-
"""
29_build_notebook.py — percent formatlı .py kaynağını .ipynb'ye çevirir.

Kaynak gerçek Python olduğu için lint edilebilir, import edilebilir ve test
edilebilir; notebook ondan üretilir (tek doğruluk kaynağı).

Hücre ayracı:
    # %%              → kod hücresi
    # %% [markdown]   → markdown hücresi ("# " yorum öneki kaldırılır)

Kullanım: python scripts/29_build_notebook.py
Çıktı:    notebooks/gridup_leakfree_submission.ipynb
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "gridup_leakfree_submission.py"
DST = ROOT / "notebooks" / "gridup_leakfree_submission.ipynb"


def split_cells(text: str) -> list[tuple[str, list[str]]]:
    """(hücre_tipi, satırlar) listesi döner."""
    cells: list[tuple[str, list[str]]] = []
    kind, buf = None, []

    def flush():
        if kind is None:
            return
        while buf and not buf[-1].strip():
            buf.pop()
        if buf:
            cells.append((kind, list(buf)))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
            continue
        if kind is None:
            continue          # ilk ayraçtan önceki satırlar (coding cookie vb.)
        buf.append(line)
    flush()
    return cells


def demote_markdown(lines: list[str]) -> list[str]:
    """'# ' yorum önekini kaldırır; boş '#' satırını boş satıra çevirir."""
    out = []
    for ln in lines:
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln.rstrip() == "#":
            out.append("")
        else:
            out.append(ln)
    return out


def as_source(lines: list[str]) -> list[str]:
    """nbformat kaynak listesi: son satır hariç hepsi \\n ile biter."""
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def main() -> int:
    if not SRC.exists():
        print(f"kaynak yok: {SRC}")
        return 1

    cells = split_cells(SRC.read_text(encoding="utf-8"))
    nb_cells = []
    for kind, lines in cells:
        if kind == "markdown":
            lines = demote_markdown(lines)
            nb_cells.append({"cell_type": "markdown", "metadata": {},
                             "source": as_source(lines)})
        else:
            nb_cells.append({"cell_type": "code", "metadata": {},
                             "execution_count": None, "outputs": [],
                             "source": as_source(lines)})

    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    n_md = sum(1 for c in nb_cells if c["cell_type"] == "markdown")
    n_code = len(nb_cells) - n_md
    print(f"yazildi: {DST}")
    print(f"  {len(nb_cells)} hucre ({n_md} markdown · {n_code} kod)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
