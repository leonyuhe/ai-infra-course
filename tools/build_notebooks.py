#!/usr/bin/env python
"""把 percent 格式的 .py 章节源文件构建为 .ipynb。

约定：
    # %% [markdown]   -> markdown cell（每行以 "# " 开头，构建时去掉）
    # %%               -> code cell
其余行原样进入 cell。

用法：
    python tools/build_notebooks.py                 # 构建全部
    python tools/build_notebooks.py --only 03       # 只构建 03_* 开头
    python tools/build_notebooks.py --check         # 只做格式校验，不写文件
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
SEP = re.compile(r"^# %%(.*)$")


def split_cells(text: str) -> list[tuple[str, list[str]]]:
    cells: list[tuple[str, list[str]]] = []
    kind = "code"
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        while buf and not buf[0].strip():
            buf.pop(0)
        while buf and not buf[-1].strip():
            buf.pop()
        if buf:
            cells.append((kind, buf))
        buf = []

    for line in text.splitlines():
        m = SEP.match(line)
        if m:
            flush()
            kind = "markdown" if "[markdown]" in m.group(1) else "code"
        else:
            buf.append(line)
    flush()
    return cells


def strip_markdown(lines: list[str]) -> str:
    out = []
    for line in lines:
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def build_notebook(path: Path) -> dict:
    cells = split_cells(path.read_text(encoding="utf-8"))
    nb_cells = []
    for idx, (kind, lines) in enumerate(cells):
        if kind == "markdown":
            nb_cells.append(
                {
                    "cell_type": "markdown",
                    "id": f"cell-{idx}",
                    "metadata": {},
                    "source": strip_markdown(lines).splitlines(keepends=True),
                }
            )
        else:
            nb_cells.append(
                {
                    "cell_type": "code",
                    "id": f"cell-{idx}",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": "\n".join(lines),
                }
            )
    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "jupytext": {
                "cell_metadata_filter": "-all",
                "formats": "py:percent",
                "text_representation": {"extension": ".py", "format_name": "percent"},
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="章节编号前缀，如 03 04")
    ap.add_argument("--check", action="store_true", help="只校验不写文件")
    args = ap.parse_args()

    sources = sorted(NB_DIR.glob("[0-9][0-9]_*.py"))
    if args.only:
        sources = [p for p in sources if p.stem[:2] in args.only]
    if not sources:
        print("没有找到章节源文件（notebooks/NN_*.py）")
        return 1

    try:
        import nbformat  # noqa: F401

        have_nbformat = True
    except Exception:
        have_nbformat = False

    rows = []
    for src in sources:
        nb = build_notebook(src)
        if have_nbformat:
            import nbformat

            nbformat.validate(nbformat.from_dict(nb))
        out = src.with_suffix(".ipynb")
        if not args.check:
            out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        n_md = sum(c["cell_type"] == "markdown" for c in nb["cells"])
        n_code = sum(c["cell_type"] == "code" for c in nb["cells"])
        rows.append((src.stem, len(nb["cells"]), n_md, n_code, "ok" if have_nbformat else "no-nbformat"))

    print(f"{'chapter':40} {'cells':>6} {'md':>4} {'code':>5}  {'validated'}")
    for r in rows:
        print(f"{r[0]:40} {r[1]:>6} {r[2]:>4} {r[3]:>5}  {r[4]}")
    if args.check:
        print("\n（--check：未写入 .ipynb）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
