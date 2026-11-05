#!/usr/bin/env python
"""无头执行 .ipynb，验证章节真的能跑通。

不依赖 jupyter kernel：直接按顺序 exec 每个 code cell，共享同一个命名空间。
这足以发现未定义变量、API 变更、超时与异常；图片由 aiinfra.plotting.save 落盘。

用法：
    python tools/execute_notebooks.py --all
    python tools/execute_notebooks.py --only 00 01
    python tools/execute_notebooks.py --only 01 --timeout 180
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"


def run_notebook(path: Path, timeout: float) -> tuple[bool, float, str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ns: dict = {"__name__": "__main__", "__file__": str(path)}
    out = io.StringIO()
    t0 = time.perf_counter()
    ok = True
    err = ""
    for i, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        elapsed = time.perf_counter() - t0
        if elapsed > timeout:
            ok, err = False, f"超时：已用 {elapsed:.1f}s > {timeout}s（cell #{i + 1}）"
            break
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(src, f"{path.name}:cell{i + 1}", "exec"), ns)
        except Exception:
            ok = False
            tb = traceback.format_exc()
            err = f"cell #{i + 1} 失败：\n{tb}\n--- 源码 ---\n{src}"
            break
        finally:
            out.write(buf.getvalue())
    dt = time.perf_counter() - t0
    if "matplotlib" in sys.modules:
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass
    return ok, dt, err or out.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--timeout", type=float, default=300.0, help="单章超时秒数，默认 300")
    ap.add_argument("--tail", type=int, default=15)
    args = ap.parse_args()

    if not args.all and not args.only:
        print("请指定 --all 或 --only NN")
        return 1

    os.chdir(NB_DIR)
    os.environ["AIINFRA_HEADLESS"] = "1"
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplcache"))
    sys.path.insert(0, str(ROOT))

    paths = sorted(NB_DIR.glob("[0-9][0-9]_*.ipynb"))
    if args.only:
        paths = [p for p in paths if p.stem[:2] in args.only]
    if not paths:
        print("没有找到 .ipynb，请先运行 tools/build_notebooks.py")
        return 1

    failures = []
    print(f"{'chapter':40} {'status':8} {'seconds':>8}")
    print("-" * 62)
    for p in paths:
        ok, dt, log = run_notebook(p, args.timeout)
        print(f"{p.stem:40} {'PASS' if ok else 'FAIL':8} {dt:>8.1f}")
        if ok:
            tail = [ln for ln in log.strip().splitlines() if ln.strip()][-args.tail :]
            for ln in tail:
                print(f"    | {ln}")
        else:
            failures.append((p.stem, log))
    print("-" * 62)
    if failures:
        for name, err in failures:
            print(f"\n### {name}\n{err}")
        return 1
    print(f"全部通过（{len(paths)} 章）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
