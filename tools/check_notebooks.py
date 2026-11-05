#!/usr/bin/env python
"""章节结构检查（STYLE_GUIDE.md 第 2/6/8 节的机检版）。

用法：
    python tools/check_notebooks.py --all
    python tools/check_notebooks.py --only 03
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
SECTIONS = ["问题与直觉", "从零实现", "实测", "对照工业实现", "练习与小结"]
MAX_CELL_LINES = 45


def check(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    stem = path.stem
    ch = stem[:2]
    problems: list[str] = []

    # 1) 五段式标题（编号不强制连续，但必须有这五个部分）
    for name in SECTIONS:
        pattern = rf"^# ## {ch}\.\d+\s*{name}"
        if not re.search(pattern, text, re.M):
            problems.append(f"缺少小节：## {ch}.x {name}")

    # 2) 头部元信息
    for key in ["预计用时", "运行环境", "前置章节"]:
        if key not in text:
            problems.append(f"头部表格缺少「{key}」")

    # 3) 概念图 + 实测图
    if not re.search(rf"assets/figures/ch{ch}_[a-z0-9_]+\.svg", text):
        problems.append(f"未引用概念图 assets/figures/ch{ch}_*.svg")
    if "save(" not in text:
        problems.append("未调用 aiinfra.plotting.save 生成实测图")

    # 4) 断言与数字表格
    if text.count("assert ") < 1:
        problems.append("没有任何 assert（关键不变量必须有断言）")
    if "DataFrame" not in text and "to_string" not in text:
        problems.append("没有数字表格（请用 pandas.DataFrame 展示实测结果）")

    # 5) 练习
    stars = text.count("⭐")
    if stars < 3:
        problems.append(f"练习数量不足：只有 {stars} 个 ⭐，至少 3 题")

    # 6) 单元格大小
    cells = re.split(r"^# %%.*$", text, flags=re.M)
    for idx, cell in enumerate(cells[1:], start=1):
        n = len([ln for ln in cell.splitlines() if ln.strip()])
        if n > MAX_CELL_LINES:
            problems.append(f"第 {idx} 个 cell 有 {n} 行（上限 {MAX_CELL_LINES}），请拆分")

    # 7) 禁词
    for bad in ["TODO", "FIXME", "待补充", "XXX"]:
        if bad in text:
            problems.append(f"包含未完成标记：{bad}")

    # 8) 文件头注释
    if not stem[0].isdigit():
        problems.append("文件名需以两位章节号开头，如 03_paged_attention.py")

    # 9) 图片内文字必须是英文（中文在无 CJK 字体的渲染器里会变成豆腐块）
    CJK = re.compile(r"[\u4e00-\u9fff]")
    for svg in sorted((ROOT / "assets" / "figures").glob(f"ch{ch}_*.svg")):
        if CJK.search(svg.read_text(encoding="utf-8")):
            problems.append(f"SVG 含中文，请改为英文：{svg.name}")
    for line in text.splitlines():
        if re.search(r"(set_title|set_xlabel|set_ylabel|suptitle|annotate|label\s*=)", line) and CJK.search(line):
            problems.append(f"matplotlib 图内文字含中文，请改为英文：{line.strip()[:60]}")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not args.all and not args.only:
        print("请指定 --all 或 --only NN")
        return 1

    sources = sorted(NB_DIR.glob("[0-9][0-9]_*.py"))
    if args.only:
        sources = [p for p in sources if p.stem[:2] in args.only]
    if not sources:
        print("没有找到章节源文件")
        return 1

    failed = 0
    for p in sources:
        problems = check(p)
        if problems:
            failed += 1
            print(f"FAIL {p.stem}")
            for item in problems:
                print(f"  - {item}")
        else:
            print(f"PASS {p.stem}")
    print(f"\n{len(sources) - failed}/{len(sources)} 章通过结构检查")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
