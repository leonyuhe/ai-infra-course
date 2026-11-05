#!/usr/bin/env python
"""把 15 章 Notebook 构建成可在线阅读的静态站点，可选导出一本 PDF 讲义。

流程：
1. 用 Jupyter kernel 重新执行每章（inline 后端，图会进 HTML）；
2. nbconvert 导出每章 HTML（图片以 base64 内嵌，离线可看）；
3. 生成 index.html（目录 + 每章一句话简介）与 style.css；
4. 复制 assets/figures，供 markdown 里的相对路径引用；
5. 可选 --pdf：把所有章节拼成一本 HTML，用 headless Chrome 打印成 PDF。

用法：
    python tools/build_site.py --out _site
    python tools/build_site.py --out _site --pdf dist/ai-infra-course-v0.1.0.pdf
    python tools/build_site.py --only 00 01 --no-execute   # 快速预览（不跑代码）
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
FIG_DIR = ROOT / "assets" / "figures"

# 必须在 kernel 启动前设置：让 matplotlib 用 inline 后端，图才会进 notebook 输出
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplcache"))

TITLE = "AI Infra 从零构建"
SUBTITLE = "从 KV cache 到 vLLM：15 章手写推理基础设施"
VERSION = "v0.1.0"

CSS = """
:root { --fg:#111827; --muted:#6B7280; --accent:#2563EB; --bg:#FFFFFF; --code:#F3F4F6; }
* { box-sizing: border-box; }
html { font-size: 16px; }
body {
  margin: 0; color: var(--fg); background: var(--bg);
  font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB",
               "Noto Sans CJK SC", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.75;
}
.wrap { max-width: 880px; margin: 0 auto; padding: 48px 24px 80px; }
h1 { font-size: 34px; line-height: 1.3; margin: 0 0 12px; letter-spacing: -0.02em; }
h2 { font-size: 24px; margin-top: 48px; padding-top: 16px; border-top: 1px solid #E5E7EB; }
h3 { font-size: 19px; margin-top: 32px; }
p.lead { color: var(--muted); font-size: 18px; margin: 0 0 8px; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code { background: var(--code); padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
pre { background: var(--code); padding: 16px; border-radius: 8px; overflow-x: auto; line-height: 1.5; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.95em; }
th, td { border: 1px solid #E5E7EB; padding: 10px 12px; text-align: left; vertical-align: top; }
th { background: #F9FAFB; }
blockquote { border-left: 4px solid var(--accent); margin: 20px 0; padding: 4px 20px; color: #374151; background: #F8FAFC; }
img { max-width: 100%; height: auto; }
.cover { float: right; width: 300px; margin: 4px 0 28px 36px; border-radius: 6px;
         box-shadow: 0 18px 40px rgba(0,0,0,.18); }
@media (max-width: 720px) { .cover { float: none; width: 68%; display: block; margin: 0 auto 24px; } }
.input_prompt, .output_prompt, .prompt,
.jp-InputPrompt, .jp-OutputPrompt { display: none !important; }
nav.topbar { position: sticky; top: 0; z-index: 10; background: rgba(255,255,255,.92);
             backdrop-filter: blur(8px); border-bottom: 1px solid #E5E7EB; }
nav.topbar .inner { max-width: 880px; margin: 0 auto; padding: 10px 24px; font-size: 14px;
                    display: flex; gap: 16px; align-items: center; }
nav.topbar a { color: var(--muted); }
nav.topbar a:hover { color: var(--accent); }
nav.topbar .spacer { flex: 1; }
.toc { list-style: none; padding: 0; margin: 24px 0 0; }
.toc li { border-bottom: 1px solid #F3F4F6; padding: 14px 0; }
.toc li a { font-size: 18px; font-weight: 600; }
.toc li p { margin: 4px 0 0; color: var(--muted); font-size: 14px; }
.btn { display: inline-block; background: var(--accent); color: #fff !important; padding: 10px 18px;
       border-radius: 8px; font-weight: 600; margin: 8px 8px 0 0; }
.btn:hover { text-decoration: none; opacity: .92; }
.meta { color: var(--muted); font-size: 14px; }
.banner { background:#FEF3C7; border:1px solid #F59E0B; color:#7C2D12; padding:10px 14px;
          border-radius:8px; font-size:13px; line-height:1.6; margin:0 0 28px; }
.banner a { color:#7C2D12; text-decoration:underline; }
@media print {
  @page { size: A4; margin: 15mm 13mm; }
  @page :first { margin: 0; }
  nav.topbar { display: none; }
  .wrap { max-width: none; padding: 0; }
  .chapter { break-before: page; }
  .chapter:first-of-type { break-before: auto; }
  a { color: inherit; text-decoration: none; }
  pre { white-space: pre-wrap; word-break: break-word; }
  h2 { break-after: avoid; }
  table, img, pre { break-inside: avoid; }
}
"""


# --------------------------------------------------------------------------- #
# Notebook 执行与导出
# --------------------------------------------------------------------------- #
def parse_meta(nb) -> tuple[str, str]:
    """从第一个 markdown cell 里取标题和一句话简介。"""
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        lines = cell.source.splitlines()
        title = next((ln.lstrip("# ").strip() for ln in lines if ln.startswith("# ")), "未命名章节")
        lead = next((ln.lstrip("> ").strip() for ln in lines if ln.startswith("> ")), "")
        return title, lead
    return "未命名章节", ""


def ensure_kernel() -> None:
    """确保 jupyter 能找到 python3 kernel（venv 内 ipykernel 已提供）。"""
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        specs = KernelSpecManager().find_kernel_specs()
        if "python3" not in specs:
            subprocess.run(
                [sys.executable, "-m", "ipykernel", "install", "--user", "--name", "python3"],
                check=True,
                capture_output=True,
            )
    except Exception as exc:  # pragma: no cover - 环境问题直接暴露
        print(f"[warn] kernel 检查失败：{exc}")


def execute_notebook(nb, timeout: int):
    from nbclient import NotebookClient

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(NB_DIR)}},
        allow_errors=False,
    )
    client.execute()
    return nb


def to_html(nb) -> str:
    from nbconvert import HTMLExporter

    exporter = HTMLExporter(template_name="lab")
    body, _ = exporter.from_notebook_node(nb)
    return body


# --------------------------------------------------------------------------- #
# 页面拼装
# --------------------------------------------------------------------------- #
def navbar(prev_href: str | None, next_href: str | None) -> str:
    left = '<a href="../index.html">← 目录</a>'
    right = []
    if prev_href:
        right.append(f'<a href="{html.escape(prev_href)}">上一章</a>')
    if next_href:
        right.append(f'<a href="{html.escape(next_href)}">下一章 →</a>')
    return f'<nav class="topbar"><div class="inner">{left}<span class="spacer"></span>{"".join(right)}</div></nav>'


BANNER = (
    '<div class="banner">⚠️ 在线阅读版由 GitHub Actions（Linux CPU）执行：'
    '带宽、tok/s 等硬件相关数字与 Mac 本地不同，但公式与趋势一致。'
    '本地复现见 <a href="https://github.com/leonyuhe/ai-infra-course#快速开始">README</a>。</div>'
)


def inject(body: str, *, nav: str, root_prefix: str, banner: str = "") -> str:
    head_add = f'<link rel="stylesheet" href="{root_prefix}style.css">'
    body = body.replace("</head>", head_add + "\n</head>", 1)
    body = re.sub(r"(<body[^>]*>)", r"\1\n" + nav + banner, body, count=1)
    return body


def md_inline(text: str) -> str:
    """把简介里常见的行内 markdown 转成 HTML（**粗体**、`code`），先转义再替换。"""
    out = html.escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out


def build_index(chapters: list[dict], pdf_name: str | None) -> str:
    items = "\n".join(
        f'<li><a href="notebooks/{c["stem"]}.html">{html.escape(c["title"])}</a>'
        f'<p>{md_inline(c["lead"])}</p></li>'
        for c in chapters
    )
    pdf_btn = (
        f'<a class="btn" href="{html.escape(pdf_name)}">下载 PDF 讲义</a>'
        if pdf_name
        else '<a class="btn" href="https://github.com/leonyuhe/ai-infra-course/releases">PDF 讲义（Release）</a>'
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title><link rel="stylesheet" href="style.css"></head>
<body><div class="wrap">
<img class="cover" src="assets/cover/ai-infra-course-cover.png" alt="课程封面">
<p class="meta">{VERSION} · 在线阅读版</p>
<h1>{TITLE}</h1>
<p class="lead">{SUBTITLE}</p>
<p>每章先给直觉与图，再从零手写最小实现，然后亲手测量，最后对照 vLLM 源码。
所有代码都在你的 Mac 上可运行。</p>
<p><a class="btn" href="https://github.com/leonyuhe/ai-infra-course">GitHub 仓库</a>{pdf_btn}</p>
<p class="meta">注：在线阅读版由 CI（Linux CPU）执行，硬件相关数字与 Mac 本地不同；趋势与公式一致。</p>
<h2 style="border:0;padding-top:24px">目录</h2>
<ul class="toc">{items}</ul>
</div></body></html>"""


def build_book(chapters: list[dict], chapter_html: dict[str, str]) -> str:
    """把各章 body 拼成一本连续 HTML（用于打印 PDF）。"""
    first = next(iter(chapter_html.values()))
    styles = "\n".join(re.findall(r"<style[^>]*>.*?</style>", first, flags=re.S))
    sections = []
    for i, c in enumerate(chapters):
        raw = chapter_html[c["stem"]]
        body = re.search(r"<body[^>]*>(.*)</body>", raw, flags=re.S)
        content = body.group(1) if body else raw
        content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.S)
        content = content.replace("../assets/", "assets/")
        sections.append(f'<section class="chapter">{content}</section>')
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{TITLE}</title>
{styles}
<style>{CSS}
.titlepage {{ min-height: 297mm; padding: 0; margin: 0; }}
.titlepage img {{ width: 100%; height: 297mm; object-fit: cover; display: block; margin: 0; }}
</style></head><body>
<section class="chapter titlepage">
  <img src="assets/cover/ai-infra-course-cover.png" alt="cover">
</section>
{"".join(sections)}
</body></html>"""


def chrome_binary() -> str:
    candidates = [
        os.environ.get("CHROME_BIN", ""),
        shutil.which("google-chrome") or "",
        shutil.which("google-chrome-stable") or "",
        shutil.which("chromium") or "",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise SystemExit("找不到 Chrome/Chromium，无法导出 PDF。请设置 CHROME_BIN。")


def print_pdf(book_html: Path, pdf_path: Path) -> None:
    tmp = book_html.parent / "_chrome-profile"
    cmd = [
        chrome_binary(),
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-breakpad",
        "--disable-crash-reporter",
        f"--crash-dumps-dir={tmp}/crashes",
        f"--user-data-dir={tmp}",
        "--virtual-time-budget=30000",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path.resolve()}",
        book_html.resolve().as_uri(),
    ]
    print("[pdf] 调用 Chrome 打印：", " ".join(cmd[1:]))
    subprocess.run(cmd, check=True, capture_output=True)
    if not pdf_path.exists():
        raise SystemExit(f"PDF 生成失败：{pdf_path}")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_site", help="输出目录（默认 _site）")
    ap.add_argument("--pdf", default=None, help="可选：PDF 输出路径，如 dist/course.pdf")
    ap.add_argument("--only", nargs="*", default=None, help="只构建指定章节，如 00 01")
    ap.add_argument("--no-execute", action="store_true", help="不执行代码，只导出静态 HTML（快速预览）")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--ci-banner", action="store_true", help="页面顶部显示 CI 执行提示（仅线上构建使用）")
    args = ap.parse_args()

    import nbformat

    ensure_kernel()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "notebooks").mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(parents=True, exist_ok=True)

    if FIG_DIR.exists():
        shutil.copytree(FIG_DIR, out / "assets" / "figures", dirs_exist_ok=True)
    cover_dir = ROOT / "assets" / "cover"
    if cover_dir.exists():
        shutil.copytree(cover_dir, out / "assets" / "cover", dirs_exist_ok=True)
    (out / "style.css").write_text(CSS, encoding="utf-8")

    sources = sorted(NB_DIR.glob("[0-9][0-9]_*.ipynb"))
    if args.only:
        sources = [p for p in sources if p.stem[:2] in args.only]
    if not sources:
        print("没有找到章节 ipynb，请先运行 tools/build_notebooks.py")
        return 1

    chapters, chapter_html = [], {}
    for i, path in enumerate(sources, start=1):
        print(f"[{i}/{len(sources)}] {path.stem} ...", flush=True)
        nb = nbformat.read(path, as_version=4)
        if not args.no_execute:
            nb = execute_notebook(nb, args.timeout)
        title, lead = parse_meta(nb)
        raw = to_html(nb)
        stem = path.stem
        chapters.append({"stem": stem, "title": title, "lead": lead})
        chapter_html[stem] = raw

        prev_href = f"{sources[i - 2].stem}.html" if i > 1 else None
        next_href = f"{sources[i].stem}.html" if i < len(sources) else None
        page = inject(raw, nav=navbar(prev_href, next_href), root_prefix="../",
                      banner=BANNER if args.ci_banner else "")
        (out / "notebooks" / f"{stem}.html").write_text(page, encoding="utf-8")

    pdf_name = None
    if args.pdf:
        pdf_path = Path(args.pdf)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        book = build_book(chapters, chapter_html)
        book_html = out / "_book.html"
        book_html.write_text(book, encoding="utf-8")
        print_pdf(book_html, pdf_path)
        pdf_name = pdf_path.name
        shutil.copy2(pdf_path, out / pdf_name)
        print(f"[pdf] 生成：{pdf_path}（{pdf_path.stat().st_size / 1024 / 1024:.1f} MB）")

    (out / "index.html").write_text(build_index(chapters, pdf_name), encoding="utf-8")
    print(f"\n站点已生成：{out}/index.html（{len(chapters)} 章）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
