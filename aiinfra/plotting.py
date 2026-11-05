"""统一图表风格。

关键点：
- import 之前就把 ``MPLCONFIGDIR`` 指到 ``outputs/.mplcache``，避免默认目录不可写。
- 无头执行（tools/execute_notebooks.py）时设置 ``AIINFRA_HEADLESS=1``，强制 Agg 后端。
- 所有图统一存到 ``outputs/<name>.png``，300 dpi。
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUTS / ".mplcache"))

import matplotlib  # noqa: E402

if os.environ.get("AIINFRA_HEADLESS") == "1":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# 与 STYLE_GUIDE.md 第 4 节一致的配色
COLORS = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "accent": "#F59E0B",
    "success": "#10B981",
    "danger": "#EF4444",
    "text": "#111827",
    "muted": "#6B7280",
    "bg": "#F3F4F6",
}

_CJK_CANDIDATES = ["PingFang SC", "Noto Sans CJK SC", "Noto Sans SC", "Hiragino Sans GB", "Microsoft YaHei"]


def _pick_cjk_font() -> str | None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    return None


def setup_style() -> None:
    cjk = _pick_cjk_font()
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.2),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "font.family": [cjk] if cjk else ["DejaVu Sans"],
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#D1D5DB",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
        }
    )


setup_style()


def save(fig, name: str, close: bool = False) -> Path:
    """保存到 outputs/<name>.png，返回路径（Notebook 里会打印出来）。"""
    path = OUTPUTS / f"{name}.png"
    fig.savefig(path)
    print(f"figure saved -> {path.relative_to(ROOT)}")
    if close:
        plt.close(fig)
    return path


def annotate(ax, text: str, xy, xytext, color: str | None = None) -> None:
    """统一风格的箭头标注。"""
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        color=color or COLORS["text"],
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color=color or COLORS["muted"], lw=1.2),
    )
