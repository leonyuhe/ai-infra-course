#!/usr/bin/env python
"""生成课程封面（致敬经典技术书版式，人物与出版社均为自有）。

素材：Ada Lovelace 肖像（Alfred Edward Chalon, 1840，公有领域，来自 Wikimedia Commons）。
画布按 A4 比例（1600x2263），可直接作为讲义 PDF 的整页封面。

用法：python tools/make_cover.py
输出：assets/cover/ai-infra-course-cover.png（1600x2263）
      assets/cover/ai-infra-course-cover-thumb.png（800x1131）
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs" / "cover_src" / "portrait.jpg"
OUT = ROOT / "assets" / "cover"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1600, 2263          # A4 比例（1 : 1.414）
PANEL_W = 850

PLUM = (59, 10, 28)
CREAM = (247, 242, 232)
YELLOW = (245, 197, 66)
RED = (200, 16, 46)
WHITE = (255, 255, 255)

FONT_ROCKWELL = "/System/Library/Fonts/Supplemental/Rockwell.ttc"
FONT_AVENIR = "/System/Library/Fonts/Avenir Next Condensed.ttc"
FONT_BODONI = "/System/Library/Fonts/Supplemental/Bodoni 72.ttc"
FONT_GEORGIA = "/System/Library/Fonts/Supplemental/Georgia.ttf"
FONT_CJK = "/System/Library/Fonts/STHeiti Medium.ttc"


def font(path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, index=index)


def text_width(draw, text: str, f, tracking: int = 0) -> float:
    if not tracking:
        return draw.textlength(text, font=f)
    return sum(draw.textlength(ch, font=f) for ch in text) + tracking * (len(text) - 1)


def draw_tracked(draw, xy, text, f, fill, tracking=0) -> None:
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=f, fill=fill, anchor="ls")
        x += draw.textlength(ch, font=f) + tracking


def build_background() -> Image.Image:
    bg = Image.new("RGB", (W, H), CREAM)
    draw = ImageDraw.Draw(bg)
    draw.rectangle((0, 0, PANEL_W, H), fill=PLUM)

    grad = Image.new("L", (1, H), 0)
    for y in range(H):
        grad.putpixel((0, y), int(14 * (y / H)))
    shade = Image.new("RGB", (PANEL_W, H), (0, 0, 0))
    bg.paste(shade, (0, 0), grad.resize((PANEL_W, H)))
    return bg


def build_portrait() -> tuple[Image.Image, Image.Image, tuple[int, int]]:
    im = Image.open(SRC).convert("RGB")
    im = im.crop((900, 230, 2200, 3260))          # 1300 x 3030
    cream = Image.new("RGB", im.size, CREAM)
    im = Image.blend(im, cream, 0.18)
    im = im.resize((770, 1795), Image.LANCZOS)

    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).ellipse((0, 0, im.width - 1, im.height - 1), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(2))
    return im, mask, (835, 290)


def draw_portrait(base: Image.Image) -> None:
    portrait, mask, (px, py) = build_portrait()

    shadow = Image.new("RGBA", (base.width, base.height), (0, 0, 0, 0))
    sh = Image.new("RGBA", portrait.size, (0, 0, 0, 0))
    sh.paste((0, 0, 0, 80), (0, 0), mask)
    shadow.paste(sh, (px + 18, py + 26))
    shadow = shadow.filter(ImageFilter.GaussianBlur(30))
    base.paste(shadow, (0, 0), shadow)
    base.paste(portrait, (px, py), mask)

    d = ImageDraw.Draw(base)
    box = (px, py, px + portrait.width - 1, py + portrait.height - 1)
    d.ellipse(box, outline=(201, 162, 39), width=5)
    d.ellipse((box[0] + 14, box[1] + 14, box[2] - 14, box[3] - 14), outline=(201, 162, 39), width=2)


def draw_ribbon(base: Image.Image) -> None:
    band_w, band_h = 1100, 150
    band = Image.new("RGBA", (band_w, band_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(band)
    d.rectangle((0, 0, band_w, band_h), fill=RED)
    f = font(FONT_AVENIR, 62, 0)
    tw = text_width(d, "FROM SCRATCH", f, tracking=8)
    draw_tracked(d, ((band_w - tw) / 2, 104), "FROM SCRATCH", f, WHITE, tracking=8)
    band = band.rotate(-38, resample=Image.BICUBIC, expand=True)
    base.paste(band, (W - band.width + 90, -70), band)


def draw_text(base: Image.Image) -> None:
    d = ImageDraw.Draw(base)

    draw_tracked(d, (108, 430), "BUILD AN", font(FONT_AVENIR, 54, 0), WHITE, tracking=8)

    f_title = font(FONT_ROCKWELL, 205, 2)
    draw_tracked(d, (108, 645), "AI", f_title, YELLOW, tracking=-4)
    draw_tracked(d, (108, 865), "Infra", f_title, YELLOW, tracking=-4)

    d.text((118, 1105), "from Scratch", font=font(FONT_BODONI, 104, 2), fill=YELLOW, anchor="ls")

    f_tag = font(FONT_AVENIR, 34, 2)
    d.text((118, 1180), "KV cache · paged attention ·", font=f_tag, fill=(233, 213, 219), anchor="ls")
    d.text((118, 1227), "continuous batching · vLLM", font=f_tag, fill=(233, 213, 219), anchor="ls")

    d.text((118, 1305), "15 章手写推理基础设施", font=font(FONT_CJK, 40, 1), fill=(233, 213, 219), anchor="ls")

    d.text((118, 1615), "leonyuhe", font=font(FONT_GEORGIA, 54), fill=WHITE, anchor="ls")
    d.text((118, 1669), "15 chapters · v0.1.1 · 2026", font=font(FONT_AVENIR, 28, 1), fill=(214, 190, 198), anchor="ls")

    bx, by = 118, 1775
    for i in range(3):
        x = bx + i * 44
        d.polygon([(x + 14, by), (x + 32, by), (x + 18, by + 52), (x, by + 52)], fill=WHITE)
    draw_tracked(d, (bx + 150, by + 40), "INFRA PRESS", font(FONT_AVENIR, 40, 0), WHITE, tracking=5)


def main() -> None:
    bg = build_background()
    draw_portrait(bg)
    draw_ribbon(bg)
    draw_text(bg)

    cover = bg.convert("RGB")
    path = OUT / "ai-infra-course-cover.png"
    cover.save(path)
    cover.resize((800, 1131), Image.LANCZOS).save(OUT / "ai-infra-course-cover-thumb.png")
    print("saved:", path, path.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
