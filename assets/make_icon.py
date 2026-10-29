# -*- coding: utf-8 -*-
"""生成 exe / 快捷方式用的图标（`assets/bigaagents.ico`）。

设计令牌与前端一致：
- 底 `#0B0E14`（页面背景）
- 主色 `#2DD4BF`（唯一操作色，青）
- 四根递增柱 + 一线上升趋势（"选股/净值"意象，不放文字 —— 16px 下文字必糊）

    py -3 assets/make_icon.py

⚠️ 只在需要改图标时跑一次；产物 `assets/bigaagents.ico` 应入库
  （打包脚本直接引用它，构建机不需要装 Pillow 才能出图标）。
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw

S = 512
BG = (11, 14, 20, 255)          # #0B0E14
TEAL = (45, 212, 191)           # #2DD4BF
RING = (45, 212, 191, 90)
LINE = (240, 244, 250, 235)
DOT = (240, 244, 250, 255)

BAR_W = int(S * 0.10)
GAP = int(S * 0.06)
TOPS = [0.60, 0.485, 0.365, 0.24]      # 柱顶（y 比例），自左向右递增


def main(out_dir: str | None = None) -> int:
    out_dir = out_dir or os.path.dirname(os.path.abspath(__file__))
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    d.rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 0.22), fill=BG)
    d.rounded_rectangle([7, 7, S - 8, S - 8], radius=int(S * 0.20), outline=RING, width=4)

    total = len(TOPS) * BAR_W + (len(TOPS) - 1) * GAP
    x0 = (S - total) // 2
    base = int(S * 0.80)

    for i, top_frac in enumerate(TOPS):
        x = x0 + i * (BAR_W + GAP)
        top = int(S * top_frac)
        k = 0.55 + 0.15 * i                       # 越靠右越亮
        col = (min(255, int(TEAL[0] * k)), min(255, int(TEAL[1] * k)),
               min(255, int(TEAL[2] * k)), 255)
        d.rounded_rectangle([x, top, x + BAR_W, base],
                            radius=int(BAR_W * 0.22), fill=col)

    pts = [(x0 + i * (BAR_W + GAP) + BAR_W / 2, int(S * t) - int(S * 0.055))
           for i, t in enumerate(TOPS)]
    d.line(pts, fill=LINE, width=int(S * 0.022), joint="curve")
    r = int(S * 0.018)
    for px, py in pts:
        d.ellipse([px - r, py - r, px + r, py + r], fill=DOT)

    png = os.path.join(out_dir, "bigaagents_512.png")
    ico = os.path.join(out_dir, "bigaagents.ico")
    img.save(png)
    img.save(ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                        (64, 64), (128, 128), (256, 256)])
    print(f"[icon] {ico}")
    print(f"[icon] {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
