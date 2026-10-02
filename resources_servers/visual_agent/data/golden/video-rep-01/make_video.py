# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Golden for video-rep-01: an animated quarterly revenue dashboard. Writes video.mp4 (deterministic)."""

import math
import subprocess
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


W, H, FPS, DURATION = 1280, 720, 30, 10.0
FONT_DIR = "/usr/share/fonts/opentype/inter"
SS = 3  # supersampling factor for shapes

BG, CARD, BORDER = "#0B1220", "#111A2E", "#1E2A44"
TEXT, MUTED, DIM, LABEL = "#F8FAFC", "#94A3B8", "#64748B", "#CBD5E1"
SKY, VIOLET, GREEN, AMBER = "#38BDF8", "#A78BFA", "#22C55E", "#F59E0B"

VALUES = [212, 228, 219, 246, 262, 251, 280, 297, 289, 318, 336, 352, 371]  # weekly revenue, $k
TOTAL_M = sum(VALUES) / 1000.0  # 3.661
TARGET_FRACTION = TOTAL_M / 4.25  # 0.8614
X0, XSTEP, Y0 = 96, 58, 632  # W1 x, week spacing, y of $0k (1 px per $1k)

LEFT_CARD = (32, 128, 820, 688)
KPI_CARD = (844, 128, 1248, 332)
DONUT_CARD = (844, 352, 1248, 688)
DONUT_C, DONUT_R, DONUT_STROKE = (1046, 528), 92, 22


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(f"{FONT_DIR}/Inter-{weight}.otf", size)


def rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgba(hex_color: str, alpha: float) -> Tuple[int, int, int, int]:
    return (*rgb(hex_color), round(255 * alpha))


def clamp01(u: float) -> float:
    return min(1.0, max(0.0, u))


def ease_out_cubic(u: float) -> float:
    u = clamp01(u)
    return 1 - (1 - u) ** 3


def ease_in_out_cubic(u: float) -> float:
    u = clamp01(u)
    return 4 * u**3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def ease_out_back(u: float) -> float:
    u = clamp01(u)
    c1 = 1.70158
    return 1 + (c1 + 1) * (u - 1) ** 3 + c1 * (u - 1) ** 2


def px(i: int) -> int:
    return X0 + XSTEP * i


def py(v: float) -> float:
    return Y0 - v


def supersampled(draw_fn, size: Tuple[int, int] = (W, H)) -> Image.Image:
    """Draw with draw_fn(draw, s) on a transparent canvas SS times larger, then downsample."""
    big = Image.new("RGBA", (size[0] * SS, size[1] * SS), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(big), SS)
    return big.resize(size, Image.LANCZOS)


def card_layer(box: Tuple[int, int, int, int]) -> Image.Image:
    def draw_card(d: ImageDraw.ImageDraw, s: int) -> None:
        x0, y0, x1, y1 = box
        d.rounded_rectangle((x0 * s, y0 * s, x1 * s, y1 * s), radius=20 * s, fill=CARD, outline=BORDER, width=s)

    return supersampled(draw_card)


def left_card_content() -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((60, 164), "Weekly revenue ($k)", font=font("SemiBold", 18), fill=LABEL, anchor="lm")
    small = font("Regular", 14)
    for v in (0, 100, 200, 300, 400):
        y = int(py(v))
        d.line((X0 - 4, y, px(12) + 12, y), fill=BORDER, width=1)
        d.text((80, y), str(v), font=small, fill=DIM, anchor="rm")
    tick = font("Regular", 13)
    for i in range(13):
        d.text((px(i), 656), f"W{i + 1}", font=tick, fill=DIM, anchor="mm")
    return layer


def kpi_card_content() -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((872, 164), "Total revenue", font=font("Medium", 16), fill=MUTED, anchor="lm")
    return layer


def donut_card_content() -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((872, 388), "Quarterly target", font=font("Medium", 16), fill=MUTED, anchor="lm")
    donut = Donut()
    alpha = donut.coverage(donut.band)[..., 0]
    track = np.zeros((H, W, 4), np.uint8)
    track[..., :3] = rgb(BORDER)
    track[..., 3] = np.round(alpha * 255).astype(np.uint8)
    return Image.alpha_composite(layer, Image.fromarray(track, "RGBA"))


def header_layer() -> Image.Image:
    def draw_pill(d: ImageDraw.ImageDraw, s: int) -> None:
        d.rounded_rectangle((1140 * s, 40 * s, 1248 * s, 72 * s), radius=16 * s, outline=BORDER, width=s)

    layer = supersampled(draw_pill)
    d = ImageDraw.Draw(layer)
    d.text((48, 54), "Q3 Revenue Review", font=font("Bold", 34), fill=TEXT, anchor="lm")
    d.text((48, 94), "Northwind Outfitters · weekly revenue, July to September", font=font("Regular", 18),
           fill=MUTED, anchor="lm")  # fmt: skip
    d.text((1194, 56), "FY2026 Q3", font=font("SemiBold", 15), fill=MUTED, anchor="mm")
    return layer


def chart_line_layer() -> Image.Image:
    """Area under the line (vertical gradient) plus the 3 px line, full width; revealed per frame."""
    pts = [(px(i), py(v)) for i, v in enumerate(VALUES)]
    mask = Image.new("L", (W * SS, H * SS), 0)
    ImageDraw.Draw(mask).polygon(
        [(x * SS, y * SS) for x, y in pts] + [(px(12) * SS, Y0 * SS), (px(0) * SS, Y0 * SS)], fill=255
    )
    mask = np.asarray(mask.resize((W, H), Image.LANCZOS), np.float32) / 255.0
    ys = np.arange(H, dtype=np.float32)[:, None]
    grad = np.clip((Y0 - ys) / (Y0 - py(400)), 0.0, 1.0) * 0.30
    area = np.zeros((H, W, 4), np.uint8)
    area[..., :3] = rgb(SKY)
    area[..., 3] = np.round(mask * grad * 255).astype(np.uint8)
    layer = Image.fromarray(area, "RGBA")

    def draw_line(d: ImageDraw.ImageDraw, s: int) -> None:
        d.line([(x * s, y * s) for x, y in pts], fill=SKY, width=3 * s, joint="curve")

    return Image.alpha_composite(layer, supersampled(draw_line))


def dot_tile(color: str = SKY) -> Image.Image:
    def draw_dot(d: ImageDraw.ImageDraw, s: int) -> None:
        d.ellipse((3 * s, 3 * s, 15 * s, 15 * s), fill=BG, outline=color, width=2 * s)

    return supersampled(draw_dot, (18, 18))


def ring_tile() -> Image.Image:
    def draw_ring(d: ImageDraw.ImageDraw, s: int) -> None:
        d.ellipse((2 * s, 2 * s, 26 * s, 26 * s), outline=AMBER, width=2 * s)

    return supersampled(draw_ring, (28, 28))


def chip_tile(text: str, fill: str, ink: str, size: Tuple[int, int], weight: str, font_size: int,
              outline: str = None) -> Image.Image:  # fmt: skip
    w, h = size

    def draw_chip(d: ImageDraw.ImageDraw, s: int) -> None:
        d.rounded_rectangle((0, 0, w * s - 1, h * s - 1), radius=h * s // 2, fill=fill,
                            outline=outline, width=s if outline else 0)  # fmt: skip

    tile = supersampled(draw_chip, (w, h))
    ImageDraw.Draw(tile).text((w / 2, h / 2), text, font=font(weight, font_size), fill=ink, anchor="mm")
    return tile


class Donut:
    """Anti-aliased ring and round-capped arc from 12 o'clock, clockwise, as distance fields."""

    def __init__(self) -> None:
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5
        cx, cy = DONUT_C
        self.dx, self.dy = xs - cx, ys - cy
        r = np.hypot(self.dx, self.dy)
        self.theta = np.mod(np.arctan2(self.dx, -self.dy), 2 * np.pi)
        self.band = np.abs(r - DONUT_R)
        self.start_d = np.hypot(self.dx, self.dy + DONUT_R)

    @staticmethod
    def coverage(distance: np.ndarray) -> np.ndarray:
        return np.clip(DONUT_STROKE / 2 - distance + 0.5, 0.0, 1.0)[..., None]

    def arc(self, p: float) -> np.ndarray:
        sweep = 2 * np.pi * p
        ex, ey = DONUT_R * math.sin(sweep), -DONUT_R * math.cos(sweep)
        end_d = np.hypot(self.dx - ex, self.dy - ey)
        distance = np.where(self.theta <= sweep, self.band, np.minimum(self.start_d, end_d))
        return self.coverage(distance)


def with_alpha(tile: Image.Image, alpha: float) -> Image.Image:
    out = tile.copy()
    out.putalpha(out.getchannel("A").point(lambda a: round(a * alpha)))
    return out


def paste_center(frame: Image.Image, tile: Image.Image, cx: float, cy: float, alpha: float = 1.0,
                 scale: float = 1.0) -> None:  # fmt: skip
    if alpha <= 0 or scale <= 0:
        return
    if scale != 1.0:
        tile = tile.resize((max(1, round(tile.width * scale)), max(1, round(tile.height * scale))), Image.LANCZOS)
    if alpha < 1.0:
        tile = with_alpha(tile, alpha)
    frame.alpha_composite(tile, (round(cx - tile.width / 2), round(cy - tile.height / 2)))


def main() -> None:
    header = header_layer()
    cards = [
        Image.alpha_composite(card_layer(LEFT_CARD), left_card_content()),
        Image.alpha_composite(card_layer(KPI_CARD), kpi_card_content()),
        Image.alpha_composite(card_layer(DONUT_CARD), donut_card_content()),
    ]
    line_layer = np.asarray(chart_line_layer(), np.float32)
    dot, ring = dot_tile(), ring_tile()
    end_chip = chip_tile("$371k", SKY, BG, (76, 30), "Bold", 15)
    delta_chip = chip_tile("+18.4% vs Q2", "#12301F", GREEN, (132, 30), "SemiBold", 15)
    callout = chip_tile("Summer sale +12%", "#2A2110", AMBER, (156, 32), "SemiBold", 14, outline=AMBER)
    donut = Donut()
    xs = np.arange(W, dtype=np.float32) + 0.5
    big_font, pct_font = font("Bold", 56), font("Bold", 44)
    target_label = Image.new("RGBA", (220, 24), (0, 0, 0, 0))
    ImageDraw.Draw(target_label).text((110, 12), "$3.66M of $4.25M", font=font("Regular", 16), fill=MUTED, anchor="mm")

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "video.mp4",
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(round(DURATION * FPS)):
        t = i / FPS
        frame = Image.new("RGBA", (W, H), BG)
        frame.alpha_composite(header)
        # Intro: cards rise 24 px and fade in, 0.6 s each, staggered by 0.1 s.
        for k, card in enumerate(cards):
            e = ease_out_cubic((t - 0.1 * k) / 0.6)
            if e >= 1.0:
                frame.alpha_composite(card)
            elif e > 0:
                frame.alpha_composite(with_alpha(card, e), (0, round(24 * (1 - e))))

        # Line chart: reveal edge moves linearly from W1 to W13 between 1.0 and 4.0 s.
        reveal = px(0) + (px(12) - px(0)) * clamp01((t - 1.0) / 3.0)
        if t > 1.0:
            cover = np.clip(reveal - xs + 0.5, 0.0, 1.0)
            part = line_layer.copy()
            part[..., 3] *= cover[None, :]
            frame.alpha_composite(Image.fromarray(np.round(part).astype(np.uint8), "RGBA"))
            for k, v in enumerate(VALUES):
                if px(k) <= reveal + 1e-6:
                    paste_center(frame, dot, px(k), py(v))
        if t >= 4.0:
            e = clamp01((t - 4.0) / 0.4)
            paste_center(frame, end_chip, px(12) - 24, py(VALUES[12]) - 34, alpha=e,
                         scale=0.6 + 0.4 * ease_out_back(e))  # fmt: skip

        # Annotation at W8: callout drops in, dashed guide draws down, ring on the point.
        if t >= 6.4:
            e = ease_out_cubic((t - 6.4) / 0.6)
            paste_center(frame, callout, px(7), 200 - 24 * (1 - e), alpha=e)
        if t >= 6.6:
            d = ImageDraw.Draw(frame, "RGBA")
            bottom = 218 + (Y0 - 218) * clamp01((t - 6.6) / 0.6)
            y = 218
            while y < bottom:
                d.line((px(7), y, px(7), min(y + 6, bottom)), fill=rgba(AMBER, 0.7), width=2)
                y += 11
            paste_center(frame, dot, px(7), py(VALUES[7]))
        if t >= 7.0:
            paste_center(frame, ring, px(7), py(VALUES[7]), alpha=clamp01((t - 7.0) / 0.2))

        d = ImageDraw.Draw(frame)
        # KPI: total counts up with ease-out cubic from 0.8 to 2.8 s, then the delta chip fades in.
        if t >= 0.8:
            value = TOTAL_M * ease_out_cubic((t - 0.8) / 2.0)
            d.text((872, 228), f"${value:.2f}M", font=big_font, fill=TEXT, anchor="lm")
        if t >= 2.8:
            paste_center(frame, delta_chip, 872 + 66, 292, alpha=clamp01((t - 2.8) / 0.4))

        # Donut: fills to the target share with ease-in-out cubic from 4.4 to 6.0 s.
        if t >= 4.4:
            p = TARGET_FRACTION * ease_in_out_cubic((t - 4.4) / 1.6)
            if p > 0:
                cov = donut.arc(p)[..., 0]
                arc = np.zeros((H, W, 4), np.uint8)
                arc[..., :3] = rgb(VIOLET)
                arc[..., 3] = np.round(cov * 255).astype(np.uint8)
                frame.alpha_composite(Image.fromarray(arc, "RGBA"))
            d = ImageDraw.Draw(frame)
            d.text(DONUT_C, f"{round(100 * p)}%", font=pct_font, fill=TEXT, anchor="mm")
        if t >= 6.0:
            paste_center(frame, target_label, DONUT_C[0], 652, alpha=clamp01((t - 6.0) / 0.3))

        proc.stdin.write(frame.convert("RGB").tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")


if __name__ == "__main__":
    main()
