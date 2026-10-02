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
"""Golden for video-rep-04: a portrait kinetic-typography festival promo. Writes video.mp4 (deterministic)."""

import math
import os
import subprocess
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont


W, H, FPS, DURATION = 720, 1280, 30, 10.0
SS = 3
FONT_DIR = "/usr/share/fonts/opentype/inter"

NAVY, AMBER, PINK, CYAN = "#0B1026", "#F5A524", "#EF476F", "#4CC9F0"
WHITE, SOFT, MUTED = "#FFFFFF", "#E2E8F0", "#94A3B8"

CIRCLE_C, CIRCLE_R = (520, 330), 220
RING_R, RING_TEXT = 168, "LIVE MUSIC · 12–18 MAY · LIVE MUSIC · 12–18 MAY · "
SPIN_DEG_PER_S, BADGE_START = 40.0, 2.2
BARS = [(600, 600, AMBER, 0.30), (630, 440, PINK, 0.45), (660, 280, CYAN, 0.60)]  # y, width, colour, start
BAR_H, LEFT = 14, 60
TITLE = [("JAZZ", 830, WHITE, 0.9), ("WEEK", 980, AMBER, 1.1)]  # text, baseline, colour, start
LINEUP = ["Ada Moreno Trio", "The Kessler Five", "Nia Okafor Quartet", "Blue Harbor Big Band", "Lior Sax Collective"]
LINEUP_Y0, LINEUP_DY, LINEUP_T0, LINEUP_DT = 1060, 38, 3.0, 0.3
DOT_COLORS = [AMBER, PINK, CYAN, AMBER, PINK]
PANEL_TOP = 1010


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    path = f"{FONT_DIR}/Inter-{weight}.otf"
    if not os.path.exists(path):  # local preview only; the sandbox has Inter
        bold = weight in ("Bold", "SemiBold")
        path = f"/usr/share/fonts/truetype/liberation/LiberationSans-{'Bold' if bold else 'Regular'}.ttf"
    return ImageFont.truetype(path, size)


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


def supersampled(draw_fn, size: Tuple[int, int]) -> Image.Image:
    big = Image.new("RGBA", (size[0] * SS, size[1] * SS), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(big), SS)
    return big.resize(size, Image.LANCZOS)


def with_alpha(tile: Image.Image, alpha: float) -> Image.Image:
    out = tile.copy()
    out.putalpha(out.getchannel("A").point(lambda a: round(a * clamp01(alpha))))
    return out


def circle_tile(radius: int, color: str) -> Image.Image:
    size = 2 * radius + 4

    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        d.ellipse((2 * s, 2 * s, (size - 2) * s, (size - 2) * s), fill=color)

    return supersampled(draw, (size, size))


def badge_tile() -> Image.Image:
    """Circular text (clockwise from 12 o'clock, letter tops facing outward) plus the centre words."""
    size = 2 * (RING_R + 40)
    c = size / 2
    big = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    f = font("SemiBold", 24 * SS)
    advances = [f.getlength(ch) for ch in RING_TEXT]
    total = sum(advances)
    k = 2 * math.pi / total
    acc = 0.0
    for ch, adv in zip(RING_TEXT, advances):
        theta = (acc + adv / 2) * k  # clockwise from 12 o'clock
        acc += adv
        if ch == " ":
            continue
        glyph = Image.new("RGBA", (60 * SS, 60 * SS), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text((30 * SS, 30 * SS), ch, font=f, fill=NAVY, anchor="mm")
        glyph = glyph.rotate(-math.degrees(theta), resample=Image.BICUBIC)
        x = c * SS + RING_R * SS * math.sin(theta)
        y = c * SS - RING_R * SS * math.cos(theta)
        big.alpha_composite(glyph, (round(x - glyph.width / 2), round(y - glyph.height / 2)))
    return big.resize((size, size), Image.LANCZOS)


def centre_words() -> Image.Image:
    tile = Image.new("RGBA", (300, 140), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    d.text((150, 44), "SEVEN", font=font("Bold", 46), fill=NAVY, anchor="mm")
    d.text((150, 96), "NIGHTS", font=font("Bold", 46), fill=NAVY, anchor="mm")
    return tile


def spaced_text(d: ImageDraw.ImageDraw, xy, text: str, f, fill, spacing: float) -> None:
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill, anchor="ls")
        x += f.getlength(ch) + spacing


def scaled_about_center(frame: Image.Image, tile: Image.Image, cx: float, cy: float, scale: float,
                        alpha: float = 1.0) -> None:  # fmt: skip
    if scale <= 0 or alpha <= 0:
        return
    if scale != 1.0:
        tile = tile.resize((max(1, round(tile.width * scale)), max(1, round(tile.height * scale))), Image.LANCZOS)
    if alpha < 1.0:
        tile = with_alpha(tile, alpha)
    frame.alpha_composite(tile, (round(cx - tile.width / 2), round(cy - tile.height / 2)))


def bar_tile(width: int, color: str) -> Image.Image:
    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        d.rounded_rectangle((0, 0, width * s - 1, BAR_H * s - 1), radius=BAR_H * s // 2, fill=color)

    return supersampled(draw, (width, BAR_H))


def title_tile(text: str, color: str) -> Image.Image:
    """Line tile 720 x 200 with the baseline at y = 170."""
    tile = Image.new("RGBA", (W, 200), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((LEFT - 6, 170), text, font=font("Bold", 170), fill=color, anchor="ls")
    return tile


def lineup_tile(index: int) -> Image.Image:
    tile = Image.new("RGBA", (W, 40), (0, 0, 0, 0))

    def dot(d: ImageDraw.ImageDraw, s: int) -> None:
        d.ellipse(((LEFT + 2) * s, 14 * s, (LEFT + 14) * s, 26 * s), fill=DOT_COLORS[index])

    tile.alpha_composite(supersampled(dot, (W, 40)))
    ImageDraw.Draw(tile).text((LEFT + 28, 30), LINEUP[index], font=font("Medium", 30), fill=SOFT, anchor="ls")
    return tile


def panel_tile() -> Image.Image:
    tile = Image.new("RGBA", (W, H - PANEL_TOP), AMBER)

    def arrow(d: ImageDraw.ImageDraw, s: int) -> None:
        d.ellipse((582 * s, 80 * s, 650 * s, 148 * s), fill=NAVY)
        d.line((600 * s, 114 * s, 632 * s, 114 * s), fill=WHITE, width=4 * s)
        d.line((620 * s, 102 * s, 632 * s, 114 * s, 620 * s, 126 * s), fill=WHITE, width=4 * s, joint="curve")

    tile.alpha_composite(supersampled(arrow, tile.size))
    d = ImageDraw.Draw(tile)
    d.text((LEFT, 100), "TICKETS FROM €25", font=font("Bold", 48), fill=NAVY, anchor="ls")
    d.text((LEFT, 150), "northsidejazz.eu", font=font("Medium", 28), fill=NAVY, anchor="ls")
    return tile


def main() -> None:
    circle = circle_tile(CIRCLE_R, AMBER)
    badge, words = badge_tile(), centre_words()
    bars = [bar_tile(width, color) for _, width, color, _ in BARS]
    titles = [title_tile(text, color) for text, _, color, _ in TITLE]
    lineup = [lineup_tile(i) for i in range(len(LINEUP))]
    panel = panel_tile()
    label = Image.new("RGBA", (W, 60), (0, 0, 0, 0))
    spaced_text(ImageDraw.Draw(label), (LEFT, 40), "NORTHSIDE ARTS PRESENTS", font("SemiBold", 20), MUTED, 3.0)

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "video.mp4",
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(round(DURATION * FPS)):
        t = i / FPS
        frame = Image.new("RGBA", (W, H), NAVY)

        # Label fades in 0.2 to 0.6 s.
        a = clamp01((t - 0.2) / 0.4)
        if a > 0:
            frame.alpha_composite(label if a >= 1 else with_alpha(label, a), (0, 44))

        # Amber circle scales up from its centre, 0.0 to 0.8 s.
        scaled_about_center(frame, circle, *CIRCLE_C, ease_out_cubic(t / 0.8))

        # Badge: the ring of text and the centre words pop in 2.2 to 2.7 s, then the ring spins clockwise.
        if t >= BADGE_START:
            u = (t - BADGE_START) / 0.5
            scale = 0.6 + 0.4 * ease_out_back(u)
            alpha = clamp01(u)
            ring = badge.rotate(-SPIN_DEG_PER_S * (t - BADGE_START), resample=Image.BICUBIC)
            scaled_about_center(frame, ring, *CIRCLE_C, scale, alpha)
            scaled_about_center(frame, words, *CIRCLE_C, scale, alpha)

        # Bars wipe in from the left.
        for (y, width, _, start), tile in zip(BARS, bars):
            w = round(width * ease_out_cubic((t - start) / 0.4))
            if w >= BAR_H:
                frame.alpha_composite(tile.crop((0, 0, w - BAR_H // 2, BAR_H)), (LEFT, y))
                frame.alpha_composite(tile.crop((width - BAR_H // 2, 0, width, BAR_H)), (LEFT + w - BAR_H // 2, y))

        # Title lines slide up out of a mask (clip box ends 30 px below the baseline).
        for (text, baseline, _, start), tile in zip(TITLE, titles):
            e = ease_out_cubic((t - start) / 0.5)
            if e <= 0:
                continue
            offset = round(150 * (1 - e))
            top = baseline - 170  # tile top in frame coords when at rest
            clip_bottom = baseline + 30
            visible = tile.crop((0, 0, W, max(0, min(200, clip_bottom - (top + offset)))))
            if visible.height > 0:
                frame.alpha_composite(visible, (0, top + offset))

        # Lineup names slide in from the left (30 px) and fade in, one every 0.3 s.
        for k, tile in enumerate(lineup):
            e = ease_out_cubic((t - LINEUP_T0 - LINEUP_DT * k) / 0.4)
            if e <= 0:
                continue
            y = LINEUP_Y0 + LINEUP_DY * k - 30
            frame.alpha_composite(tile if e >= 1 else with_alpha(tile, e), (round(-30 * (1 - e)), y))

        # Ticket panel rises from the bottom edge, 6.0 to 6.6 s.
        e = ease_in_out_cubic((t - 6.0) / 0.6)
        if e > 0:
            top = round(H - (H - PANEL_TOP) * e)
            frame.alpha_composite(panel.crop((0, 0, W, H - top)), (0, top))

        proc.stdin.write(frame.convert("RGB").tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")


if __name__ == "__main__":
    main()
