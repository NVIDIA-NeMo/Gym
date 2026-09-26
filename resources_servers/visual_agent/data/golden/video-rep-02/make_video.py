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
"""Golden for video-rep-02: a three-slide onboarding carousel with parallax. Writes video.mp4 (deterministic)."""

import math
import subprocess
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


W, H, FPS, DURATION = 1080, 1080, 30, 12.0
FONT_DIR = "/usr/share/fonts/opentype/inter"
SS = 3  # supersampling factor for shapes

BGS = ["#EEF2FF", "#ECFDF5", "#FFF7ED"]
ACCENTS = ["#6366F1", "#10B981", "#F97316"]
INK, BODY, DOT, SKIP, TRACK, WHITE = "#0F172A", "#475569", "#CBD5E1", "#64748B", "#E2E8F0", "#FFFFFF"
TITLES = ["Plan your week", "Track every habit", "Celebrate streaks"]
BODIES = [
    ("Drag tasks onto the days that suit you", "and Tempo fills in the rest."),
    ("Tick off habits in one tap and watch", "your weekly progress grow."),
    ("Hit a seven-day streak and unlock", "new themes for your planner."),
]
# Horizontal travel per slide change, as a multiple of the frame width: parallax between layers.
ILLUSTRATION_FACTOR, TITLE_FACTOR, BODY_FACTOR = 1.25, 1.0, 0.85
TRANSITIONS = [(3.0, 4.0), (7.0, 8.0)]
ILLU_BOX = (240, 120, 840, 600)  # region holding each illustration (blob centered at 540, 360)
BUTTON = (300, 930, 780, 1010)


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(f"{FONT_DIR}/Inter-{weight}.otf", size)


def rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def mix(a: str, b: str, u: float) -> Tuple[int, int, int]:
    ca, cb = rgb(a), rgb(b)
    return tuple(round(x + (y - x) * u) for x, y in zip(ca, cb))


def clamp01(u: float) -> float:
    return min(1.0, max(0.0, u))


def ease_in_out_cubic(u: float) -> float:
    u = clamp01(u)
    return 4 * u**3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def ease_out_quad(u: float) -> float:
    u = clamp01(u)
    return 1 - (1 - u) ** 2


def ease_out_cubic(u: float) -> float:
    u = clamp01(u)
    return 1 - (1 - u) ** 3


def slide_position(t: float) -> float:
    """0 on slide 1, 1 on slide 2, 2 on slide 3; eased in between."""
    s = 0.0
    for start, end in TRANSITIONS:
        s += ease_in_out_cubic((t - start) / (end - start))
    return s


def star(
    cx: float, cy: float, outer: float, inner: float, points: int, rotation: float = -90
) -> List[Tuple[float, float]]:
    pts = []
    for k in range(points * 2):
        r = outer if k % 2 == 0 else inner
        a = math.radians(rotation + k * 180 / points)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def illustration(k: int) -> Image.Image:
    """Full-frame RGBA layer with slide k's illustration (blob + icon) at rest."""
    s = SS
    big = Image.new("RGBA", (W * s, H * s), (0, 0, 0, 0))
    accent = ACCENTS[k]
    blob = Image.new("RGBA", big.size, (0, 0, 0, 0))
    ImageDraw.Draw(blob).ellipse(((540 - 200) * s, (360 - 200) * s, (540 + 200) * s, (360 + 200) * s),
                                 fill=(*rgb(accent), 36))  # fmt: skip
    big.alpha_composite(blob)
    if k in (0, 1):
        # Soft card shadow, then a white card.
        cw, ch = (260, 230) if k == 0 else (300, 232)
        x0, y0 = 540 - cw // 2, 370 - ch // 2
        shadow = Image.new("RGBA", big.size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle(
            (x0 * s, (y0 + 12) * s, (x0 + cw) * s, (y0 + ch + 12) * s), radius=24 * s, fill=(*rgb(INK), 40)
        )
        big.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(16 * s)))
        d = ImageDraw.Draw(big)
        d.rounded_rectangle((x0 * s, y0 * s, (x0 + cw) * s, (y0 + ch) * s), radius=24 * s, fill=WHITE)
    d = ImageDraw.Draw(big)
    if k == 0:
        # Calendar: accent header, two binder tabs, 3x4 grid of days with one highlighted.
        d.rounded_rectangle((x0 * s, y0 * s, (x0 + cw) * s, (y0 + 58) * s), radius=24 * s, fill=accent)
        d.rectangle((x0 * s, (y0 + 30) * s, (x0 + cw) * s, (y0 + 58) * s), fill=accent)
        for tx in (x0 + 70, x0 + cw - 70):
            d.rounded_rectangle(((tx - 6) * s, (y0 - 14) * s, (tx + 6) * s, (y0 + 18) * s), radius=6 * s, fill=INK)
        for row in range(3):
            for col in range(4):
                cx0 = x0 + 26 + col * 54
                cy0 = y0 + 80 + row * 46
                fill = accent if (row, col) == (1, 2) else TRACK
                d.rounded_rectangle((cx0 * s, cy0 * s, (cx0 + 46) * s, (cy0 + 34) * s), radius=8 * s, fill=fill)
    elif k == 1:
        # Habit list: check circles and progress bars.
        for row, frac in enumerate((1.0, 0.7, 0.35)):
            cy = y0 + 52 + row * 64
            cx = x0 + 46
            if row < 2:
                d.ellipse(((cx - 16) * s, (cy - 16) * s, (cx + 16) * s, (cy + 16) * s), fill=accent)
                d.line([((cx - 7) * s, cy * s), ((cx - 2) * s, (cy + 6) * s), ((cx + 8) * s, (cy - 6) * s)],
                       fill=WHITE, width=4 * s, joint="curve")  # fmt: skip
            else:
                d.ellipse(((cx - 16) * s, (cy - 16) * s, (cx + 16) * s, (cy + 16) * s), outline=accent, width=4 * s)
            bx0, bx1 = x0 + 82, x0 + cw - 30
            d.rounded_rectangle((bx0 * s, (cy - 8) * s, bx1 * s, (cy + 8) * s), radius=8 * s, fill=TRACK)
            d.rounded_rectangle((bx0 * s, (cy - 8) * s, (bx0 + (bx1 - bx0) * frac) * s, (cy + 8) * s),
                                radius=8 * s, fill=accent)  # fmt: skip
    else:
        # Streak: a big star with sparkles.
        d.polygon([(x * s, y * s) for x, y in star(540, 368, 132, 56, 5)], fill=accent)
        faded = (*rgb(accent), 150)
        for cx, cy, r in ((392, 238, 26), (700, 262, 18), (690, 482, 30)):
            d.polygon([(x * s, y * s) for x, y in star(cx, cy, r, r * 0.3, 4)], fill=faded)
    layer = big.resize((W, H), Image.LANCZOS)
    if k == 2:
        ImageDraw.Draw(layer).text((540, 378), "7", font=font("Black", 72), fill=WHITE, anchor="mm")
    return layer


def text_layer(lines: List[Tuple[str, str, str, int, int]]) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for text, weight, color, size, y in lines:
        d.text((540, y), text, font=font(weight, size), fill=color, anchor="mm")
    return layer


def shifted(layer: Image.Image, dx: int) -> Image.Image:
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if abs(dx) < W:
        out.alpha_composite(layer.crop((max(0, -dx), 0, min(W, W - dx), H)), (max(0, dx), 0))
    return out


def button_tile(color: Tuple[int, int, int], labels: List[Tuple[str, float]]) -> Image.Image:
    bw, bh = BUTTON[2] - BUTTON[0], BUTTON[3] - BUTTON[1]
    big = Image.new("RGBA", (bw * SS, bh * SS), (0, 0, 0, 0))
    ImageDraw.Draw(big).rounded_rectangle((0, 0, bw * SS - 1, bh * SS - 1), radius=bh * SS // 2, fill=color)
    tile = big.resize((bw, bh), Image.LANCZOS)
    label_font = font("SemiBold", 30)
    for text, alpha in labels:
        if alpha <= 0:
            continue
        txt = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        ImageDraw.Draw(txt).text((bw / 2, bh / 2), text, font=label_font, fill=(255, 255, 255, round(255 * alpha)),
                                 anchor="mm")  # fmt: skip
        tile.alpha_composite(txt)
    return tile


def main() -> None:
    illus = [illustration(k) for k in range(3)]
    titles = [text_layer([(TITLES[k], "Bold", INK, 60, 664)]) for k in range(3)]
    bodies = [text_layer([(BODIES[k][0], "Regular", BODY, 30, 740), (BODIES[k][1], "Regular", BODY, 30, 782)])
              for k in range(3)]  # fmt: skip
    wordmark_font, skip_font = font("Bold", 34), font("Medium", 26)
    bw, bh = BUTTON[2] - BUTTON[0], BUTTON[3] - BUTTON[1]
    bcx, bcy = (BUTTON[0] + BUTTON[2]) / 2, (BUTTON[1] + BUTTON[3]) / 2
    ys, xs = np.mgrid[0:bh, 0:bw].astype(np.float32) + 0.5

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "video.mp4",
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(round(DURATION * FPS)):
        t = i / FPS
        s = slide_position(t)
        k0 = min(int(math.floor(s)), 2)
        frac = s - k0
        k1 = min(k0 + 1, 2)
        bg = mix(BGS[k0], BGS[k1], frac)
        accent = mix(ACCENTS[k0], ACCENTS[k1], frac)
        frame = Image.new("RGBA", (W, H), bg)

        for k in range(3):
            offset = k - s
            if abs(offset) >= 1.0:
                continue
            frame.alpha_composite(shifted(illus[k], round(offset * W * ILLUSTRATION_FACTOR)))
            frame.alpha_composite(shifted(titles[k], round(offset * W * TITLE_FACTOR)))
            frame.alpha_composite(shifted(bodies[k], round(offset * W * BODY_FACTOR)))

        # Top bar: logo dot + wordmark; Skip fades out while moving to the last slide.
        top = Image.new("RGBA", (W * SS, 168 * SS), (0, 0, 0, 0))
        ImageDraw.Draw(top).ellipse((70 * SS, 70 * SS, 98 * SS, 98 * SS), fill=accent)
        frame.alpha_composite(top.resize((W, 168), Image.LANCZOS))
        d = ImageDraw.Draw(frame)
        d.text((112, 84), "Tempo", font=wordmark_font, fill=INK, anchor="lm")
        skip_alpha = 1.0 - clamp01(s - 1.0)
        if skip_alpha > 0:
            d.text((1008, 84), "Skip", font=skip_font, fill=(*rgb(SKIP), round(255 * skip_alpha)), anchor="rm")

        # Page indicator: the active dot stretches into a pill in the slide's accent color.
        dots = Image.new("RGBA", (W * SS, 40 * SS), (0, 0, 0, 0))
        dd = ImageDraw.Draw(dots)
        x = 540 - 48
        for k in range(3):
            active = max(0.0, 1.0 - abs(s - k))
            w = 14 + 30 * active
            dd.rounded_rectangle(
                (x * SS, 13 * SS, (x + w) * SS, 27 * SS), radius=7 * SS, fill=mix(DOT, ACCENTS[k], active)
            )
            x += w + 12
        frame.alpha_composite(dots.resize((W, 40), Image.LANCZOS), (0, 880 - 20))

        # Button: Next becomes Get started on the last slide; pressed at 10.0 s with a ripple.
        labels = [("Next", 1.0 - clamp01((s - 1.0) * 2)), ("Get started", clamp01((s - 1.5) * 2))]
        tile = button_tile(accent, labels)
        if 10.0 <= t < 10.6:
            u = (t - 10.0) / 0.6
            radius = 20 + 300 * ease_out_cubic(u)
            dist = np.hypot(xs - bw / 2, ys - bh / 2)
            cover = np.clip(radius - dist + 0.5, 0.0, 1.0) * 0.35 * (1 - u)
            arr = np.asarray(tile, np.float32).copy()
            a = cover * (arr[..., 3] / 255.0)
            arr[..., :3] = arr[..., :3] * (1 - a[..., None]) + 255.0 * a[..., None]
            tile = Image.fromarray(np.round(arr).astype(np.uint8), "RGBA")
        scale = 1.0
        if 10.0 <= t < 10.12:
            scale = 1.0 - 0.06 * ease_out_quad((t - 10.0) / 0.12)
        elif 10.12 <= t < 10.4:
            scale = 0.94 + 0.06 * ease_out_quad((t - 10.12) / 0.28)
        if scale != 1.0:
            tile = tile.resize((round(bw * scale), round(bh * scale)), Image.LANCZOS)
        frame.alpha_composite(tile, (round(bcx - tile.width / 2), round(bcy - tile.height / 2)))

        proc.stdin.write(frame.convert("RGB").tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")


if __name__ == "__main__":
    main()
