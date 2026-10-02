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
"""Golden for video-rep-03: a delivery-tracking map animation. Writes video.mp4 (deterministic)."""

import math
import os
import subprocess
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont


W, H, FPS, DURATION = 1280, 720, 30, 10.0
SS = 3  # supersampling factor for shapes
FONT_DIR = "/usr/share/fonts/opentype/inter"

LAND, WATER, PARK, ROAD = "#EEF1EA", "#A9D2F3", "#CFE7B8", "#FFFFFF"
INK, MUTED, FAINT, TRACK = "#0F172A", "#64748B", "#94A3B8", "#E2E8F0"
BLUE, GREY, RED, GREEN = "#2563EB", "#94A3B8", "#EF4444", "#16A34A"

MAJOR_Y, MAJOR_X = (150, 380, 610), (160, 460, 760)
MINOR_Y, MINOR_X = (265, 495), (310, 610)
ROUTE = [(210, 610), (460, 610), (460, 495), (610, 495), (610, 380), (760, 380), (760, 205)]
DEPOT, DROP = ROUTE[0], ROUTE[-1]
ROUTE_KM, ETA_MIN = 2.4, 12
CARD1, CARD2 = (872, 32, 1248, 250), (872, 272, 1248, 520)


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    path = f"{FONT_DIR}/Inter-{weight}.otf"
    if not os.path.exists(path):  # local preview only; the sandbox has Inter
        bold = weight in ("Bold", "SemiBold")
        path = f"/usr/share/fonts/truetype/liberation/LiberationSans-{'Bold' if bold else 'Regular'}.ttf"
    return ImageFont.truetype(path, size)


def rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgba(hex_color: str, alpha: float) -> Tuple[int, int, int, int]:
    return (*rgb(hex_color), round(255 * max(0.0, min(1.0, alpha))))


def clamp01(u: float) -> float:
    return min(1.0, max(0.0, u))


def ease_out_cubic(u: float) -> float:
    u = clamp01(u)
    return 1 - (1 - u) ** 3


def ease_out_back(u: float) -> float:
    u = clamp01(u)
    c1 = 1.70158
    return 1 + (c1 + 1) * (u - 1) ** 3 + c1 * (u - 1) ** 2


def ease_in_out_sine(u: float) -> float:
    u = clamp01(u)
    return -(math.cos(math.pi * u) - 1) / 2


def supersampled(draw_fn, size: Tuple[int, int] = (W, H)) -> Image.Image:
    big = Image.new("RGBA", (size[0] * SS, size[1] * SS), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(big), SS)
    return big.resize(size, Image.LANCZOS)


def river_polygon(s: int) -> List[Tuple[float, float]]:
    """The river is a band of constant 54 px width around y = 300 + 40 sin(x / 140)."""
    top, bottom = [], []
    for x in range(-20, W + 21, 2):
        y = 300 + 40 * math.sin(x / 140.0)
        half = 27 * math.sqrt(1 + (40 / 140.0 * math.cos(x / 140.0)) ** 2)
        top.append((x * s, (y - half) * s))
        bottom.append((x * s, (y + half) * s))
    return top + bottom[::-1]


def thick_polyline(d: ImageDraw.ImageDraw, pts, width: float, fill, s: int, origin=(0, 0)) -> None:
    scaled = [((x - origin[0]) * s, (y - origin[1]) * s) for x, y in pts]
    d.line(scaled, fill=fill, width=round(width * s), joint="curve")
    r = width * s / 2
    for x, y in (scaled[0], scaled[-1]):
        d.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def base_map() -> Image.Image:
    frame = Image.new("RGBA", (W, H), LAND)

    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        d.polygon(river_polygon(s), fill=WATER)
        d.rounded_rectangle((330 * s, 405 * s, 440 * s, 475 * s), radius=10 * s, fill=PARK)
        d.rounded_rectangle((630 * s, 520 * s, 740 * s, 590 * s), radius=10 * s, fill=PARK)
        d.rounded_rectangle((180 * s, 170 * s, 290 * s, 245 * s), radius=10 * s, fill=PARK)
        for y in MINOR_Y:
            d.line((0, y * s, W * s, y * s), fill=ROAD, width=6 * s)
        for x in MINOR_X:
            d.line((x * s, 0, x * s, H * s), fill=ROAD, width=6 * s)
        for y in MAJOR_Y:
            d.line((0, y * s, W * s, y * s), fill=ROAD, width=14 * s)
        for x in MAJOR_X:
            d.line((x * s, 0, x * s, H * s), fill=ROAD, width=14 * s)

    frame.alpha_composite(supersampled(draw))
    d = ImageDraw.Draw(frame)
    park_font, street_font = font("Medium", 13), font("Medium", 12)
    d.text((385, 440), "Kestrel Park", font=park_font, fill="#4D7C3A", anchor="mm")
    d.text((685, 555), "Juniper Green", font=park_font, fill="#4D7C3A", anchor="mm")
    d.text((235, 207), "Alder Field", font=park_font, fill="#4D7C3A", anchor="mm")
    d.text((560, 381), "HARBOR AVE", font=street_font, fill=FAINT, anchor="mm")
    d.text((240, 151), "NORTH ST", font=street_font, fill=FAINT, anchor="mm")
    d.text((1010, 611), "QUAY ROAD", font=street_font, fill=FAINT, anchor="mm")
    d.text((110, 262 + 40 * math.sin(110 / 140.0) + 38), "Wren River", font=font("Medium", 14), fill="#2F6FA8",
           anchor="mm")  # fmt: skip
    # Vertical street name, rotated.
    label = Image.new("RGBA", (120, 20), (0, 0, 0, 0))
    ImageDraw.Draw(label).text((60, 10), "MILL LANE", font=street_font, fill=FAINT, anchor="mm")
    label = label.rotate(90, expand=True, resample=Image.BICUBIC)
    frame.alpha_composite(label, (461 - label.width // 2, 250 - label.height // 2))

    # Map chrome: zoom buttons and a scale bar.
    def chrome(d: ImageDraw.ImageDraw, s: int) -> None:
        d.rounded_rectangle((24 * s, 24 * s, 64 * s, 104 * s), radius=10 * s, fill="#FFFFFF", outline="#D5DBD2",
                            width=s)  # fmt: skip
        d.line((30 * s, 64 * s, 58 * s, 64 * s), fill="#D5DBD2", width=s)
        for cy in (44, 84):
            d.line((36 * s, cy * s, 52 * s, cy * s), fill=INK, width=2 * s)
        d.line((44 * s, 36 * s, 44 * s, 52 * s), fill=INK, width=2 * s)
        d.line((24 * s, 690 * s, 124 * s, 690 * s), fill=INK, width=2 * s)
        d.line((24 * s, 684 * s, 24 * s, 690 * s), fill=INK, width=2 * s)
        d.line((124 * s, 684 * s, 124 * s, 690 * s), fill=INK, width=2 * s)

    frame.alpha_composite(supersampled(chrome))
    ImageDraw.Draw(frame).text((74, 676), "200 m", font=font("Medium", 12), fill=INK, anchor="mm")
    return frame


def seg_lengths() -> List[float]:
    return [math.dist(a, b) for a, b in zip(ROUTE, ROUTE[1:])]


ROUTE_LEN = sum(seg_lengths())


def point_at(s_len: float) -> Tuple[float, float]:
    for (a, b), L in zip(zip(ROUTE, ROUTE[1:]), seg_lengths()):
        if s_len <= L:
            u = s_len / L
            return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u
        s_len -= L
    return ROUTE[-1]


def sub_route(s0: float, s1: float) -> List[Tuple[float, float]]:
    """Polyline covering arc length s0..s1."""
    pts = [point_at(s0)]
    acc = 0.0
    for p, L in zip(ROUTE[1:], seg_lengths()):
        acc += L
        if s0 < acc < s1:
            pts.append(p)
    pts.append(point_at(s1))
    return pts


ROUTE_BOX = (180, 180, 790, 640)  # region that contains the route and its casing


def route_layer(drawn: float, travelled: float) -> Image.Image:
    """Route layer for the ROUTE_BOX region (paste at its top-left corner)."""
    origin = ROUTE_BOX[:2]

    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        if drawn <= 0:
            return
        thick_polyline(d, sub_route(0.0, drawn), 12, "#FFFFFF", s, origin)
        if travelled > 0:
            thick_polyline(d, sub_route(0.0, min(travelled, drawn)), 6, GREY, s, origin)
        if drawn > travelled:
            thick_polyline(d, sub_route(travelled, drawn), 6, BLUE, s, origin)

    return supersampled(draw, (ROUTE_BOX[2] - ROUTE_BOX[0], ROUTE_BOX[3] - ROUTE_BOX[1]))


def depot_tile() -> Image.Image:
    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        d.ellipse((4 * s, 4 * s, 32 * s, 32 * s), fill="#FFFFFF")
        d.ellipse((7 * s, 7 * s, 29 * s, 29 * s), fill=INK)
        d.ellipse((14 * s, 14 * s, 22 * s, 22 * s), fill="#FFFFFF")

    return supersampled(draw, (36, 36))


def drop_tile() -> Image.Image:
    """Teardrop pin, 40x52, tip at the bottom centre (20, 50)."""

    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        d.polygon([(8 * s, 26 * s), (32 * s, 26 * s), (20 * s, 50 * s)], fill=RED)
        d.ellipse((4 * s, 4 * s, 36 * s, 36 * s), fill=RED)
        d.ellipse((14 * s, 14 * s, 26 * s, 26 * s), fill="#FFFFFF")

    return supersampled(draw, (40, 52))


def vehicle_tile() -> Image.Image:
    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        d.ellipse((2 * s, 2 * s, 30 * s, 30 * s), fill="#FFFFFF")
        d.ellipse((5 * s, 5 * s, 27 * s, 27 * s), fill=BLUE)
        d.ellipse((11 * s, 11 * s, 21 * s, 21 * s), fill="#FFFFFF")

    return supersampled(draw, (32, 32))


def chip(text: str) -> Image.Image:
    f = font("SemiBold", 13)
    w = round(f.getlength(text)) + 20

    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        d.rounded_rectangle((0, 0, w * s - 1, 26 * s - 1), radius=13 * s, fill=INK)

    tile = supersampled(draw, (w, 26))
    ImageDraw.Draw(tile).text((w / 2, 13), text, font=f, fill="#FFFFFF", anchor="mm")
    return tile


def scaled(tile: Image.Image, scale: float) -> Image.Image:
    return tile.resize((max(1, round(tile.width * scale)), max(1, round(tile.height * scale))), Image.LANCZOS)


def with_alpha(tile: Image.Image, alpha: float) -> Image.Image:
    out = tile.copy()
    out.putalpha(out.getchannel("A").point(lambda a: round(a * alpha)))
    return out


def card_base(box: Tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = box
    pad = 24
    shadow = Image.new("RGBA", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((pad, pad + 6, pad + x1 - x0, pad + y1 - y0 + 6), radius=20,
                                             fill=(15, 23, 42, 46))  # fmt: skip
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))

    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        d.rounded_rectangle((pad * s, pad * s, (pad + x1 - x0) * s, (pad + y1 - y0) * s), radius=20 * s,
                            fill="#FFFFFF")  # fmt: skip

    return Image.alpha_composite(shadow, supersampled(draw, shadow.size))


def check_icon(d: ImageDraw.ImageDraw, cx: float, cy: float) -> None:
    d.line([(cx - 4, cy), (cx - 1, cy + 3), (cx + 4, cy - 3)], fill="#FFFFFF", width=2)


def order_card(p: float, delivered: bool) -> Image.Image:
    x0, y0, x1, y1 = CARD1
    tile = card_base(CARD1)
    ox, oy = 24 - x0, 24 - y0  # frame coords -> tile coords
    d = ImageDraw.Draw(tile)
    d.text((900 + ox, 68 + oy), "Order #4821", font=font("SemiBold", 16), fill=MUTED, anchor="lm")
    d.text((900 + ox, 106 + oy), "Delivered" if delivered else "On the way", font=font("Bold", 30), fill=INK,
           anchor="lm")  # fmt: skip
    if delivered:
        d.text((900 + ox, 148 + oy), "Arrived at 14:32", font=font("Medium", 18), fill=GREEN, anchor="lm")
    else:
        minutes = math.ceil(ETA_MIN * (1 - p) - 1e-9)
        d.text((900 + ox, 148 + oy), f"Arriving in {minutes} min", font=font("Medium", 18), fill=BLUE, anchor="lm")
    d.rounded_rectangle((900 + ox, 182 + oy, 1220 + ox, 192 + oy), radius=5, fill=TRACK)
    if p > 0:
        d.rounded_rectangle((900 + ox, 182 + oy, 900 + ox + max(10, round(320 * p)), 192 + oy), radius=5,
                            fill=GREEN if delivered else BLUE)  # fmt: skip
    d.text((900 + ox, 222 + oy), f"{ROUTE_KM * (1 - p):.1f} km left", font=font("Regular", 15), fill=MUTED,
           anchor="lm")  # fmt: skip
    return tile


def driver_card(delivered: bool) -> Image.Image:
    x0, y0, x1, y1 = CARD2
    tile = card_base(CARD2)
    ox, oy = 24 - x0, 24 - y0

    def draw(d: ImageDraw.ImageDraw, s: int) -> None:
        d.ellipse(((898 + ox) * s, (298 + oy) * s, (950 + ox) * s, (350 + oy) * s), fill="#DBEAFE")
        # Stepper connectors and circles.
        for ya, yb, done in ((408, 448, True), (448, 488, delivered)):
            d.line(((908 + ox) * s, (ya + 10 + oy) * s, (908 + ox) * s, (yb - 10 + oy) * s),
                   fill=GREEN if done else "#CBD5E1", width=2 * s)  # fmt: skip
        states = ["done", "done" if delivered else "active", "done" if delivered else "pending"]
        for y, state in zip((408, 448, 488), states):
            box = ((899 + ox) * s, (y - 9 + oy) * s, (917 + ox) * s, (y + 9 + oy) * s)
            if state == "done":
                d.ellipse(box, fill=GREEN)
            elif state == "active":
                d.ellipse(box, fill="#FFFFFF", outline=BLUE, width=3 * s)
                d.ellipse(((904 + ox) * s, (y - 4 + oy) * s, (912 + ox) * s, (y + 4 + oy) * s), fill=BLUE)
            else:
                d.ellipse(box, fill="#FFFFFF", outline="#CBD5E1", width=2 * s)

    tile.alpha_composite(supersampled(draw, tile.size))
    d = ImageDraw.Draw(tile)
    d.text((924 + ox, 324 + oy), "MK", font=font("SemiBold", 18), fill="#1D4ED8", anchor="mm")
    d.text((966 + ox, 312 + oy), "Mara Kowalski", font=font("SemiBold", 18), fill=INK, anchor="lm")
    d.text((966 + ox, 338 + oy), "Cargo e-bike · NB-204", font=font("Regular", 14), fill=MUTED, anchor="lm")
    d.line((900 + ox, 372 + oy, 1220 + ox, 372 + oy), fill=TRACK, width=1)
    labels = ["Picked up · 14:20", "On the way", "Delivered · 14:32" if delivered else "Delivered"]
    for y, text, active in zip((408, 448, 488), labels, (False, not delivered, delivered)):
        d.text((932 + ox, y + oy), text, font=font("SemiBold" if active else "Medium", 16),
               fill=INK if active or text.startswith("Picked") else MUTED, anchor="lm")  # fmt: skip
    for y, done in zip((408, 448, 488), (True, delivered, delivered)):
        if done:
            check_icon(d, 908 + ox, y + oy)
    return tile


def main() -> None:
    base = base_map()
    depot, drop, vehicle = depot_tile(), drop_tile(), vehicle_tile()
    depot_chip, drop_chip = chip("Depot"), chip("Drop-off")
    driver_live, driver_done = driver_card(False), driver_card(True)
    route_full = route_layer(ROUTE_LEN, 0.0)

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "video.mp4",
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(round(DURATION * FPS)):
        t = i / FPS
        frame = base.copy()

        # Route: drawn from the depot to the drop-off between 0.6 and 2.0 s; travelled part turns grey.
        drawn = ROUTE_LEN * clamp01((t - 0.6) / 1.4)
        p = ease_in_out_sine((t - 2.0) / 6.0)
        travelled = ROUTE_LEN * p
        if drawn >= ROUTE_LEN and travelled <= 0:
            frame.alpha_composite(route_full, ROUTE_BOX[:2])
        elif drawn > 0:
            frame.alpha_composite(route_layer(drawn, travelled), ROUTE_BOX[:2])

        # Pins drop in with easeOutBack (scale about the anchor point).
        e = ease_out_back(t / 0.4)
        if e > 0:
            tile = scaled(depot, e)
            frame.alpha_composite(tile, (round(DEPOT[0] - tile.width / 2), round(DEPOT[1] - tile.height / 2)))
            chip_tile = with_alpha(depot_chip, clamp01(t / 0.4))
            frame.alpha_composite(chip_tile, (round(DEPOT[0] - chip_tile.width / 2), DEPOT[1] - 50))
        e = ease_out_back((t - 0.2) / 0.4)
        if e > 0:
            tile = scaled(drop, e)
            frame.alpha_composite(tile, (round(DROP[0] - tile.width / 2), round(DROP[1] - tile.height * 50 / 52)))
            chip_tile = with_alpha(drop_chip, clamp01((t - 0.2) / 0.4))
            frame.alpha_composite(chip_tile, (DROP[0] + 26, DROP[1] - 52))

        # Vehicle travels 2.0 to 8.0 s with easeInOutSine along the route (arc length).
        if t >= 2.0 and t < 8.0:
            vx, vy = point_at(travelled)
            frame.alpha_composite(vehicle, (round(vx - 16), round(vy - 16)))

        # Arrival pulse around the drop-off pin head, 8.0 to 8.8 s.
        if 8.0 <= t < 8.8:
            u = (t - 8.0) / 0.8
            radius = 14 + 50 * ease_out_cubic(u)
            cx, cy = DROP[0], DROP[1] - 30

            def pulse(d: ImageDraw.ImageDraw, s: int) -> None:
                c = 70  # the ring is drawn in a 140x140 tile centred on the pin head
                d.ellipse(((c - radius) * s, (c - radius) * s, (c + radius) * s, (c + radius) * s),
                          outline=rgba(RED, 0.6 * (1 - u)), width=3 * s)  # fmt: skip

            frame.alpha_composite(supersampled(pulse, (140, 140)), (round(cx) - 70, round(cy) - 70))

        # Cards slide in from the right (40 px) and fade in.
        delivered = t >= 8.0
        for card, start in ((order_card(p if t >= 2.0 else 0.0, delivered), 0.2),
                            (driver_done if delivered else driver_live, 0.35)):  # fmt: skip
            e = ease_out_cubic((t - start) / 0.5)
            if e <= 0:
                continue
            x0, y0 = (CARD1 if start == 0.2 else CARD2)[:2]
            tile = card if e >= 1 else with_alpha(card, e)
            frame.alpha_composite(tile, (x0 - 24 + round(40 * (1 - e)), y0 - 24))

        proc.stdin.write(frame.convert("RGB").tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")


if __name__ == "__main__":
    main()
