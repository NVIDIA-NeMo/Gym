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
"""Render a Figma-style design file (subset of the Figma REST file format) to absolute-positioned HTML.

Figma replication tasks give the policy `design.json` plus the rendered `reference.png`, and ask
for a real HTML/CSS implementation. This module produces the hidden golden page that
`reference.png` is rendered from, so the JSON and the image always agree.

Supported nodes: DOCUMENT > CANVAS > FRAME artboards (one, or several for a responsive design, e.g.
Desktop + Mobile) containing FRAME, GROUP, RECTANGLE,
ELLIPSE, LINE and TEXT. Supported properties: absoluteBoundingBox, fills (SOLID,
GRADIENT_LINEAR), strokes + strokeWeight, cornerRadius / rectangleCornerRadii, effects
(DROP_SHADOW, INNER_SHADOW), opacity, visible, clipsContent. TEXT uses characters and style
(fontFamily, fontWeight, fontSize, lineHeightPx, letterSpacing, textAlignHorizontal,
textAlignVertical, italic, textCase, textDecoration).
"""

import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


SUPPORTED_TYPES = {"FRAME", "GROUP", "RECTANGLE", "ELLIPSE", "LINE", "TEXT"}


def _rgba(color: Dict[str, float], opacity: float = 1.0) -> str:
    r, g, b = (round(255 * float(color.get(k, 0.0))) for k in ("r", "g", "b"))
    a = float(color.get("a", 1.0)) * opacity
    return f"rgba({r},{g},{b},{a:.3f})"


def _paint_css(paint: Dict[str, Any]) -> Optional[str]:
    if paint.get("visible") is False:
        return None
    opacity = float(paint.get("opacity", 1.0))
    if paint["type"] == "SOLID":
        return _rgba(paint["color"], opacity)
    if paint["type"] == "GRADIENT_LINEAR":
        handles = paint.get("gradientHandlePositions") or [{"x": 0.5, "y": 0}, {"x": 0.5, "y": 1}]
        dx = handles[1]["x"] - handles[0]["x"]
        dy = handles[1]["y"] - handles[0]["y"]
        angle = (math.degrees(math.atan2(dy, dx)) + 90) % 360
        stops = ", ".join(
            f"{_rgba(s['color'], opacity)} {100 * float(s['position']):.1f}%" for s in paint["gradientStops"]
        )
        return f"linear-gradient({angle:.1f}deg, {stops})"
    raise ValueError(f"Unsupported paint type {paint['type']!r}")


def _box(node: Dict[str, Any], origin: Dict[str, float]) -> Dict[str, float]:
    bb = node["absoluteBoundingBox"]
    return {
        "left": bb["x"] - origin["x"],
        "top": bb["y"] - origin["y"],
        "width": bb["width"],
        "height": bb["height"],
    }


def _node_style(node: Dict[str, Any], box: Dict[str, float]) -> List[str]:
    style = [
        "position:absolute",
        f"left:{box['left']:g}px",
        f"top:{box['top']:g}px",
        f"width:{box['width']:g}px",
        f"height:{box['height']:g}px",
        "box-sizing:border-box",
    ]
    if node.get("opacity") is not None and float(node["opacity"]) < 1:
        style.append(f"opacity:{float(node['opacity']):.3f}")
    kind = node["type"]
    shadows = []
    if kind != "TEXT":
        fills = [c for c in (_paint_css(p) for p in node.get("fills") or []) if c]
        if fills:
            # CSS lists the top layer first; Figma lists the bottom layer first.
            layers = [f if f.startswith("linear-gradient") else f"linear-gradient({f},{f})" for f in reversed(fills)]
            style.append("background:" + ",".join(layers))
        strokes = [c for c in (_paint_css(p) for p in node.get("strokes") or []) if c]
        if strokes and node.get("strokeWeight") and kind != "LINE":
            # Figma strokes sit inside the box without moving children; a CSS border would shift them.
            shadows.append(f"inset 0 0 0 {float(node['strokeWeight']):g}px {strokes[0]}")
        if kind == "ELLIPSE":
            style.append("border-radius:50%")
        elif node.get("rectangleCornerRadii"):
            style.append("border-radius:" + " ".join(f"{float(r):g}px" for r in node["rectangleCornerRadii"]))
        elif node.get("cornerRadius"):
            style.append(f"border-radius:{float(node['cornerRadius']):g}px")
        if node.get("clipsContent"):
            style.append("overflow:hidden")
    for effect in node.get("effects") or []:
        if effect.get("visible") is False or effect["type"] not in ("DROP_SHADOW", "INNER_SHADOW"):
            continue
        offset = effect.get("offset") or {"x": 0, "y": 0}
        inset = "inset " if effect["type"] == "INNER_SHADOW" else ""
        shadows.append(
            f"{inset}{offset['x']:g}px {offset['y']:g}px {float(effect.get('radius', 0)):g}px "
            f"{float(effect.get('spread', 0)):g}px {_rgba(effect['color'])}"
        )
    if shadows:
        style.append("box-shadow:" + ",".join(shadows))
    return style


def _text_style(node: Dict[str, Any]) -> List[str]:
    s = node.get("style") or {}
    fills = [c for c in (_paint_css(p) for p in node.get("fills") or []) if c]
    align_h = {"LEFT": "left", "CENTER": "center", "RIGHT": "right", "JUSTIFIED": "justify"}
    align_v = {"TOP": "flex-start", "CENTER": "center", "BOTTOM": "flex-end"}
    justify = {"LEFT": "flex-start", "CENTER": "center", "RIGHT": "flex-end", "JUSTIFIED": "stretch"}
    style = [
        "display:flex",
        "flex-direction:column",
        f"justify-content:{align_v.get(s.get('textAlignVertical', 'TOP'), 'flex-start')}",
        f"align-items:{justify.get(s.get('textAlignHorizontal', 'LEFT'), 'flex-start')}",
        f"text-align:{align_h.get(s.get('textAlignHorizontal', 'LEFT'), 'left')}",
        f"font-family:'{s.get('fontFamily', 'Inter')}', sans-serif",
        f"font-size:{float(s.get('fontSize', 14)):g}px",
        f"font-weight:{int(s.get('fontWeight', 400))}",
        "white-space:pre-wrap",
        "margin:0",
    ]
    if s.get("lineHeightPx"):
        style.append(f"line-height:{float(s['lineHeightPx']):g}px")
    if s.get("letterSpacing"):
        style.append(f"letter-spacing:{float(s['letterSpacing']):g}px")
    if s.get("italic"):
        style.append("font-style:italic")
    if s.get("textCase") == "UPPER":
        style.append("text-transform:uppercase")
    if s.get("textDecoration") == "UNDERLINE":
        style.append("text-decoration:underline")
    if s.get("textDecoration") == "STRIKETHROUGH":
        style.append("text-decoration:line-through")
    if fills:
        style.append(f"color:{fills[0]}")
    return style


def _render(node: Dict[str, Any], origin: Dict[str, float], out: List[str], depth: int) -> None:
    if node.get("visible") is False:
        return
    kind = node["type"]
    if kind not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported node type {kind!r} (node {node.get('name')!r})")
    box = _box(node, origin)
    style = _node_style(node, box)
    indent = "  " * depth
    name = html.escape(node.get("name", ""), quote=True)
    if kind == "TEXT":
        style += _text_style(node)
        text = html.escape(node.get("characters", ""))
        out.append(f'{indent}<div data-name="{name}" style="{";".join(style)}"><span>{text}</span></div>')
        return
    if kind == "LINE":
        strokes = [c for c in (_paint_css(p) for p in node.get("strokes") or []) if c]
        weight = float(node.get("strokeWeight", 1))
        horizontal = box["width"] >= box["height"]
        style = [
            "position:absolute",
            f"left:{box['left']:g}px",
            f"top:{box['top']:g}px",
            f"width:{box['width'] if horizontal else weight:g}px",
            f"height:{weight if horizontal else box['height']:g}px",
            f"background:{strokes[0] if strokes else 'rgba(0,0,0,1)'}",
        ]
        out.append(f'{indent}<div data-name="{name}" style="{";".join(style)}"></div>')
        return
    children = node.get("children") or []
    out.append(f'{indent}<div data-name="{name}" style="{";".join(style)}">')
    # Children are positioned relative to this node, since it is `position:absolute`.
    child_origin = {"x": node["absoluteBoundingBox"]["x"], "y": node["absoluteBoundingBox"]["y"]}
    for child in children:
        _render(child, child_origin, out, depth + 1)
    out.append(f"{indent}</div>")


def root_frames(design: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Top-level FRAMEs of the first page, widest first."""
    canvases = design["document"]["children"]
    frames = [n for n in canvases[0]["children"] if n["type"] == "FRAME"]
    if not frames:
        raise ValueError("Expected at least one top-level FRAME on the first page")
    return sorted(frames, key=lambda f: -f["absoluteBoundingBox"]["width"])


def design_to_html(design: Dict[str, Any]) -> str:
    """One page for the design. With several top-level frames (e.g. Desktop and Mobile), each frame
    is shown at viewports closest to its width, switching at the midpoints between frame widths."""
    frames = root_frames(design)
    title = html.escape(design.get("name") or frames[0].get("name") or "Design")
    body: List[str] = []
    css = ["html,body{margin:0;padding:0;background:#ffffff}", ".frame{position:relative}"]
    for index, frame in enumerate(frames):
        bb = frame["absoluteBoundingBox"]
        body.append(f'  <div class="frame" id="frame-{index}">')
        _render(frame, {"x": bb["x"], "y": bb["y"]}, body, 2)
        body.append("  </div>")
        css.append(f"#frame-{index}{{width:{bb['width']:g}px;height:{bb['height']:g}px}}")
        if len(frames) > 1:
            upper = None if index == 0 else (frames[index - 1]["absoluteBoundingBox"]["width"] + bb["width"]) / 2
            lower = (
                None
                if index == len(frames) - 1
                else (bb["width"] + frames[index + 1]["absoluteBoundingBox"]["width"]) / 2
            )
            hidden = []
            if lower is not None:
                hidden.append(f"@media (max-width:{lower - 0.02:g}px){{#frame-{index}{{display:none}}}}")
            if upper is not None:
                hidden.append(f"@media (min-width:{upper:g}px){{#frame-{index}{{display:none}}}}")
            css.extend(hidden)
    return (
        '<!doctype html>\n<html>\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>{title}</title>\n<style>{''.join(css)}</style>\n"
        "</head>\n<body>\n" + "\n".join(body) + "\n</body>\n</html>\n"
    )


def count_nodes(design: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    def walk(node: Dict[str, Any]) -> None:
        counts[node["type"]] = counts.get(node["type"], 0) + 1
        for child in node.get("children") or []:
            walk(child)

    for frame in root_frames(design):
        walk(frame)
    return counts


if __name__ == "__main__":
    import sys

    source = Path(sys.argv[1])
    print(design_to_html(json.loads(source.read_text())))
