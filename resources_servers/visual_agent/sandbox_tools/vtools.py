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
"""Rendering, interaction and measurement tools for visual artifacts.

Runs inside the task sandbox (Python 3.10+, Playwright with Chromium, numpy, Pillow, ffmpeg).
It uses only those packages and has no nemo_gym import, so it can be uploaded as one file.

Subcommands:
  preview   render an artifact to PNGs and print a runtime/layout report (for the policy)
  interact  drive an HTML artifact with a list of actions and capture screenshots (policy and judge)
  compare   visual similarity between two images
  measure   the grader's deterministic pass: renders, runtime checks, similarity to the
            reference and reference-copy detection, written as JSON
"""

import argparse
import base64
import functools
import hashlib
import http.server
import io
import json
import math
import re
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


CHROMIUM_ARGS = [
    # Software WebGL for three.js scenes in a GPU-less sandbox.
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--autoplay-policy=no-user-gesture-required",
]
DEFAULT_VIEWPORT = {"width": 1280, "height": 800}
MOBILE_VIEWPORT = {"width": 390, "height": 844}
SLIDE_VIEWPORT = {"width": 1280, "height": 720}
MAX_FULL_PAGE_HEIGHT = 6000
MAX_SLIDES = 40
SETTLE_MS = 1500
RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
TEXT_EXTENSIONS = {".html", ".htm", ".css", ".js", ".mjs", ".svg", ".json", ".txt", ".md"}
DATA_URL_RE = re.compile(r"data:image/(?:png|jpe?g|webp|gif);base64,([A-Za-z0-9+/=]{200,})")


# ----------------------------------------------------------------------------------------------
# Static file server: ES modules and fetch() do not work from file:// origins.
# ----------------------------------------------------------------------------------------------
class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class StaticServer:
    """Serve a directory on 127.0.0.1 for the lifetime of a `with` block."""

    def __init__(self, root: Path) -> None:
        handler = functools.partial(_QuietHandler, directory=str(root))
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "StaticServer":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def url(self, rel_path: str) -> str:
        return f"http://127.0.0.1:{self.port}/{rel_path.lstrip('/')}"


class PageRecorder:
    """Attach to a Playwright page and record errors and blocked network requests."""

    def __init__(self, page: Any, allowed_origin: str) -> None:
        self.page_errors: List[str] = []
        self.console_errors: List[str] = []
        self.failed_requests: List[str] = []
        self.external_requests: List[str] = []
        page.on("pageerror", lambda exc: self.page_errors.append(str(exc)[:500]))
        page.on("console", self._on_console)
        page.on("requestfailed", lambda req: self.failed_requests.append(req.url[:300]))
        page.on("response", self._on_response)

        def route(route: Any) -> None:
            url = route.request.url
            if url.startswith(allowed_origin) or url.startswith("data:") or url.startswith("blob:"):
                route.continue_()
            else:
                self.external_requests.append(url[:300])
                route.abort()

        page.route("**/*", route)

    def _on_console(self, msg: Any) -> None:
        if msg.type == "error":
            self.console_errors.append(msg.text[:500])

    def _on_response(self, response: Any) -> None:
        if response.status >= 400:
            self.failed_requests.append(f"{response.status} {response.url[:300]}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "page_errors": self.page_errors[:20],
            "console_errors": self.console_errors[:20],
            "failed_requests": sorted(set(self.failed_requests))[:20],
            "external_requests": sorted(set(self.external_requests))[:20],
        }


def _launch(playwright: Any) -> Any:
    return playwright.chromium.launch(args=CHROMIUM_ARGS)


# ----------------------------------------------------------------------------------------------
# Image statistics and similarity
# ----------------------------------------------------------------------------------------------
def load_rgb(path_or_bytes: Any) -> np.ndarray:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        image = Image.open(io.BytesIO(path_or_bytes))
    else:
        image = Image.open(path_or_bytes)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def image_stats(rgb: np.ndarray) -> Dict[str, Any]:
    gray = rgb.mean(axis=2)
    small = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((64, 64)), dtype=np.int32)
    quantized = (small // 32).reshape(-1, 3)
    codes = quantized[:, 0] * 64 + quantized[:, 1] * 8 + quantized[:, 2]
    counts = np.bincount(codes, minlength=512)
    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "gray_std": round(float(gray.std()), 3),
        "dominant_color_fraction": round(float(counts.max() / counts.sum()), 4),
        "distinct_coarse_colors": int((counts > 0).sum()),
    }


def is_blank(stats: Dict[str, Any]) -> bool:
    return stats["gray_std"] < 2.0 or stats["dominant_color_fraction"] > 0.995


def _resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    return np.asarray(image.resize((width, height), Image.BILINEAR), dtype=np.float32)


def align_to_reference(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Scale the candidate to the reference width, then crop or white-pad it to the reference height."""
    ref_h, ref_w = reference.shape[:2]
    cand_h, cand_w = candidate.shape[:2]
    if cand_w != ref_w:
        new_h = max(1, round(cand_h * ref_w / cand_w))
        candidate = _resize(candidate, ref_w, new_h)
        cand_h = new_h
    if cand_h >= ref_h:
        return candidate[:ref_h]
    pad = np.full((ref_h - cand_h, ref_w, 3), 255.0, dtype=np.float32)
    return np.concatenate([candidate, pad], axis=0)


def _box_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Mean over k x k windows (valid region) via integral images."""
    s = np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]) / float(k * k)


def ssim(a: np.ndarray, b: np.ndarray, k: int = 7) -> float:
    """Mean SSIM of two grayscale images of equal shape (uniform window, as scikit-image's default)."""
    if min(a.shape) < k:
        return float(1.0 - np.abs(a - b).mean() / 255.0)
    # float64: E[x^2] - E[x]^2 from float32 cumulative sums goes negative in flat regions and
    # pushed SSIM above 1.
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = _box_filter(a, k), _box_filter(b, k)
    var_a = np.maximum(_box_filter(a * a, k) - mu_a**2, 0.0)
    var_b = np.maximum(_box_filter(b * b, k) - mu_b**2, 0.0)
    cov = _box_filter(a * b, k) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(np.clip(num / den, -1.0, 1.0).mean())


def _edges(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.hypot(gx, gy) > 40.0


def _dilate(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


def edge_f1(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    ea, eb = _edges(a_gray), _edges(b_gray)
    if not ea.any() and not eb.any():
        return 1.0
    if not ea.any() or not eb.any():
        return 0.0
    precision = (ea & _dilate(eb)).sum() / ea.sum()
    recall = (eb & _dilate(ea)).sum() / eb.sum()
    return float(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))


def color_histogram_similarity(a: np.ndarray, b: np.ndarray, bins: int = 8) -> float:
    def hist(x: np.ndarray) -> np.ndarray:
        q = np.clip((x // (256 // bins)).astype(np.int64), 0, bins - 1).reshape(-1, 3)
        codes = q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2]
        h = np.bincount(codes, minlength=bins**3).astype(np.float64)
        return h / h.sum()

    return float(np.minimum(hist(a), hist(b)).sum())


SIMILARITY_WEIGHTS = {"ssim": 0.35, "pixel": 0.25, "edge": 0.25, "color_hist": 0.15}


def _raw_similarity(candidate: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    """Similarity of two RGB arrays of equal shape, compared at most 512 px on the long side."""
    h, w = reference.shape[:2]
    scale = min(1.0, 512.0 / max(h, w))
    size = (max(8, round(w * scale)), max(8, round(h * scale)))
    a = _resize(candidate, *size)
    b = _resize(reference, *size)
    a_gray, b_gray = a.mean(axis=2), b.mean(axis=2)
    metrics = {
        "ssim": ssim(a_gray, b_gray),
        "pixel": 1.0 - float(np.abs(a - b).mean()) / 255.0,
        "edge": edge_f1(a_gray, b_gray),
        "color_hist": color_histogram_similarity(a, b),
    }
    metrics = {k: min(1.0, max(0.0, float(v))) for k, v in metrics.items()}
    metrics["combined"] = sum(SIMILARITY_WEIGHTS[k] * metrics[k] for k in SIMILARITY_WEIGHTS)
    return metrics


def visual_similarity(candidate: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
    """Similarity in [0, 1] plus a floor-normalized score.

    The floor is the similarity a flat canvas in the reference's median color would get, so a
    blank or background-only attempt scores ~0 whatever the reference's background is.
    """
    aligned = align_to_reference(candidate, reference)
    metrics = _raw_similarity(aligned, reference)
    flat = np.empty_like(reference)
    flat[...] = np.median(reference.reshape(-1, 3), axis=0)
    floor = _raw_similarity(flat, reference)["combined"]
    normalized = 0.0 if floor >= 0.999 else (metrics["combined"] - floor) / (1.0 - floor)
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    metrics["floor"] = round(floor, 4)
    metrics["normalized"] = round(float(min(1.0, max(0.0, normalized))), 4)
    metrics["size_candidate"] = [int(candidate.shape[1]), int(candidate.shape[0])]
    metrics["size_reference"] = [int(reference.shape[1]), int(reference.shape[0])]
    return metrics


def contact_sheet(paths: List[Path], out: Path, columns: int = 3, thumb_width: int = 480) -> Optional[Path]:
    images = [Image.open(p).convert("RGB") for p in paths if p.exists()]
    if not images:
        return None
    thumbs = [im.resize((thumb_width, max(1, round(im.height * thumb_width / im.width)))) for im in images]
    rows = math.ceil(len(thumbs) / columns)
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (columns * (thumb_width + 8) + 8, rows * (cell_h + 8) + 8), (40, 40, 40))
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, columns)
        sheet.paste(thumb, (8 + c * (thumb_width + 8), 8 + r * (cell_h + 8)))
    sheet.save(out)
    return out


# ----------------------------------------------------------------------------------------------
# Renderers
# ----------------------------------------------------------------------------------------------
def _settle(page: Any, ms: int = SETTLE_MS) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=10000)
    except Exception:
        pass
    page.wait_for_timeout(ms)


def _layout_probe(page: Any) -> Dict[str, Any]:
    return page.evaluate(
        """() => {
        const doc = document.documentElement;
        const imgs = Array.from(document.images);
        const texts = Array.from(document.querySelectorAll('h1,h2,h3,h4,p,li,a,button,span,label,td,th'))
            .filter(el => el.offsetParent !== null && el.innerText && el.innerText.trim().length > 0);
        let clipped = 0;
        for (const el of texts) {
            const cs = getComputedStyle(el);
            if ((cs.overflow === 'hidden' || cs.textOverflow === 'ellipsis') &&
                (el.scrollWidth > el.clientWidth + 2 || el.scrollHeight > el.clientHeight + 2)) clipped++;
        }
        return {
            scroll_width: doc.scrollWidth, scroll_height: doc.scrollHeight,
            viewport_width: window.innerWidth, viewport_height: window.innerHeight,
            horizontal_overflow_px: Math.max(0, doc.scrollWidth - window.innerWidth),
            images: imgs.length, broken_images: imgs.filter(i => i.complete && i.naturalWidth === 0).length,
            canvases: document.querySelectorAll('canvas').length,
            text_elements: texts.length, clipped_text_elements: clipped,
            title: document.title,
        };
    }"""
    )


def render_html(
    playwright: Any,
    artifact_dir: Path,
    entry: str,
    out_dir: Path,
    viewport: Dict[str, int],
    *,
    mobile: bool,
    animation: bool,
    compare_sizes: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"renders": {}}
    with StaticServer(artifact_dir) as server:
        browser = _launch(playwright)
        try:
            page = browser.new_page(viewport=viewport)
            recorder = PageRecorder(page, f"http://127.0.0.1:{server.port}")
            start = time.time()
            response = page.goto(server.url(entry), wait_until="load", timeout=30000)
            report["http_status"] = response.status if response else None
            _settle(page)
            report["load_seconds"] = round(time.time() - start, 2)
            report["layout"] = _layout_probe(page)

            viewport_png = out_dir / "desktop.png"
            page.screenshot(path=str(viewport_png))
            report["renders"]["desktop"] = str(viewport_png)
            if report["layout"]["scroll_height"] > viewport["height"] + 8:
                full_png = out_dir / "desktop_full.png"
                height = min(report["layout"]["scroll_height"], MAX_FULL_PAGE_HEIGHT)
                page.screenshot(
                    path=str(full_png),
                    full_page=True,
                    clip={"x": 0, "y": 0, "width": viewport["width"], "height": height},
                )
                report["renders"]["desktop_full"] = str(full_png)

            if animation:
                first = load_rgb(viewport_png)
                page.wait_for_timeout(1500)
                later_png = out_dir / "desktop_t2.png"
                page.screenshot(path=str(later_png))
                report["renders"]["desktop_t2"] = str(later_png)
                diff = float(np.abs(load_rgb(later_png) - first).mean())
                report["animation_mean_abs_diff"] = round(diff, 3)
                report["animated"] = diff > 0.05
            report["runtime"] = recorder.as_dict()
            page.close()

            # Full-page renders at each comparison viewport (one per reference image, in order).
            for index, (width, height) in enumerate(compare_sizes or []):
                name = compare_render_name(index)
                page = browser.new_page(viewport={"width": width, "height": height})
                PageRecorder(page, f"http://127.0.0.1:{server.port}")
                page.goto(server.url(entry), wait_until="load", timeout=30000)
                _settle(page)
                compare_png = out_dir / f"{name}.png"
                page.screenshot(path=str(compare_png), full_page=True)
                report["renders"][name] = str(compare_png)
                page.close()

            if mobile:
                page = browser.new_page(viewport=MOBILE_VIEWPORT, is_mobile=True, has_touch=True)
                mobile_recorder = PageRecorder(page, f"http://127.0.0.1:{server.port}")
                page.goto(server.url(entry), wait_until="load", timeout=30000)
                _settle(page)
                mobile_png = out_dir / "mobile.png"
                page.screenshot(path=str(mobile_png))
                report["renders"]["mobile"] = str(mobile_png)
                report["mobile_layout"] = _layout_probe(page)
                report["mobile_runtime"] = mobile_recorder.as_dict()
                page.close()
        finally:
            browser.close()
    return report


def render_slides(playwright: Any, artifact_dir: Path, entry: str, out_dir: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {"renders": {}}
    with StaticServer(artifact_dir) as server:
        browser = _launch(playwright)
        try:
            page = browser.new_page(viewport=SLIDE_VIEWPORT)
            recorder = PageRecorder(page, f"http://127.0.0.1:{server.port}")
            response = page.goto(server.url(entry), wait_until="load", timeout=30000)
            report["http_status"] = response.status if response else None
            _settle(page)
            # Show every slide stacked so each can be captured, whatever navigation the deck uses. Hidden
            # slides get the display type of the visible one, so flex/grid slide layouts survive.
            display = page.evaluate(
                """() => {
                const shown = Array.from(document.querySelectorAll('.slide'))
                    .find(el => getComputedStyle(el).display !== 'none');
                return shown ? getComputedStyle(shown).display : 'block';
            }"""
            )
            page.add_style_tag(
                content=f".slide{{display:{display}!important;position:relative!important;visibility:visible!important;"
                "opacity:1!important;transform:none!important;left:auto!important;top:auto!important;"
                "margin:0 0 16px 0!important;}"
            )
            report["slide_display"] = display
            page.wait_for_timeout(300)
            slides = page.query_selector_all(".slide")
            report["slide_count"] = len(slides)
            overflow: List[Dict[str, Any]] = []
            paths: List[Path] = []
            for index, slide in enumerate(slides[:MAX_SLIDES], start=1):
                box = slide.bounding_box() or {"width": 0, "height": 0}
                info = slide.evaluate(
                    "el => ({sw: el.scrollWidth, sh: el.scrollHeight, cw: el.clientWidth, ch: el.clientHeight})"
                )
                if info["sh"] > info["ch"] + 2 or info["sw"] > info["cw"] + 2:
                    overflow.append({"slide": index, **info})
                path = out_dir / f"slide_{index:02d}.png"
                slide.screenshot(path=str(path))
                paths.append(path)
                report["renders"][f"slide_{index:02d}"] = str(path)
                report.setdefault("slide_sizes", []).append([round(box["width"]), round(box["height"])])
            report["overflowing_slides"] = overflow
            sheet = contact_sheet(paths, out_dir / "slides_overview.png")
            if sheet:
                report["renders"]["slides_overview"] = str(sheet)
            report["runtime"] = recorder.as_dict()
        finally:
            browser.close()
    return report


def svg_facts(svg_path: Path) -> Dict[str, Any]:
    facts: Dict[str, Any] = {"bytes": svg_path.stat().st_size}
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError as exc:
        facts["well_formed"] = False
        facts["parse_error"] = str(exc)
        return facts
    facts["well_formed"] = True
    facts["root_tag"] = root.tag.split("}")[-1]
    for attr in ("width", "height", "viewBox"):
        facts[attr] = root.get(attr)
    tags: Dict[str, int] = {}
    named_groups: List[str] = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        tags[tag] = tags.get(tag, 0) + 1
        if tag == "g" and (el.get("id") or el.get("data-name")):
            named_groups.append(el.get("id") or el.get("data-name"))
    facts["element_counts"] = dict(sorted(tags.items(), key=lambda kv: -kv[1])[:25])
    facts["total_elements"] = sum(tags.values())
    facts["embedded_raster_images"] = tags.get("image", 0)
    facts["has_script"] = tags.get("script", 0) > 0
    facts["named_groups"] = named_groups[:60]
    facts["top_level_groups"] = [
        (child.get("id") or child.get("data-name") or "") for child in root if child.tag.split("}")[-1] == "g"
    ][:40]
    return facts


def _svg_render_size(facts: Dict[str, Any], max_side: int = 1200) -> Tuple[int, int]:
    width = height = None
    if facts.get("viewBox"):
        parts = re.split(r"[\s,]+", facts["viewBox"].strip())
        if len(parts) == 4:
            try:
                width, height = float(parts[2]), float(parts[3])
            except ValueError:
                pass
    if width is None:
        try:
            width = float(re.sub(r"[a-z%]+$", "", str(facts.get("width") or "")))
            height = float(re.sub(r"[a-z%]+$", "", str(facts.get("height") or "")))
        except ValueError:
            width, height = 800.0, 600.0
    if not width or not height:
        width, height = 800.0, 600.0
    scale = max_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def render_svg(
    playwright: Any,
    artifact_dir: Path,
    entry: str,
    out_dir: Path,
    compare_size: Optional[Tuple[int, int]] = None,
    *,
    animation: bool = False,
) -> Dict[str, Any]:
    svg_path = artifact_dir / entry
    facts = svg_facts(svg_path)
    report: Dict[str, Any] = {"renders": {}, "svg": facts}
    if not facts.get("well_formed"):
        report["runtime"] = {"page_errors": [facts.get("parse_error", "SVG parse error")]}
        return report
    width, height = compare_size or _svg_render_size(facts)
    wrapper = artifact_dir / "__vtools_svg_wrapper.html"
    wrapper.write_text(
        "<!doctype html><html><head><style>html,body{margin:0;background:#fff}"
        f"img{{display:block;width:{width}px;height:{height}px}}</style></head>"
        f'<body><img src="{entry}"></body></html>'
    )
    try:
        with StaticServer(artifact_dir) as server:
            browser = _launch(playwright)
            try:
                page = browser.new_page(viewport={"width": width, "height": height})
                recorder = PageRecorder(page, f"http://127.0.0.1:{server.port}")
                page.goto(server.url(wrapper.name), wait_until="load", timeout=30000)
                _settle(page, 500)
                ok = page.evaluate("() => { const i = document.images[0]; return i.complete && i.naturalWidth > 0; }")
                path = out_dir / ("compare.png" if compare_size else "svg.png")
                page.screenshot(path=str(path))
                report["renders"]["compare" if compare_size else "svg"] = str(path)
                report["svg_image_loaded"] = bool(ok)
                runtime = recorder.as_dict()
                if not ok:
                    runtime["page_errors"].append("The SVG failed to load as an image (invalid SVG or bad size)")
                report["runtime"] = runtime
                if animation and not compare_size:
                    # SMIL and CSS animations run inside <img>; scripts do not (and should not be needed).
                    page.wait_for_timeout(1500)
                    later = out_dir / "svg_t2.png"
                    page.screenshot(path=str(later))
                    report["renders"]["svg_t2"] = str(later)
                    diff = float(np.abs(load_rgb(later) - load_rgb(path)).mean())
                    report["animation_mean_abs_diff"] = round(diff, 3)
                    report["animated"] = diff > 0.05
            finally:
                browser.close()
    finally:
        wrapper.unlink(missing_ok=True)
    return report


def ffprobe(video_path: Path) -> Dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video_path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
    )
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr[-500:]}
    data = json.loads(result.stdout or "{}")
    video_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    info: Dict[str, Any] = {"ok": bool(video_streams), "has_audio": bool(audio_streams)}
    if video_streams:
        v = video_streams[0]
        num, _, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        duration = float(v.get("duration") or data.get("format", {}).get("duration") or 0.0)
        info |= {
            "codec": v.get("codec_name"),
            "width": v.get("width"),
            "height": v.get("height"),
            "fps": round(fps, 3),
            "duration_s": round(duration, 3),
            "nb_frames": int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None,
            "pix_fmt": v.get("pix_fmt"),
        }
    info["format"] = data.get("format", {}).get("format_name")
    return info


def extract_frame(video_path: Path, t: float, out: Path) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video_path), "-frames:v", "1", str(out)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
    )
    return result.returncode == 0 and out.exists()


def render_video(
    artifact_dir: Path, entry: str, out_dir: Path, frame_times: Optional[List[float]] = None
) -> Dict[str, Any]:
    video_path = artifact_dir / entry
    info = ffprobe(video_path)
    report: Dict[str, Any] = {"renders": {}, "video": info}
    if not info.get("ok"):
        report["runtime"] = {"page_errors": [f"ffprobe could not read a video stream: {info.get('error', '')}"]}
        return report
    duration = info.get("duration_s") or 0.0
    times = frame_times or [round(duration * f, 3) for f in (0.02, 0.2, 0.4, 0.6, 0.8, 0.97)]
    paths: List[Path] = []
    for index, t in enumerate(times):
        path = out_dir / f"frame_{index:02d}_{t:.2f}s.png"
        if extract_frame(video_path, min(t, max(0.0, duration - 0.04)), path):
            paths.append(path)
            report["renders"][path.stem] = str(path)
    report["frame_times"] = times
    motion = []
    for a, b in zip(paths, paths[1:]):
        motion.append(round(float(np.abs(load_rgb(a) - load_rgb(b)).mean()), 3))
    report["motion_between_frames"] = motion
    report["animated"] = bool(motion) and max(motion) > 0.5
    sheet = contact_sheet(paths, out_dir / "frames_overview.png")
    if sheet:
        report["renders"]["frames_overview"] = str(sheet)
    report["runtime"] = {"page_errors": [] if paths else ["No frames could be extracted"]}
    return report


# ----------------------------------------------------------------------------------------------
# Reference-copy detection (replication tasks)
# ----------------------------------------------------------------------------------------------
def detect_reference_copy(artifact_dir: Path, reference_paths: List[Path]) -> Dict[str, Any]:
    """Flag artifacts that embed a reference image instead of rebuilding it.

    Checks byte-identical copies, the reference's base64 in any text file, and any raster
    (file or data URL) that looks like the reference after re-encoding or resizing.
    """
    findings: List[str] = []
    if not reference_paths:
        return {"detected": False, "findings": findings}
    ref_bytes = {p.name: p.read_bytes() for p in reference_paths if p.exists()}
    ref_hashes = {hashlib.sha256(b).hexdigest(): name for name, b in ref_bytes.items()}
    ref_b64_prefixes = {name: base64.b64encode(b).decode()[:120] for name, b in ref_bytes.items()}
    ref_small = {name: _resize(load_rgb(b), 128, 128) for name, b in ref_bytes.items()}

    def looks_like_reference(data: bytes, where: str) -> None:
        try:
            rgb = load_rgb(data)
        except Exception:
            return
        if rgb.shape[0] < 32 or rgb.shape[1] < 32:
            return
        small = _resize(rgb, 128, 128)
        for name, ref in ref_small.items():
            score = _raw_similarity(small, ref)["combined"]
            if score > 0.9:
                findings.append(f"{where} is a raster copy of reference {name} (similarity {score:.3f})")

    for path in sorted(artifact_dir.rglob("*")):
        if not path.is_file() or path.stat().st_size > 50 * 1024 * 1024:
            continue
        rel = str(path.relative_to(artifact_dir))
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest in ref_hashes:
            findings.append(f"{rel} is a byte-identical copy of reference {ref_hashes[digest]}")
            continue
        suffix = path.suffix.lower()
        if suffix in RASTER_EXTENSIONS:
            looks_like_reference(data, rel)
        elif suffix in TEXT_EXTENSIONS:
            text = data.decode("utf-8", errors="replace")
            for name, prefix in ref_b64_prefixes.items():
                if prefix in text:
                    findings.append(f"{rel} contains the base64 of reference {name}")
            if "/workspace/task/" in text or "../task/" in text:
                findings.append(f"{rel} loads files from the task directory")
            for match in DATA_URL_RE.finditer(text):
                try:
                    looks_like_reference(base64.b64decode(match.group(1)), f"data URL in {rel}")
                except Exception:
                    continue
    return {"detected": bool(findings), "findings": findings[:20]}


# ----------------------------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------------------------
def _entry_for(task: Dict[str, Any], artifact_dir: Path) -> Tuple[str, Optional[str]]:
    artifact = task["artifact"]
    entry = artifact["entry"]
    if (artifact_dir / entry).is_file():
        return entry, None
    return entry, f"Expected artifact {artifact.get('output_dir', 'the output folder')}/{entry} was not found"


def render_task_artifact(
    task: Dict[str, Any],
    artifact_dir: Path,
    out_dir: Path,
    compare_sizes: Optional[List[Tuple[int, int]]] = None,
    frame_times: Optional[List[float]] = None,
) -> Dict[str, Any]:
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    kind = task["artifact"]["kind"]
    entry, missing = _entry_for(task, artifact_dir)
    files = sorted(str(p.relative_to(artifact_dir)) for p in artifact_dir.rglob("*") if p.is_file())
    report: Dict[str, Any] = {"kind": kind, "entry": entry, "artifact_files": files[:200], "file_count": len(files)}
    report["artifact_bytes"] = sum((artifact_dir / f).stat().st_size for f in files)
    if missing:
        report |= {"artifact_found": False, "error": missing, "renders": {}}
        return report
    report["artifact_found"] = True
    checks = task.get("checks") or {}
    try:
        if kind == "video":
            report |= render_video(artifact_dir, entry, out_dir, frame_times)
        else:
            with sync_playwright() as playwright:
                if kind == "html":
                    report |= render_html(
                        playwright,
                        artifact_dir,
                        entry,
                        out_dir,
                        task.get("viewport") or DEFAULT_VIEWPORT,
                        mobile=bool(checks.get("mobile")),
                        animation=bool(checks.get("animation")),
                        compare_sizes=compare_sizes,
                    )
                elif kind == "slides":
                    report |= render_slides(playwright, artifact_dir, entry, out_dir)
                elif kind == "svg":
                    report |= render_svg(
                        playwright, artifact_dir, entry, out_dir, animation=bool(checks.get("animation"))
                    )
                    if compare_sizes:
                        report["renders"] |= render_svg(playwright, artifact_dir, entry, out_dir, compare_sizes[0])[
                            "renders"
                        ]
                else:
                    raise ValueError(f"Unknown artifact kind {kind!r}")
    except Exception as exc:  # A crashing renderer is a runtime failure of the artifact, not of the grader.
        report["render_exception"] = f"{type(exc).__name__}: {str(exc)[:800]}"

    stats: Dict[str, Any] = {}
    for name, path in report.get("renders", {}).items():
        if name.endswith("overview"):
            continue
        stats[name] = image_stats(load_rgb(path))
    report["render_stats"] = stats
    primary = [n for n in ("desktop", "svg", "slide_01", "compare") if n in stats] or list(stats)[:1]
    report["blank"] = (not primary) or all(is_blank(stats[n]) for n in primary)
    runtime = report.get("runtime") or {}
    report["load_ok"] = (
        "render_exception" not in report
        and not runtime.get("page_errors")
        and report.get("http_status", 200) in (200, None)
        and bool(stats)
    )
    return report


def compare_render_name(index: int) -> str:
    return "compare" if index == 0 else f"compare_{index + 1}"


def compare_sizes_for(task: Dict[str, Any], references: List[Path]) -> Optional[List[Tuple[int, int]]]:
    """Viewports used to render a candidate for comparison; references are rendered the same way.

    HTML is captured full-page, by default at the reference width and the task viewport height.
    With `reference_viewports` (responsive replication) it is captured once per viewport and
    compared with the matching reference image. SVG is rendered at exactly the reference size.
    """
    kind = task["artifact"]["kind"]
    if kind not in ("html", "svg") or not references:
        return None
    if kind == "html" and task.get("reference_viewports"):
        return [(int(vp["width"]), int(vp["height"])) for vp in task["reference_viewports"]]
    with Image.open(references[0]) as ref:
        if kind == "svg":
            return [(ref.width, ref.height)]
        return [(ref.width, (task.get("viewport") or DEFAULT_VIEWPORT)["height"])]


def measure(task: Dict[str, Any], artifact_dir: Path, reference_dir: Optional[Path], out_dir: Path) -> Dict[str, Any]:
    """Grader-side deterministic measurement: render, runtime checks and replication similarity."""
    replication = task.get("mode") == "replication"
    references = [reference_dir / r for r in (task.get("reference_images") or [])] if reference_dir else []
    compare_sizes = compare_sizes_for(task, references) if replication else None
    frame_times = None
    if replication and task["artifact"]["kind"] == "video":
        frame_times = [float(t) for t in task.get("reference_frame_times") or []] or None

    report = render_task_artifact(task, artifact_dir, out_dir, compare_sizes=compare_sizes, frame_times=frame_times)
    if not replication:
        return report

    similarity: Dict[str, Any] = {"pairs": []}
    renders = report.get("renders", {})
    kind = task["artifact"]["kind"]
    if kind == "slides":
        candidates = [renders.get(f"slide_{i:02d}") for i in range(1, len(references) + 1)]
    elif kind == "video":
        frames = [k for k in renders if k.startswith("frame_") and k != "frames_overview"]
        candidates = [renders[k] for k in sorted(frames)][: len(references)]
        candidates += [None] * (len(references) - len(candidates))
    elif kind == "html" and task.get("reference_viewports"):
        candidates = [renders.get(compare_render_name(i)) for i in range(len(references))]
    else:
        candidates = [renders.get("compare") or renders.get("desktop") or renders.get("svg")]
    scores: List[float] = []
    for ref_path, cand in zip(references, candidates):
        if not cand or not Path(cand).exists():
            similarity["pairs"].append({"reference": ref_path.name, "candidate": None, "normalized": 0.0})
            scores.append(0.0)
            continue
        metrics = visual_similarity(load_rgb(cand), load_rgb(ref_path))
        similarity["pairs"].append({"reference": ref_path.name, "candidate": Path(cand).name, **metrics})
        scores.append(metrics["normalized"])
    similarity["normalized"] = round(float(np.mean(scores)) if scores else 0.0, 4)
    similarity["combined_mean"] = round(
        float(np.mean([p.get("combined", 0.0) for p in similarity["pairs"]])) if similarity["pairs"] else 0.0, 4
    )
    report["similarity"] = similarity
    report["reference_copy"] = detect_reference_copy(artifact_dir, references)
    return report


def interact(
    artifact_dir: Path, entry: str, actions: List[Dict[str, Any]], out_dir: Path, viewport: Dict[str, int]
) -> Dict[str, Any]:
    """Run actions against an HTML artifact. Screenshots go to out_dir; returns a log of what happened."""
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    log: List[Dict[str, Any]] = []
    with StaticServer(artifact_dir) as server, sync_playwright() as playwright:
        browser = _launch(playwright)
        try:
            page = browser.new_page(viewport=viewport)
            recorder = PageRecorder(page, f"http://127.0.0.1:{server.port}")
            page.goto(server.url(entry), wait_until="load", timeout=30000)
            _settle(page, 800)
            shot_index = 0
            for step, action in enumerate(actions):
                kind = action.get("do")
                entry_log: Dict[str, Any] = {"step": step, "do": kind}
                try:
                    if kind == "wait":
                        page.wait_for_timeout(int(action.get("ms", 500)))
                    elif kind == "press":
                        for _ in range(int(action.get("times", 1))):
                            page.keyboard.press(action["key"])
                            page.wait_for_timeout(int(action.get("delay_ms", 60)))
                    elif kind == "keydown":
                        page.keyboard.down(action["key"])
                    elif kind == "keyup":
                        page.keyboard.up(action["key"])
                    elif kind in ("click", "dblclick"):
                        button = action.get("button", "left")
                        count = 2 if kind == "dblclick" else 1
                        if "selector" in action:
                            page.click(action["selector"], button=button, click_count=count, timeout=5000)
                        else:
                            page.mouse.click(float(action["x"]), float(action["y"]), button=button, click_count=count)
                    elif kind == "move":
                        page.mouse.move(float(action["x"]), float(action["y"]), steps=int(action.get("steps", 5)))
                    elif kind == "drag":
                        page.mouse.move(float(action["from"][0]), float(action["from"][1]))
                        page.mouse.down()
                        page.mouse.move(
                            float(action["to"][0]), float(action["to"][1]), steps=int(action.get("steps", 12))
                        )
                        page.mouse.up()
                    elif kind == "hover":
                        if "selector" in action:
                            page.hover(action["selector"], timeout=5000)
                        else:
                            page.mouse.move(float(action["x"]), float(action["y"]))
                    elif kind == "type":
                        if "selector" in action:
                            page.fill(action["selector"], action["text"], timeout=5000)
                        else:
                            page.keyboard.type(action["text"], delay=20)
                    elif kind == "select":
                        entry_log["result"] = page.select_option(
                            action["selector"], str(action["value"]), timeout=5000
                        )
                    elif kind == "reload":
                        page.reload(wait_until="load")
                        _settle(page, 800)
                    elif kind == "media":
                        page.emulate_media(media=action.get("media", "print"))
                    elif kind == "scroll":
                        page.mouse.wheel(0, float(action.get("dy", 600)))
                    elif kind == "eval":
                        entry_log["result"] = json.loads(json.dumps(page.evaluate(action["js"]), default=str))
                    elif kind == "screenshot":
                        shot_index += 1
                        name = re.sub(r"[^A-Za-z0-9_-]", "_", str(action.get("name") or f"shot_{shot_index:02d}"))
                        path = out_dir / f"{name}.png"
                        page.screenshot(path=str(path), full_page=bool(action.get("full_page", False)))
                        entry_log["path"] = str(path)
                    else:
                        entry_log["error"] = f"unknown action {kind!r}"
                except Exception as exc:
                    entry_log["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                log.append(entry_log)
            final = out_dir / "final.png"
            page.screenshot(path=str(final))
        finally:
            browser.close()
    return {"steps": log, "final_screenshot": str(final), **recorder.as_dict()}


SVG_REFERENCE_MAX_SIDE = 1000


def render_reference(task: Dict[str, Any], golden_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """Render a replication task's hidden golden artifact into its reference image(s).

    Uses the same renderers and sizes that `measure` later uses for candidates, then scores the
    golden against its own references as a determinism check (should be ~1.0).
    """
    import shutil

    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    kind = task["artifact"]["kind"]
    golden_entry = (task.get("golden") or {}).get("entry") or task["artifact"]["entry"]
    golden_task = {**task, "artifact": {**task["artifact"], "entry": golden_entry}}
    refs = list(task["reference_images"])
    scratch = out_dir / "_golden_renders"
    info: Dict[str, Any] = {"references": {}}
    if kind == "video":
        script = golden_dir / (task.get("golden") or {}).get("script", "make_video.py")
        result = subprocess.run(
            [sys.executable, script.name],
            cwd=golden_dir,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=900,
        )
        if result.returncode != 0:
            raise RuntimeError(f"golden video script failed: {result.stderr[-2000:]}")
        times = [float(t) for t in task["reference_frame_times"]]
        if len(times) != len(refs):
            raise ValueError("reference_frame_times and reference_images must have the same length")
        for t, name in zip(times, refs):
            if not extract_frame(golden_dir / golden_entry, t, out_dir / name):
                raise RuntimeError(f"could not extract the golden frame at {t}s")
        info["video"] = ffprobe(golden_dir / golden_entry)
    else:
        with sync_playwright() as playwright:
            if kind == "html":
                viewport = task.get("viewport") or DEFAULT_VIEWPORT
                sizes = [(int(vp["width"]), int(vp["height"])) for vp in task.get("reference_viewports") or []]
                sizes = sizes or [(viewport["width"], viewport["height"])]
                if len(sizes) != len(refs):
                    raise ValueError("reference_viewports and reference_images must have the same length")
                report = render_html(
                    playwright,
                    golden_dir,
                    golden_entry,
                    scratch,
                    viewport,
                    mobile=False,
                    animation=False,
                    compare_sizes=sizes,
                )
                for index, name in enumerate(refs):
                    shutil.copy(report["renders"][compare_render_name(index)], out_dir / name)
            elif kind == "svg":
                size = _svg_render_size(svg_facts(golden_dir / golden_entry), SVG_REFERENCE_MAX_SIDE)
                report = render_svg(playwright, golden_dir, golden_entry, scratch, compare_size=size)
                shutil.copy(report["renders"]["compare"], out_dir / refs[0])
            elif kind == "slides":
                report = render_slides(playwright, golden_dir, golden_entry, scratch)
                if report.get("slide_count", 0) != len(refs):
                    raise ValueError(f"golden deck has {report.get('slide_count')} slides but {len(refs)} references")
                for index, name in enumerate(refs, start=1):
                    shutil.copy(report["renders"][f"slide_{index:02d}"], out_dir / name)
            else:
                raise ValueError(f"Unknown artifact kind {kind!r}")
            errors = (report.get("runtime") or {}).get("page_errors") or []
            if errors:
                raise RuntimeError(f"golden artifact has page errors: {errors[:3]}")
    for name in refs:
        info["references"][name] = image_stats(load_rgb(out_dir / name))
    check = measure(golden_task, golden_dir, out_dir, scratch / "self_check")
    info["self_similarity"] = (check.get("similarity") or {}).get("normalized")
    info["self_check_blank"] = check.get("blank")
    shutil.rmtree(scratch, ignore_errors=True)
    return info


def _load_task(path: Optional[str], kind: Optional[str], entry: Optional[str], output_dir: str) -> Dict[str, Any]:
    task: Dict[str, Any] = json.loads(Path(path).read_text()) if path else {}
    if kind or entry:
        artifact = dict(task.get("artifact") or {})
        if kind:
            artifact["kind"] = kind
        if entry:
            artifact["entry"] = entry
        task["artifact"] = artifact
    task.setdefault("artifact", {}).setdefault("output_dir", output_dir)
    if "kind" not in task["artifact"] or "entry" not in task["artifact"]:
        raise SystemExit("Pass --task (task.json) or both --kind and --entry")
    return task


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preview", help="Render the artifact and print a runtime/layout report")
    p.add_argument("--task", default="/workspace/task/task.json")
    p.add_argument("--artifact-dir", default="/workspace/output")
    p.add_argument("--kind", choices=["html", "slides", "svg", "video"])
    p.add_argument("--entry")
    p.add_argument("--out", default="/workspace/previews")
    p.add_argument("--compare", nargs="*", default=None, help="Reference image(s) to score the render against")

    p = sub.add_parser("interact", help="Drive an HTML artifact with actions and capture screenshots")
    p.add_argument("--artifact-dir", default="/workspace/output")
    p.add_argument("--entry", default="index.html")
    p.add_argument("--actions", required=True, help="JSON list, or @path/to/actions.json")
    p.add_argument("--out", default="/workspace/previews/interact")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)

    p = sub.add_parser("compare", help="Visual similarity between a candidate and a reference image")
    p.add_argument("candidate")
    p.add_argument("reference")

    p = sub.add_parser("measure", help="Grader measurement pass")
    p.add_argument("--task", required=True)
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--reference-dir")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--report", required=True)

    p = sub.add_parser("reference", help="Render a replication task's golden artifact into its references")
    p.add_argument("--task", required=True)
    p.add_argument("--golden-dir", required=True)
    p.add_argument("--out-dir", required=True)

    args = parser.parse_args(argv)
    if args.command == "preview":
        task = _load_task(
            args.task if Path(args.task).exists() else None, args.kind, args.entry, str(args.artifact_dir)
        )
        out = Path(args.out)
        if args.compare:
            # Score exactly the way the grader does (same renders, frames and reference-copy check).
            refs = [Path(r) for r in args.compare]
            if len({r.parent for r in refs}) != 1:
                raise SystemExit("--compare reference images must be in one folder")
            compare_task = {**task, "mode": "replication", "reference_images": [r.name for r in refs]}
            report = measure(compare_task, Path(args.artifact_dir), refs[0].parent, out)
        else:
            report = render_task_artifact(task, Path(args.artifact_dir), out)
        print(json.dumps(report, indent=1))
        print(f"\nView the PNGs under {out} with your file-reading tool to check the result visually.")
    elif args.command == "interact":
        actions = args.actions
        if actions.startswith("@"):
            actions = Path(actions[1:]).read_text()
        result = interact(
            Path(args.artifact_dir),
            args.entry,
            json.loads(actions),
            Path(args.out),
            {"width": args.width, "height": args.height},
        )
        print(json.dumps(result, indent=1))
    elif args.command == "compare":
        print(json.dumps(visual_similarity(load_rgb(args.candidate), load_rgb(args.reference)), indent=1))
    elif args.command == "measure":
        task = json.loads(Path(args.task).read_text())
        report = measure(
            task,
            Path(args.artifact_dir),
            Path(args.reference_dir) if args.reference_dir else None,
            Path(args.out_dir),
        )
        Path(args.report).write_text(json.dumps(report, indent=1))
        print(json.dumps({k: report.get(k) for k in ("artifact_found", "load_ok", "blank", "similarity")}, indent=1))
    elif args.command == "reference":
        task = json.loads(Path(args.task).read_text())
        print(json.dumps(render_reference(task, Path(args.golden_dir), Path(args.out_dir)), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
