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
"""Pure-function tests for the in-sandbox toolkit (rendering itself needs Playwright and runs in sandboxes)."""

import base64
import io
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from resources_servers.visual_agent.sandbox_tools import vtools  # noqa: E402


def design(width: int = 320, height: int = 200) -> "np.ndarray":
    """A synthetic 'page': light background, dark header bar, two colored cards and some text-like stripes."""
    img = np.full((height, width, 3), 245.0, dtype=np.float32)
    img[:30] = (31, 42, 68)
    img[50:150, 20:150] = (42, 122, 226)
    img[50:150, 170:300] = (226, 74, 42)
    for y in range(165, 195, 8):
        img[y : y + 3, 20:260] = (60, 60, 60)
    return img


def png_bytes(rgb: "np.ndarray", fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format=fmt)
    return buf.getvalue()


class TestSimilarity:
    def test_identical_images_score_one(self) -> None:
        ref = design()
        metrics = vtools.visual_similarity(ref, ref)
        assert metrics["combined"] == pytest.approx(1.0)
        assert metrics["normalized"] == pytest.approx(1.0)

    def test_flat_background_scores_near_zero(self) -> None:
        ref = design()
        flat = np.full_like(ref, 245.0)
        assert vtools.visual_similarity(flat, ref)["normalized"] < 0.1

    def test_ordering_close_variant_beats_distant_variant(self) -> None:
        ref = design()
        close = ref.copy()
        close[50:150, 170:300] = (200, 80, 60)  # one card slightly recolored
        distant = ref[:, ::-1].copy()  # mirrored layout
        s_close = vtools.visual_similarity(close, ref)["normalized"]
        s_distant = vtools.visual_similarity(distant, ref)["normalized"]
        assert 1.0 > s_close > s_distant

    def test_align_to_reference_scales_width_and_pads_height(self) -> None:
        ref = design(320, 200)
        cand = design(640, 300)  # 2x wide and relatively shorter
        aligned = vtools.align_to_reference(cand, ref)
        assert aligned.shape == ref.shape
        assert (aligned[-10:] == 255.0).all()

    def test_metrics_stay_in_unit_interval_for_near_identical_flat_images(self) -> None:
        # With float32 cumulative sums, E[x^2]-E[x]^2 lost precision on large flat areas: SSIM came
        # out as 0.78 here and above 1 on near-perfect video replications.
        ref = np.full((1080, 1080, 3), 241.0, dtype=np.float32)
        ref[200:900, 200:900] = (99, 102, 241)
        cand = ref.copy()
        cand[540:542] += 1.0
        metrics = vtools.visual_similarity(cand, ref)
        for key in ("ssim", "pixel", "edge", "color_hist", "combined", "normalized"):
            assert 0.0 <= metrics[key] <= 1.0, (key, metrics[key])
        assert metrics["ssim"] > 0.99 and metrics["normalized"] > 0.99

    def test_ssim_bounds(self) -> None:
        a = design().mean(axis=2)
        assert vtools.ssim(a, a) == pytest.approx(1.0)
        assert vtools.ssim(a, 255.0 - a) < 0.5

    def test_blank_detection(self) -> None:
        assert vtools.is_blank(vtools.image_stats(np.full((50, 50, 3), 10.0)))
        assert not vtools.is_blank(vtools.image_stats(design()))


class TestReferenceCopy:
    def test_detects_byte_copy_base64_and_reencoded_copy(self, tmp_path: Path) -> None:
        ref_dir = tmp_path / "ref"
        art = tmp_path / "artifact"
        ref_dir.mkdir()
        art.mkdir()
        ref_png = png_bytes(design())
        (ref_dir / "reference.png").write_bytes(ref_png)

        (art / "index.html").write_text("<html><body><h1>Rebuilt page</h1></body></html>")
        assert not vtools.detect_reference_copy(art, [ref_dir / "reference.png"])["detected"]

        (art / "copy.png").write_bytes(ref_png)
        (art / "inline.html").write_text(f'<img src="data:image/png;base64,{base64.b64encode(ref_png).decode()}">')
        # A screenshot of the target, upscaled and re-encoded, is still caught.
        upscaled = np.asarray(Image.fromarray(design().astype(np.uint8)).resize((640, 400)), dtype=np.float32)
        (art / "shot.jpg").write_bytes(png_bytes(upscaled, fmt="JPEG"))
        findings = vtools.detect_reference_copy(art, [ref_dir / "reference.png"])["findings"]
        joined = "\n".join(findings)
        assert "copy.png is a byte-identical copy" in joined
        assert "inline.html contains the base64" in joined
        assert "shot.jpg is a raster copy" in joined

    def test_flags_loading_from_task_dir(self, tmp_path: Path) -> None:
        (tmp_path / "ref.png").write_bytes(png_bytes(design()))
        art = tmp_path / "artifact"
        art.mkdir()
        (art / "index.html").write_text('<img src="/workspace/task/reference.png">')
        result = vtools.detect_reference_copy(art, [tmp_path / "ref.png"])
        assert result["detected"] and "task directory" in result["findings"][0]


class TestSvgFacts:
    def test_facts(self, tmp_path: Path) -> None:
        svg = tmp_path / "logo.svg"
        svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 150">'
            '<g id="mark"><circle cx="10" cy="10" r="5"/></g><g id="wordmark"><text>Hi</text></g>'
            '<image href="data:image/png;base64,AAAA"/></svg>'
        )
        facts = vtools.svg_facts(svg)
        assert facts["well_formed"] and facts["viewBox"] == "0 0 300 150"
        assert facts["top_level_groups"] == ["mark", "wordmark"]
        assert facts["embedded_raster_images"] == 1
        assert vtools._svg_render_size(facts, 600) == (600, 300)

    def test_malformed(self, tmp_path: Path) -> None:
        svg = tmp_path / "bad.svg"
        svg.write_text("<svg><g></svg>")
        assert vtools.svg_facts(svg)["well_formed"] is False


def test_contact_sheet(tmp_path: Path) -> None:
    paths = []
    for i in range(4):
        p = tmp_path / f"f{i}.png"
        p.write_bytes(png_bytes(design()))
        paths.append(p)
    out = vtools.contact_sheet(paths, tmp_path / "sheet.png", columns=2, thumb_width=100)
    with Image.open(out) as sheet:
        assert sheet.size == (2 * 108 + 8, 2 * (62 + 8) + 8)  # 320x200 thumbnails at 100px wide are 62px tall
