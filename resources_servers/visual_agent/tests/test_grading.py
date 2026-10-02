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
import json
from typing import Any, Dict

import pytest

from resources_servers.visual_agent.grading import (
    RewardConfig,
    RubricItem,
    automatic_items,
    compute_reward,
    groupwise_adjust,
    parse_group_verdict,
    parse_verdict,
    runtime_gate,
)


RUBRIC = [
    RubricItem(id="R1", type="instruction", criterion="has a hero", weight=2),
    RubricItem(id="R2", type="interaction", criterion="faq toggles"),
    RubricItem(id="R3", type="aesthetic", criterion="looks polished"),
]


def html_measurements(**overrides: Any) -> Dict[str, Any]:
    m = {
        "kind": "html",
        "artifact_found": True,
        "render_stats": {"desktop": {"gray_std": 40}},
        "blank": False,
        "runtime": {"page_errors": [], "console_errors": [], "failed_requests": [], "external_requests": []},
        "layout": {"horizontal_overflow_px": 0, "broken_images": 0},
    }
    m.update(overrides)
    return m


def verdict(passes: Dict[str, bool], **extra: Any) -> str:
    return json.dumps(
        {
            "rubric": [{"id": k, "pass": v, "evidence": f"saw {k}"} for k, v in passes.items()],
            "scores": {"aesthetics": 7, "fidelity": None},
            "reward_hacking": {"suspected": False, "evidence": ""},
            "summary": "ok",
            **extra,
        }
    )


class TestParseVerdict:
    def test_valid_verdict(self) -> None:
        parsed = parse_verdict(verdict({"R1": True, "R2": False, "R3": True}), RUBRIC)
        assert parsed.valid
        assert parsed.items == {"R1": True, "R2": False, "R3": True}
        assert parsed.evidence["R1"] == "saw R1"
        assert parsed.aesthetics == 7.0 and parsed.fidelity is None

    def test_lenient_formats(self) -> None:
        raw = {"rubric": {"R1": {"pass": "yes"}, "R2": "fail", "R3": 1}, "scores": {"aesthetics": 42}}
        parsed = parse_verdict(raw, RUBRIC)
        assert parsed.items == {"R1": True, "R2": False, "R3": True}
        assert parsed.aesthetics == 10.0  # clipped to the 1-10 scale

    def test_unknown_ids_ignored_and_low_coverage_invalid(self) -> None:
        parsed = parse_verdict(verdict({"R3": True, "BOGUS": True}), RUBRIC)
        # R3 alone is 1/4 of the rubric weight, below the default 50% coverage.
        assert not parsed.valid
        assert parsed.missing_items == ["R1", "R2"]
        assert "covers" in parsed.error

    @pytest.mark.parametrize("raw", ["not json", "[1, 2]", json.dumps({"summary": "no rubric"})])
    def test_malformed(self, raw: str) -> None:
        assert not parse_verdict(raw, RUBRIC).valid


class TestRuntimeGate:
    def test_passes_for_rendered_artifact(self) -> None:
        assert runtime_gate(html_measurements()) == []

    @pytest.mark.parametrize(
        "measurements, reason",
        [
            ({"artifact_found": False, "error": "missing index.html"}, "missing index.html"),
            (html_measurements(blank=True), "blank"),
            (html_measurements(render_exception="TimeoutError: x"), "renderer failed"),
            (html_measurements(render_stats={}), "nothing was rendered"),
            ({"kind": "video", "artifact_found": True, "video": {"ok": False}, "render_stats": {}}, "video"),
            ({"kind": "svg", "artifact_found": True, "svg": {"well_formed": False}, "render_stats": {}}, "SVG"),
        ],
    )
    def test_failures(self, measurements: Dict[str, Any], reason: str) -> None:
        reasons = runtime_gate(measurements)
        assert reasons and any(reason in r for r in reasons)


class TestAutomaticItems:
    def test_html_errors_and_mobile_overflow_fail(self) -> None:
        m = html_measurements(
            runtime={"page_errors": ["TypeError: x is undefined"], "external_requests": ["https://cdn.x/y.js"]},
            mobile_layout={"horizontal_overflow_px": 120},
        )
        items = {i.id: i for i in automatic_items({"checks": {"mobile": True}}, m)}
        assert not items["AUTO-runtime"].passed and "TypeError" in items["AUTO-runtime"].evidence
        assert "external request" in items["AUTO-runtime"].evidence
        assert not items["AUTO-layout"].passed and "mobile horizontal overflow" in items["AUTO-layout"].evidence

    def test_favicon_404_is_ignored(self) -> None:
        m = html_measurements(runtime={"failed_requests": ["404 http://127.0.0.1:1/favicon.ico"]})
        items = {i.id: i for i in automatic_items({}, m)}
        assert items["AUTO-runtime"].passed

    def test_animation_check(self) -> None:
        items = {i.id: i for i in automatic_items({"checks": {"animation": True}}, html_measurements(animated=False))}
        assert not items["AUTO-animation"].passed

    def test_slides(self) -> None:
        m = {
            "kind": "slides",
            "runtime": {},
            "slide_count": 4,
            "overflowing_slides": [{"slide": 2}],
            "slide_sizes": [[1280, 720]] * 4,
        }
        items = {i.id: i for i in automatic_items({"slides_spec": {"min_slides": 6}}, m)}
        evidence = items["AUTO-layout"].evidence
        assert not items["AUTO-layout"].passed
        assert "slide 2" in evidence and "4 slides < 6" in evidence

    def test_video_spec(self) -> None:
        m = {"kind": "video", "video": {"duration_s": 3.0, "width": 640, "height": 360, "fps": 30}, "animated": True}
        spec = {"video_spec": {"min_duration_s": 5, "width": 1280, "height": 720, "min_fps": 24}}
        item = automatic_items(spec, m)[0]
        assert not item.passed
        assert "duration 3.00s < 5s" in item.evidence and "resolution 640x360" in item.evidence

    def test_svg_raster_fails(self) -> None:
        m = {"kind": "svg", "svg_image_loaded": True, "svg": {"embedded_raster_images": 1, "has_script": False}}
        assert not automatic_items({}, m)[0].passed


class TestComputeReward:
    def test_open_ended_weighted_score(self) -> None:
        parsed = parse_verdict(verdict({"R1": True, "R2": False, "R3": True}), RUBRIC)
        breakdown = compute_reward({"mode": "open_ended"}, RUBRIC, html_measurements(), parsed, RewardConfig())
        # AUTO-runtime(1) + AUTO-layout(1) + R1(2) + R3(1) pass; R2(1) fails -> 5/6.
        assert breakdown.gate_passed
        assert breakdown.reward == pytest.approx(5 / 6, abs=1e-6)
        assert [i.source for i in breakdown.items] == ["automatic", "automatic", "judge", "judge", "judge"]

    def test_gate_failure_is_zero(self) -> None:
        parsed = parse_verdict(verdict({"R1": True, "R2": True, "R3": True}), RUBRIC)
        breakdown = compute_reward(
            {"mode": "open_ended"}, RUBRIC, html_measurements(blank=True), parsed, RewardConfig()
        )
        assert breakdown.reward == 0.0 and not breakdown.gate_passed

    def test_replication_blend(self) -> None:
        m = html_measurements(similarity={"normalized": 0.5})
        parsed = parse_verdict(verdict({"R1": True, "R2": True, "R3": True}), RUBRIC)
        breakdown = compute_reward({"mode": "replication"}, RUBRIC, m, parsed, RewardConfig())
        assert breakdown.reward == pytest.approx(0.6 * 0.5 + 0.4 * 1.0)
        assert breakdown.similarity == 0.5

    def test_reference_copy_zeroes_reward(self) -> None:
        m = html_measurements(
            similarity={"normalized": 1.0}, reference_copy={"detected": True, "findings": ["ref.png is a copy"]}
        )
        parsed = parse_verdict(verdict({"R1": True, "R2": True, "R3": True}), RUBRIC)
        breakdown = compute_reward({"mode": "replication"}, RUBRIC, m, parsed, RewardConfig())
        assert breakdown.reward == 0.0 and breakdown.hack_detected
        assert breakdown.hack_reasons == ["ref.png is a copy"]

    def test_judge_hack_flag_needs_evidence(self) -> None:
        cfg = RewardConfig()
        no_evidence = parse_verdict(
            verdict({"R1": True, "R2": True, "R3": True}, reward_hacking={"suspected": True, "evidence": ""}), RUBRIC
        )
        assert compute_reward({"mode": "open_ended"}, RUBRIC, html_measurements(), no_evidence, cfg).reward == 1.0
        with_evidence = parse_verdict(
            verdict(
                {"R1": True, "R2": True, "R3": True},
                reward_hacking={"suspected": True, "evidence": "page is the target PNG in an img tag"},
            ),
            RUBRIC,
        )
        breakdown = compute_reward({"mode": "open_ended"}, RUBRIC, html_measurements(), with_evidence, cfg)
        assert breakdown.reward == 0.0 and breakdown.hack_reasons == ["judge: page is the target PNG in an img tag"]

    def test_without_verdict_only_automatic_items_count(self) -> None:
        breakdown = compute_reward({"mode": "open_ended"}, RUBRIC, html_measurements(), None, RewardConfig())
        assert breakdown.reward == 1.0 and {i.source for i in breakdown.items} == {"automatic"}
        replication = compute_reward(
            {"mode": "replication"},
            RUBRIC,
            html_measurements(similarity={"normalized": 0.3}),
            None,
            RewardConfig(use_automatic_items=False),
        )
        assert replication.reward == pytest.approx(0.3)


class TestGroupwise:
    def test_parse_group_verdict(self) -> None:
        raw = {"candidates": [{"label": "A", "tier": "stronger"}, {"label": "B", "tier": "Weaker"}]}
        assert parse_group_verdict(json.dumps(raw), ["A", "B"]) == {"A": 1, "B": -1}

    def test_parse_group_verdict_incomplete_or_uniform(self) -> None:
        assert parse_group_verdict({"candidates": [{"label": "A", "tier": "stronger"}]}, ["A", "B"]) is None
        assert parse_group_verdict("nope", ["A"]) is None
        uniform = {"candidates": [{"label": "A", "tier": "stronger"}, {"label": "B", "tier": "stronger"}]}
        assert parse_group_verdict(uniform, ["A", "B"]) == {"A": 0, "B": 0}

    def test_groupwise_adjust_clips_and_skips_ungated(self) -> None:
        adjusted = groupwise_adjust({0: 0.95, 1: 0.5, 2: 0.1, 3: 0.0}, {0: 1, 1: 0, 2: -1}, bonus=0.15)
        assert adjusted == {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0}
