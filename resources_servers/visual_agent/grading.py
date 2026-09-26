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
"""Reward computation for visual agent tasks (MiMo-V2.6 §4.2.3).

Open-ended design is scored with pointwise binary rubrics: runtime correctness, instruction
adherence, layout integrity and basic aesthetics. Some items are decided by deterministic
checks and the rest by the agentic judge. A groupwise pass can then adjust rewards within a
rollout group (see `groupwise_adjust`).

High-fidelity replication is scored mainly by rule-based pixel similarity to the reference,
blended with the judge's fidelity rubric.
"""

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


RubricType = Literal["runtime", "instruction", "layout", "aesthetic", "fidelity", "interaction"]


class RubricItem(BaseModel):
    id: str
    type: RubricType
    criterion: str
    weight: float = Field(default=1.0, gt=0)


class RewardConfig(BaseModel):
    # Share of the replication reward from pixel similarity; the rest comes from the rubric.
    replication_similarity_weight: float = Field(default=0.6, ge=0, le=1)
    # Zero the reward when the judge reports reward hacking with non-empty evidence.
    judge_hack_flag_zeroes_reward: bool = True
    # Include the deterministic runtime/layout items in the rubric score.
    use_automatic_items: bool = True
    automatic_item_weight: float = Field(default=1.0, gt=0)


class ItemResult(BaseModel):
    id: str
    type: str
    weight: float
    passed: bool
    source: Literal["automatic", "judge", "missing"]
    evidence: str = ""


class ParsedVerdict(BaseModel):
    valid: bool
    items: Dict[str, bool] = Field(default_factory=dict)
    evidence: Dict[str, str] = Field(default_factory=dict)
    missing_items: List[str] = Field(default_factory=list)
    aesthetics: Optional[float] = None
    fidelity: Optional[float] = None
    hack_suspected: bool = False
    hack_evidence: str = ""
    summary: str = ""
    error: Optional[str] = None


class RewardBreakdown(BaseModel):
    reward: float
    gate_passed: bool
    gate_reasons: List[str] = Field(default_factory=list)
    rubric_score: Optional[float] = None
    similarity: Optional[float] = None
    hack_detected: bool = False
    hack_reasons: List[str] = Field(default_factory=list)
    items: List[ItemResult] = Field(default_factory=list)


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value >= 1
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "pass", "passed", "yes", "y", "1"):
            return True
        if lowered in ("false", "fail", "failed", "no", "n", "0"):
            return False
    return None


def _as_score(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return min(10.0, max(1.0, score))


def parse_verdict(raw: Any, rubric: List[RubricItem], *, min_coverage: float = 0.5) -> ParsedVerdict:
    """Parse the judge's verdict JSON (text or dict) against the task rubric.

    A verdict counts as valid when it covers at least `min_coverage` of the rubric weight.
    Uncovered items are treated as failed and listed in `missing_items`.
    """
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            return ParsedVerdict(valid=False, error=f"verdict is not JSON: {exc}")
    if not isinstance(raw, dict):
        return ParsedVerdict(valid=False, error="verdict is not a JSON object")

    entries = raw.get("rubric")
    if isinstance(entries, dict):
        entries = [{"id": k, **(v if isinstance(v, dict) else {"pass": v})} for k, v in entries.items()]
    if not isinstance(entries, list):
        return ParsedVerdict(valid=False, error="verdict has no rubric list")

    known = {item.id for item in rubric}
    items: Dict[str, bool] = {}
    evidence: Dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item_id = str(entry.get("id", "")).strip()
        passed = _as_bool(entry.get("pass", entry.get("passed")))
        if item_id in known and passed is not None:
            items[item_id] = passed
            evidence[item_id] = str(entry.get("evidence", ""))[:600]

    total_weight = sum(item.weight for item in rubric) or 1.0
    covered_weight = sum(item.weight for item in rubric if item.id in items)
    scores = raw.get("scores") if isinstance(raw.get("scores"), dict) else {}
    hacking = raw.get("reward_hacking") if isinstance(raw.get("reward_hacking"), dict) else {}
    coverage_ok = covered_weight / total_weight >= min_coverage
    return ParsedVerdict(
        valid=coverage_ok,
        items=items,
        evidence=evidence,
        missing_items=[item.id for item in rubric if item.id not in items],
        aesthetics=_as_score(scores.get("aesthetics")),
        fidelity=_as_score(scores.get("fidelity")),
        hack_suspected=bool(_as_bool(hacking.get("suspected"))),
        hack_evidence=str(hacking.get("evidence") or "")[:1000],
        summary=str(raw.get("summary") or "")[:2000],
        error=None if coverage_ok else f"verdict covers {covered_weight:.1f}/{total_weight:.1f} of the rubric weight",
    )


def runtime_gate(measurements: Dict[str, Any]) -> List[str]:
    """Reasons the artifact cannot be graded at all; empty when it rendered."""
    reasons: List[str] = []
    if not measurements.get("artifact_found"):
        reasons.append(measurements.get("error") or "artifact not found")
        return reasons
    if measurements.get("render_exception"):
        reasons.append(f"renderer failed: {measurements['render_exception']}")
    if measurements.get("kind") == "video" and not (measurements.get("video") or {}).get("ok"):
        reasons.append("video has no decodable video stream")
    if measurements.get("kind") == "svg" and not (measurements.get("svg") or {}).get("well_formed"):
        reasons.append("SVG is not well-formed XML")
    if not measurements.get("render_stats"):
        reasons.append("nothing was rendered")
    elif measurements.get("blank"):
        reasons.append("render is blank")
    return reasons


def automatic_items(task: Dict[str, Any], measurements: Dict[str, Any], weight: float = 1.0) -> List[ItemResult]:
    """Deterministic rubric items decided from the grader's measurements."""
    kind = measurements.get("kind")
    runtime = measurements.get("runtime") or {}
    checks = task.get("checks") or {}
    results: List[ItemResult] = []

    if kind in ("html", "slides"):
        errors = list(runtime.get("page_errors") or []) + list(runtime.get("console_errors") or [])
        errors += list((measurements.get("mobile_runtime") or {}).get("page_errors") or [])
        failed = [r for r in runtime.get("failed_requests") or [] if "favicon" not in r]
        external = list(runtime.get("external_requests") or [])
        problems = (
            errors[:3] + [f"failed request {r}" for r in failed[:3]] + [f"external request {r}" for r in external[:3]]
        )
        results.append(
            ItemResult(
                id="AUTO-runtime",
                type="runtime",
                weight=weight,
                passed=not problems,
                source="automatic",
                evidence="; ".join(problems) or "no JS errors, failed or external requests during load",
            )
        )
    elif kind == "svg":
        svg = measurements.get("svg") or {}
        problems = []
        if not measurements.get("svg_image_loaded"):
            problems.append("does not load as an image")
        if svg.get("embedded_raster_images"):
            problems.append(f"{svg['embedded_raster_images']} embedded raster <image> element(s)")
        if svg.get("has_script"):
            problems.append("contains <script>")
        results.append(
            ItemResult(
                id="AUTO-runtime",
                type="runtime",
                weight=weight,
                passed=not problems,
                source="automatic",
                evidence="; ".join(problems) or "pure vector SVG that loads as an image",
            )
        )
    elif kind == "video":
        video = measurements.get("video") or {}
        spec = task.get("video_spec") or {}
        problems = []
        duration = video.get("duration_s") or 0.0
        if spec.get("min_duration_s") and duration < spec["min_duration_s"] - 0.05:
            problems.append(f"duration {duration:.2f}s < {spec['min_duration_s']}s")
        if spec.get("max_duration_s") and duration > spec["max_duration_s"] + 0.05:
            problems.append(f"duration {duration:.2f}s > {spec['max_duration_s']}s")
        if spec.get("width") and (video.get("width"), video.get("height")) != (spec["width"], spec["height"]):
            problems.append(
                f"resolution {video.get('width')}x{video.get('height')} != {spec['width']}x{spec['height']}"
            )
        if spec.get("min_fps") and (video.get("fps") or 0) < spec["min_fps"] - 0.5:
            problems.append(f"fps {video.get('fps')} < {spec['min_fps']}")
        if not measurements.get("animated"):
            problems.append("frames barely change (static video)")
        results.append(
            ItemResult(
                id="AUTO-runtime",
                type="runtime",
                weight=weight,
                passed=not problems,
                source="automatic",
                evidence="; ".join(problems)
                or f"{duration:.2f}s {video.get('width')}x{video.get('height')} @ {video.get('fps')} fps",
            )
        )

    if kind == "html":
        layout = measurements.get("layout") or {}
        problems = []
        if layout.get("horizontal_overflow_px", 0) > 4:
            problems.append(f"desktop horizontal overflow {layout['horizontal_overflow_px']}px")
        if layout.get("broken_images"):
            problems.append(f"{layout['broken_images']} broken image(s)")
        if checks.get("mobile"):
            mobile = measurements.get("mobile_layout") or {}
            if mobile.get("horizontal_overflow_px", 0) > 4:
                problems.append(f"mobile horizontal overflow {mobile['horizontal_overflow_px']}px")
        results.append(
            ItemResult(
                id="AUTO-layout",
                type="layout",
                weight=weight,
                passed=not problems,
                source="automatic",
                evidence="; ".join(problems) or "no overflow or broken images",
            )
        )
        if checks.get("animation"):
            results.append(
                ItemResult(
                    id="AUTO-animation",
                    type="runtime",
                    weight=weight,
                    passed=bool(measurements.get("animated")),
                    source="automatic",
                    evidence=f"mean frame difference over 1.5s: {measurements.get('animation_mean_abs_diff')}",
                )
            )
    elif kind == "slides":
        overflow = measurements.get("overflowing_slides") or []
        count = measurements.get("slide_count") or 0
        spec = task.get("slides_spec") or {}
        problems = [f"slide {o['slide']} content overflows its 1280x720 frame" for o in overflow[:5]]
        if spec.get("min_slides") and count < spec["min_slides"]:
            problems.append(f"{count} slides < {spec['min_slides']}")
        if spec.get("max_slides") and count > spec["max_slides"]:
            problems.append(f"{count} slides > {spec['max_slides']}")
        sizes = measurements.get("slide_sizes") or []
        if any(abs(w - 1280) > 2 or abs(h - 720) > 2 for w, h in sizes):
            problems.append("some slides are not 1280x720")
        results.append(
            ItemResult(
                id="AUTO-layout",
                type="layout",
                weight=weight,
                passed=not problems,
                source="automatic",
                evidence="; ".join(problems) or f"{count} slides, all 1280x720 without overflow",
            )
        )
    elif kind == "svg" and checks.get("animation"):
        results.append(
            ItemResult(
                id="AUTO-animation",
                type="runtime",
                weight=weight,
                passed=bool(measurements.get("animated")),
                source="automatic",
                evidence=f"mean frame difference over 1.5s: {measurements.get('animation_mean_abs_diff')}",
            )
        )
    return results


def compute_reward(
    task: Dict[str, Any],
    rubric: List[RubricItem],
    measurements: Dict[str, Any],
    verdict: Optional[ParsedVerdict],
    config: RewardConfig,
) -> RewardBreakdown:
    """Combine the runtime gate, deterministic items, judge items and similarity into one reward."""
    gate_reasons = runtime_gate(measurements)
    if gate_reasons:
        return RewardBreakdown(reward=0.0, gate_passed=False, gate_reasons=gate_reasons)

    hack_reasons: List[str] = list((measurements.get("reference_copy") or {}).get("findings") or [])
    if verdict is not None and verdict.valid and verdict.hack_suspected and config.judge_hack_flag_zeroes_reward:
        if verdict.hack_evidence.strip():
            hack_reasons.append(f"judge: {verdict.hack_evidence.strip()}")

    items: List[ItemResult] = []
    if config.use_automatic_items:
        items.extend(automatic_items(task, measurements, config.automatic_item_weight))
    if verdict is not None and verdict.valid:
        for item in rubric:
            if item.id in verdict.items:
                items.append(
                    ItemResult(
                        id=item.id,
                        type=item.type,
                        weight=item.weight,
                        passed=verdict.items[item.id],
                        source="judge",
                        evidence=verdict.evidence.get(item.id, ""),
                    )
                )
            else:
                items.append(
                    ItemResult(id=item.id, type=item.type, weight=item.weight, passed=False, source="missing")
                )

    total = sum(i.weight for i in items)
    rubric_score = sum(i.weight for i in items if i.passed) / total if total else None

    similarity = None
    if task.get("mode") == "replication":
        similarity = float((measurements.get("similarity") or {}).get("normalized") or 0.0)
        w = config.replication_similarity_weight
        reward = similarity if rubric_score is None else w * similarity + (1 - w) * rubric_score
    else:
        reward = rubric_score if rubric_score is not None else 0.0

    if hack_reasons:
        reward = 0.0
    return RewardBreakdown(
        reward=round(float(min(1.0, max(0.0, reward))), 6),
        gate_passed=True,
        rubric_score=None if rubric_score is None else round(rubric_score, 6),
        similarity=similarity,
        hack_detected=bool(hack_reasons),
        hack_reasons=hack_reasons,
        items=items,
    )


TIER_VALUES = {"stronger": 1, "comparable": 0, "weaker": -1}


def parse_group_verdict(raw: Any, labels: List[str]) -> Optional[Dict[str, int]]:
    """Map candidate label -> tier (+1 stronger, 0 comparable, -1 weaker); None if unusable."""
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict) or not isinstance(raw.get("candidates"), list):
        return None
    tiers: Dict[str, int] = {}
    for entry in raw["candidates"]:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label", "")).strip()
        tier = TIER_VALUES.get(str(entry.get("tier", "")).strip().lower())
        if label in labels and tier is not None:
            tiers[label] = tier
    if set(tiers) != set(labels):
        return None
    # A group judged all-stronger or all-weaker carries no relative information.
    if len(set(tiers.values())) == 1:
        return {label: 0 for label in labels}
    return tiers


def groupwise_adjust(pointwise: Dict[int, float], tiers: Dict[int, int], bonus: float) -> Dict[int, float]:
    """Shift gated rewards by +/- bonus for clearly stronger/weaker candidates, clipped to [0, 1].

    Candidates that failed the runtime gate (reward 0, absent from `tiers`) keep their reward.
    """
    adjusted = dict(pointwise)
    for index, tier in tiers.items():
        adjusted[index] = round(min(1.0, max(0.0, pointwise[index] + bonus * tier)), 6)
    return adjusted
