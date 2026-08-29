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
"""Per-provider judge transport: overflow planning, eligibility, seeded replay."""

import base64
import random

import fitz
import pytest

from resources_servers.gdpval.comparison import (
    Judge,
    apply_native_pdf_overflow,
    filter_media_eligible_judges,
    plan_native_pdf_overflow,
    preview_trial_judges,
)
from resources_servers.gdpval.judge_panel import sample_judge
from resources_servers.gdpval.media_conversion import pdf_page_count
from resources_servers.gdpval.multistage_orchestrator import (
    MultiStageRunConfig,
    compute_fingerprint,
    parse_multistage_config,
)
from resources_servers.gdpval.transport_assignment import PairCost, _solve_capacity_assignment


PDF_PREFIX = "data:application/pdf;base64,"


def _pdf_bytes(pages: int) -> bytes:
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=200, height=200)
        page.insert_text((20, 40), f"page {index}")
    payload = document.tobytes()
    document.close()
    return payload


def _pdf_block(payload: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": PDF_PREFIX + base64.b64encode(payload).decode()}}


def _judge(name: str, **kwargs) -> Judge:
    return Judge(name=name, client=None, model="judge-model", **kwargs)


class TestPlanNativePdfOverflow:
    def test_forces_full_rasterization_above_per_document_byte_cap(self) -> None:
        small = _pdf_bytes(1)
        big = _pdf_bytes(2)
        sections = {"refs": [_pdf_block(small)], "submission_a": [_pdf_block(big)]}

        plan = plan_native_pdf_overflow(
            sections,
            native_page_cap=100,
            native_pdf_bytes_per_document=len(big) - 1,
            image_cap=50,
        )

        assert plan["eligible"] is True
        (selected,) = plan["selected"]
        assert selected["section"] == "submission_a"
        assert selected["reason"] == "native_pdf_bytes_per_document"
        assert selected["raster_page_count"] == 2
        assert selected["native_page_count"] == 0
        # Every source page stays represented, natively or as images.
        assert plan["native_pages_after"] + plan["raster_pages"] == plan["total_pdf_pages"] == 3

    def test_prefix_only_rasterization_for_page_cap_overflow(self) -> None:
        short = _pdf_bytes(3)
        long = _pdf_bytes(5)
        sections = {"refs": [_pdf_block(short), _pdf_block(long)]}

        plan = plan_native_pdf_overflow(
            sections,
            native_page_cap=6,
            native_pdf_bytes_per_document=10**9,
            image_cap=50,
        )

        # Two pages over the cap: the smallest document gives up exactly its
        # two-page prefix; its final page and the whole long document stay native.
        assert plan["eligible"] is True
        (selected,) = plan["selected"]
        assert selected["pages"] == 3
        assert selected["reason"] == "native_pdf_page_cap"
        assert selected["raster_page_start"] == 0
        assert selected["raster_page_count"] == 2
        assert selected["native_page_count"] == 1
        assert plan["native_pages_after"] == 6
        assert plan["raster_pages"] == 2

    def test_ineligible_when_raster_pages_exceed_image_cap(self) -> None:
        big = _pdf_bytes(3)
        plan = plan_native_pdf_overflow(
            {"refs": [_pdf_block(big)]},
            native_page_cap=100,
            native_pdf_bytes_per_document=len(big) - 1,
            image_cap=2,
        )
        assert plan["eligible"] is False


class TestApplyNativePdfOverflow:
    def test_every_page_retained_across_raster_and_native_suffix(self) -> None:
        short = _pdf_bytes(3)
        long = _pdf_bytes(5)
        sections = {"refs": [_pdf_block(short), _pdf_block(long)]}
        plan = plan_native_pdf_overflow(
            sections,
            native_page_cap=6,
            native_pdf_bytes_per_document=10**9,
            image_cap=50,
        )

        converted = apply_native_pdf_overflow(sections, plan, render_dpi=36, max_pages=100, include_text=False)

        blocks = converted["refs"]
        urls = [str((block.get("image_url") or {}).get("url", "")) for block in blocks]
        rendered_images = sum(1 for url in urls if url.startswith("data:image/"))
        native_pages = sum(
            pdf_page_count(base64.b64decode(url[len(PDF_PREFIX) :])) for url in urls if url.startswith(PDF_PREFIX)
        )
        assert rendered_images == 2
        assert native_pages == 6

    def test_rejects_plan_hash_drift(self) -> None:
        payload = _pdf_bytes(2)
        sections = {"refs": [_pdf_block(payload)]}
        plan = plan_native_pdf_overflow(
            sections,
            native_page_cap=1,
            native_pdf_bytes_per_document=10**9,
            image_cap=50,
        )
        drifted = {"refs": [_pdf_block(_pdf_bytes(2))]}
        with pytest.raises(ValueError, match="hash drift"):
            apply_native_pdf_overflow(drifted, plan, render_dpi=36, max_pages=100, include_text=False)


class TestFilterMediaEligibleJudges:
    NATIVE_STATS = {"pages": 120, "documents": 6, "bytes": 30_000_000}

    def test_exclusion_reasons_are_recorded_per_cap(self) -> None:
        judges = [
            _judge("claude", media_mode="native_pdf", max_native_pdf_bytes=24 * 1024 * 1024),
            _judge("gpt", media_mode="images_and_text"),
            _judge("gemini", media_mode="native_pdf_overflow_images"),
            _judge("roomy", media_mode="native_pdf"),
        ]

        eligible, exclusions = filter_media_eligible_judges(
            judges,
            native_stats=self.NATIVE_STATS,
            estimated_images=500,
            image_cap=450,
            overflow_plan=None,
        )

        assert [judge.name for judge in eligible] == ["roomy"]
        reasons = {exclusion["judges"][0]: exclusion["reason"] for exclusion in exclusions}
        assert reasons == {
            "claude": "native_pdf_cap",
            "gpt": "request_image_cap_preflight",
            "gemini": "native_pdf_overflow_unavailable",
        }

    def test_overflow_judge_eligible_only_with_eligible_plan(self) -> None:
        judge = _judge("gemini", media_mode="native_pdf_overflow_images")
        eligible, _ = filter_media_eligible_judges(
            [judge],
            native_stats=self.NATIVE_STATS,
            estimated_images=0,
            image_cap=450,
            overflow_plan={"eligible": True},
        )
        assert [j.name for j in eligible] == ["gemini"]
        _, exclusions = filter_media_eligible_judges(
            [judge],
            native_stats=self.NATIVE_STATS,
            estimated_images=0,
            image_cap=450,
            overflow_plan={"eligible": False},
        )
        assert exclusions[0]["reason"] == "native_pdf_overflow_unavailable"


class TestPreviewTrialJudges:
    def test_replays_seeded_schedule_without_consuming_rng(self) -> None:
        judges = [_judge("a"), _judge("b"), _judge("c")]
        rng = random.Random(7)
        state = rng.getstate()

        first = [judge.name for judge in preview_trial_judges(judges, 5, rng)]
        second = [judge.name for judge in preview_trial_judges(judges, 5, rng)]

        assert first == second
        assert rng.getstate() == state
        # The preview is exactly what dispatch will draw from the untouched RNG.
        assert [sample_judge(judges, rng).name for _ in range(5)] == first

    def test_exclusion_then_replay_is_deterministic(self) -> None:
        judges = [_judge("a"), _judge("b"), _judge("c")]
        rng = random.Random(11)
        survivors = [judge for judge in judges if judge.name != "b"]
        assert [j.name for j in preview_trial_judges(survivors, 4, rng)] == [
            j.name for j in preview_trial_judges(survivors, 4, rng)
        ]


class TestSolveCapacityAssignment:
    def test_minimum_change_reassignment_respects_capacities(self) -> None:
        tasks = ["t0", "t1", "t2"]
        references = ["ra", "rb"]
        original = {"t0": "ra", "t1": "ra", "t2": "rb"}

        def cost(compatible: bool, wire: int = 1) -> PairCost:
            return PairCost(compatible=compatible, wire_bytes=wire, raw_bytes=wire, max_file_bytes=wire, reasons=())

        # t1/ra is incompatible; the only capacity-preserving repair swaps t1 and t2.
        costs = {
            ("t0", "ra"): cost(True),
            ("t0", "rb"): cost(True),
            ("t1", "ra"): cost(False),
            ("t1", "rb"): cost(True),
            ("t2", "ra"): cost(True),
            ("t2", "rb"): cost(True),
        }

        result = _solve_capacity_assignment(tasks, references, original, costs)

        assert result == {"t0": "ra", "t1": "rb", "t2": "ra"}
        assert sorted(result.values()) == sorted(original.values())


class TestFingerprintTransportRepair:
    REF_ELOS = {"ref_a": 1000.0}
    DIST = {"grp": {"task_ids": ["t0"], "percentage": 100.0}}

    def _cfg(self) -> MultiStageRunConfig:
        return MultiStageRunConfig(
            enabled=True,
            stages=parse_multistage_config({"enabled": True, "stages": [{"num_tasks": 1}]}).stages,
            seed=0,
        )

    def test_repair_settings_invalidate_fingerprint(self) -> None:
        cfg = self._cfg()
        base = compute_fingerprint(cfg, self.REF_ELOS, self.DIST, resolved_global_config={"multistage": {}})
        repair_a = compute_fingerprint(
            cfg,
            self.REF_ELOS,
            self.DIST,
            resolved_global_config={"multistage": {"transport_assignment_repair": {"max_file_bytes": 1}}},
        )
        repair_b = compute_fingerprint(
            cfg,
            self.REF_ELOS,
            self.DIST,
            resolved_global_config={"multistage": {"transport_assignment_repair": {"max_file_bytes": 2}}},
        )
        assert repair_a != base
        assert repair_b != base
        assert repair_a != repair_b

    def test_absent_repair_config_preserves_legacy_fingerprint(self) -> None:
        cfg = self._cfg()
        without_block = compute_fingerprint(cfg, self.REF_ELOS, self.DIST, resolved_global_config={})
        empty_block = compute_fingerprint(cfg, self.REF_ELOS, self.DIST, resolved_global_config={"multistage": {}})
        assert without_block == empty_block


class TestOverflowRenderPageCap:
    def test_ineligible_when_forced_raster_exceeds_render_page_cap(self) -> None:
        big = _pdf_bytes(3)
        sections = {"refs": [_pdf_block(big)]}
        capped = plan_native_pdf_overflow(
            sections,
            native_page_cap=100,
            native_pdf_bytes_per_document=len(big) - 1,
            image_cap=50,
            render_page_cap=2,
        )
        assert capped["eligible"] is False

        roomy = plan_native_pdf_overflow(
            sections,
            native_page_cap=100,
            native_pdf_bytes_per_document=len(big) - 1,
            image_cap=50,
            render_page_cap=3,
        )
        assert roomy["eligible"] is True
        # An eligible plan renders successfully under the same page bound.
        converted = apply_native_pdf_overflow(sections, roomy, render_dpi=36, max_pages=3, include_text=False)
        rendered = [
            b for b in converted["refs"] if str((b.get("image_url") or {}).get("url", "")).startswith("data:image/")
        ]
        assert len(rendered) == 3


class TestOverflowAggregateCaps:
    def test_plan_reports_post_overflow_aggregates(self) -> None:
        short = _pdf_bytes(3)
        long = _pdf_bytes(5)
        sections = {"refs": [_pdf_block(short), _pdf_block(long)]}
        plan = plan_native_pdf_overflow(
            sections,
            native_page_cap=6,
            native_pdf_bytes_per_document=10**9,
            image_cap=50,
        )
        # Prefix split keeps a native suffix, so both documents remain native.
        assert plan["native_documents_after"] == 2
        assert plan["native_bytes_after_bound"] == len(short) + len(long)

        forced = plan_native_pdf_overflow(
            {"refs": [_pdf_block(short), _pdf_block(long)]},
            native_page_cap=100,
            native_pdf_bytes_per_document=len(long) - 1,
            image_cap=50,
        )
        # The long document is fully rasterized and leaves the native payload.
        assert forced["native_documents_after"] == 1
        assert forced["native_bytes_after_bound"] == len(short)

    def test_overflow_judge_excluded_when_aggregate_caps_exceeded_after_overflow(self) -> None:
        stats = {"pages": 8, "documents": 2, "bytes": 5_000}
        plan = {"eligible": True, "native_documents_after": 2, "native_bytes_after_bound": 4_000}

        capped_documents = _judge("gemini", media_mode="native_pdf_overflow_images", max_native_pdf_documents=1)
        _, exclusions = filter_media_eligible_judges(
            [capped_documents], native_stats=stats, estimated_images=0, image_cap=450, overflow_plan=plan
        )
        assert exclusions[0]["reason"] == "native_pdf_cap_after_overflow"

        capped_bytes = _judge("gemini", media_mode="native_pdf_overflow_images", max_native_pdf_bytes=3_000)
        _, exclusions = filter_media_eligible_judges(
            [capped_bytes], native_stats=stats, estimated_images=0, image_cap=450, overflow_plan=plan
        )
        assert exclusions[0]["reason"] == "native_pdf_cap_after_overflow"

        roomy = _judge(
            "gemini",
            media_mode="native_pdf_overflow_images",
            max_native_pdf_documents=2,
            max_native_pdf_bytes=4_000,
        )
        eligible, exclusions = filter_media_eligible_judges(
            [roomy], native_stats=stats, estimated_images=0, image_cap=450, overflow_plan=plan
        )
        assert [j.name for j in eligible] == ["gemini"] and not exclusions

    def test_legacy_plan_without_aggregates_stays_eligible(self) -> None:
        judge = _judge("gemini", media_mode="native_pdf_overflow_images", max_native_pdf_documents=1)
        eligible, exclusions = filter_media_eligible_judges(
            [judge],
            native_stats={"pages": 8, "documents": 2, "bytes": 5_000},
            estimated_images=0,
            image_cap=450,
            overflow_plan={"eligible": True},
        )
        assert [j.name for j in eligible] == ["gemini"] and not exclusions
