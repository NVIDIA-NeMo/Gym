# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How /verify reports faults: judge failures, per-reference isolation, and what counts as the model result."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.verification import judge_failsafe
from resources_servers.aa_briefcase_lite.tests.conftest import (
    NO_SESSION,
    deliverables,
    pairwise_result,
    verify_body,
)


PASSED = ({"passed": True, "reasoning": "ok"}, "{}")


def _row(response) -> dict:
    return json.loads(response.body) if hasattr(response, "body") else response.model_dump()


async def _verify(server, tmp_path: Path, files: dict[str, str]) -> dict:
    return _row(await judge_failsafe(server.verify)(NO_SESSION, verify_body(deliverables(tmp_path, files))))


async def test_a_failed_binary_judge_call_masks_the_row_and_keeps_the_deliverable(
    make_server, tmp_path: Path, monkeypatch
) -> None:
    server = make_server(reward_mode="binary")
    monkeypatch.setattr(server, "_binary_call", AsyncMock(side_effect=ConnectionError("judge unreachable")))

    row = await _verify(server, tmp_path, {"report.md": "the report"})

    assert row["mask_sample"] is True
    assert row["failure_kind"] == "judge_failed"
    assert row["reward"] == 0.0
    assert (tmp_path / "submitted" / "report.md").read_text() == "the report"


async def test_one_failed_reference_does_not_fail_the_task(make_server, tmp_path: Path, monkeypatch) -> None:
    server = make_server(reward_mode="pairwise")

    def run_trials(**kwargs):
        if "ref_b report" in str(kwargs["submission_b"]):
            raise ConnectionError("judge unreachable")
        return pairwise_result(wins=2, losses=0)

    monkeypatch.setattr("resources_servers.aa_briefcase_lite.app.run_trials", run_trials)

    row = await _verify(server, tmp_path, {"report.md": "x"})

    assert not row.get("mask_sample")
    assert row["reward"] == 1.0
    assert row["pairwise_wins"] == 2
    skipped = [r for r in row["judge_response"]["pairwise_results"] if r.get("skipped_reason") == "judge_failed"]
    assert [r["reference_id"] for r in skipped] == ["ref_b"]


async def test_a_task_whose_every_reference_failed_is_a_judge_failure(
    make_server, tmp_path: Path, monkeypatch
) -> None:
    server = make_server(reward_mode="pairwise")

    def run_trials(**kwargs):
        raise ConnectionError("judge unreachable")

    monkeypatch.setattr("resources_servers.aa_briefcase_lite.app.run_trials", run_trials)

    row = await _verify(server, tmp_path, {"report.md": "x"})

    assert row["mask_sample"] is True
    assert row["failure_kind"] == "judge_failed"


async def test_binary_results_survive_a_failed_reference_in_all_mode(make_server, tmp_path: Path, monkeypatch) -> None:
    server = make_server(reward_mode="all")
    monkeypatch.setattr(server, "_binary_call", AsyncMock(return_value=PASSED))

    def run_trials(**kwargs):
        if "ref_b report" in str(kwargs["submission_b"]):
            raise ConnectionError("judge unreachable")
        return pairwise_result(wins=0, losses=2)

    monkeypatch.setattr("resources_servers.aa_briefcase_lite.app.run_trials", run_trials)

    row = await _verify(server, tmp_path, {"report.md": "x"})

    assert not row.get("mask_sample")
    assert row["rubric_passed"] == 2
    assert row["reward"] == pytest.approx(0.5)
