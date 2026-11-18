# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How /verify reports faults: judge failures, per-reference isolation, and what counts as the model result."""

import json
import logging
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.config import AggregateMetricsRequest
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


async def test_an_agent_that_ran_out_of_wall_time_is_a_scored_zero_not_a_masked_row(
    make_server, tmp_path: Path, monkeypatch
) -> None:
    server = make_server(reward_mode="all")
    judged = AsyncMock(side_effect=AssertionError("nothing was delivered, so nothing is judged"))
    monkeypatch.setattr(server, "_binary_call", judged)
    body = verify_body(deliverables(tmp_path, {"report.md": "half a report"}))
    body.response.metadata = {"stop_reason": "wall_time"}

    row = _row(await judge_failsafe(server.verify)(NO_SESSION, body))

    assert (row["reward"], row["mask_sample"]) == (0.0, False)
    assert (row["failure_kind"], row["failure_outcome_attribution"]) == ("agent_timeout", "model")
    judged.assert_not_called()


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


async def test_an_unusable_binary_verdict_masks_the_row_as_judge_failed(
    make_server, tmp_path: Path, monkeypatch
) -> None:
    server = make_server(reward_mode="binary")
    # The first check passes; the judge never gives a usable answer to the second.
    monkeypatch.setattr(server, "_binary_call", AsyncMock(side_effect=[PASSED, (None, "")]))

    row = await _verify(server, tmp_path, {"report.md": "the report"})

    assert row["mask_sample"] is True
    assert row["failure_kind"] == "judge_failed"
    assert "no usable verdict for check a2" in row["failure_reason"]
    assert row["reward"] == 0.0
    assert (tmp_path / "submitted" / "report.md").read_text() == "the report"


async def test_an_unusable_pairwise_answer_drops_its_reference_like_a_failed_call(
    make_server, tmp_path: Path, monkeypatch
) -> None:
    server = make_server(reward_mode="pairwise")

    def run_trials(**kwargs):
        invalid = 1 if "ref_b report" in str(kwargs["submission_b"]) else 0
        return pairwise_result(wins=2 - invalid, losses=0, invalid=invalid)

    monkeypatch.setattr("resources_servers.aa_briefcase_lite.app.run_trials", run_trials)

    row = await _verify(server, tmp_path, {"report.md": "x"})

    assert not row.get("mask_sample")
    assert row["pairwise_wins"] == 2
    skipped = [r for r in row["judge_response"]["pairwise_results"] if r.get("skipped_reason") == "judge_failed"]
    assert [r["reference_id"] for r in skipped] == ["ref_b"]


async def test_a_task_whose_every_reference_gave_unusable_answers_is_a_judge_failure(
    make_server, tmp_path: Path, monkeypatch
) -> None:
    server = make_server(reward_mode="pairwise")
    monkeypatch.setattr(
        "resources_servers.aa_briefcase_lite.app.run_trials",
        lambda **kwargs: pairwise_result(wins=1, losses=0, invalid=1),
    )

    row = await _verify(server, tmp_path, {"report.md": "x"})

    assert row["mask_sample"] is True
    assert row["failure_kind"] == "judge_failed"
    assert row["reward"] == 0.0


async def test_a_task_with_no_reference_to_judge_masks_the_row_as_a_verifier_error(
    make_server, tmp_path: Path, monkeypatch
) -> None:
    server = make_server(reward_mode="pairwise")
    for name in ("ref_a", "ref_b"):
        shutil.rmtree(tmp_path / name / "task_w1_t1")
    monkeypatch.setattr("resources_servers.aa_briefcase_lite.app.run_trials", pytest.fail)

    row = await _verify(server, tmp_path, {"report.md": "x"})

    assert row["mask_sample"] is True
    assert row["failure_kind"] == "verifier_error"
    assert "no pairwise reference could be judged" in row["failure_reason"]


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


async def test_a_conversion_failure_masks_the_row_as_a_verifier_error(
    make_server, tmp_path: Path, monkeypatch
) -> None:
    server = make_server(reward_mode="binary")
    server.config = server.config.model_copy(update={"preconvert_office_to_pdf": True})
    monkeypatch.setattr(
        "resources_servers.aa_briefcase_lite.app.preconvert_dir_async",
        AsyncMock(return_value=(0, 1, ["report.docx: soffice exited 1"])),
    )
    server._aa_checks = [{**c, "taskdoer_output_file": "report.docx"} for c in server._aa_checks]

    row = await _verify(server, tmp_path, {"report.docx": "not really a document"})

    assert row["mask_sample"] is True
    assert row["failure_kind"] == "verifier_error"
    assert "conversion failed" in row["failure_reason"]


async def test_an_office_file_without_a_render_masks_the_row_as_a_verifier_error(make_server, tmp_path: Path) -> None:
    server = make_server(reward_mode="binary")
    server._aa_checks = [{**c, "taskdoer_output_file": "report.docx"} for c in server._aa_checks]

    row = await _verify(server, tmp_path, {"report.docx": "not really a document"})

    assert row["mask_sample"] is True
    assert row["failure_kind"] == "verifier_error"
    assert "filename-only stub" in row["failure_reason"]


async def test_a_task_without_checks_masks_the_row_as_a_verifier_error(make_server, tmp_path: Path) -> None:
    server = make_server(reward_mode="binary")
    server._aa_checks = []

    row = await _verify(server, tmp_path, {"report.md": "x"})

    assert row["failure_kind"] == "verifier_error"
    assert row["mask_sample"] is True


def test_pairwise_mode_without_references_is_refused_at_startup(make_server) -> None:
    server = make_server(reward_mode="pairwise", pairwise_references={}, pairwise_reference_ids=[])

    with pytest.raises(ValueError, match="at least one pairwise reference"):
        server._require_pairwise_references()


async def test_a_missing_deliverable_is_a_model_zero_and_is_not_judged(
    make_server, tmp_path: Path, monkeypatch
) -> None:
    server = make_server(reward_mode="all")
    binary = AsyncMock(return_value=PASSED)
    monkeypatch.setattr(server, "_binary_call", binary)
    pairwise = []
    monkeypatch.setattr(
        "resources_servers.aa_briefcase_lite.app.run_trials", lambda **kwargs: pairwise.append(kwargs) or {}
    )

    # The agent abandoned the task: only the finish record is in the directory.
    row = await _verify(server, tmp_path, {"finish_params.json": "null"})

    assert not row.get("mask_sample")
    assert row["reward"] == 0.0
    assert row["rubric_passed"] == 0
    assert row["rubric_total"] == 2
    assert row["pairwise_losses"] == 4
    assert row["pairwise_wins"] == 0
    assert binary.await_count == 0
    assert pairwise == []
    binary_results = row["judge_response"]["binary_results"]
    assert [r["missing_deliverables"] for r in binary_results] == [["report.md"], ["report.md"]]
    assert all(r["passed"] is False and r["judge_name"] is None for r in binary_results)


async def test_a_check_is_judged_when_its_own_file_was_submitted(make_server, tmp_path: Path, monkeypatch) -> None:
    server = make_server(reward_mode="binary")
    server._aa_checks = [
        {**server._aa_checks[0], "taskdoer_output_file": "report.md"},
        {**server._aa_checks[1], "taskdoer_output_file": "slides.md"},
        server._aa_checks[2],
    ]
    binary = AsyncMock(return_value=PASSED)
    monkeypatch.setattr(server, "_binary_call", binary)

    row = await _verify(server, tmp_path, {"report.md": "the report"})

    assert binary.await_count == 1
    assert row["rubric_passed"] == 1
    assert row["rubric_total"] == 2
    assert row["reward"] == 0.5


async def test_a_skipped_reference_is_logged_and_counted(make_server, tmp_path: Path, monkeypatch, caplog) -> None:
    server = make_server(reward_mode="pairwise")
    (tmp_path / "ref_b" / "task_w1_t1" / "repeat_0" / "report.md").unlink()
    (tmp_path / "ref_b" / "task_w1_t1" / "repeat_0").rmdir()
    monkeypatch.setattr(
        "resources_servers.aa_briefcase_lite.app.run_trials", lambda **kwargs: pairwise_result(wins=1, losses=1)
    )

    with caplog.at_level(logging.WARNING):
        row = await _verify(server, tmp_path, {"report.md": "x"})

    assert "skipped pairwise reference ref_b" in caplog.text
    assert row["pairwise_references_skipped"] == 1
    assert row["judge_response"]["pairwise_skipped_references"] == [
        {"reference_id": "ref_b", "reason": "no_submission_for_task"}
    ]
    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[row]))
    assert metrics.key_metrics["aa_lite/pairwise_references_skipped"] == 1


async def test_conversion_uses_the_configured_libreoffice_command(make_server, tmp_path: Path, monkeypatch) -> None:
    # The host has no LibreOffice of its own: only the configured command works.
    monkeypatch.setattr(
        "resources_servers.gdpval.setup_libreoffice.ensure_libreoffice", lambda command=None, **kwargs: bool(command)
    )
    server = make_server(reward_mode="binary", libreoffice_command=["wrapper", "soffice"])
    server.config = server.config.model_copy(update={"preconvert_office_to_pdf": True})
    convert = AsyncMock(return_value=(1, 0, []))
    monkeypatch.setattr("resources_servers.aa_briefcase_lite.app.preconvert_dir_async", convert)
    monkeypatch.setattr(server, "_binary_call", AsyncMock(return_value=PASSED))

    server._require_office_rendering()
    await _verify(server, tmp_path, {"report.md": "x"})

    assert convert.await_args.kwargs["libreoffice_command"] == ["wrapper", "soffice"]


async def test_the_rate_of_a_path_that_did_not_run_is_absent_not_zero(
    make_server, tmp_path: Path, monkeypatch
) -> None:
    server = make_server(reward_mode="binary")
    monkeypatch.setattr(server, "_binary_call", AsyncMock(return_value=PASSED))
    row = await _verify(server, tmp_path, {"report.md": "x"})

    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[row]))

    assert metrics.key_metrics["aa_lite/binary_pass_rate"] == 1.0
    assert "aa_lite/pairwise_win_rate" not in metrics.key_metrics


async def test_a_missing_deliverables_directory_masks_the_row_instead_of_scoring_every_check_as_missing(
    make_server, tmp_path: Path, monkeypatch
) -> None:
    server = make_server(reward_mode="all")
    binary = AsyncMock(return_value=PASSED)
    monkeypatch.setattr(server, "_binary_call", binary)

    # A reverify on a host that cannot see the persisted deliverables, or a row whose directory was moved.
    row = _row(await judge_failsafe(server.verify)(NO_SESSION, verify_body(str(tmp_path / "gone"))))

    assert row["mask_sample"] is True
    assert row["failure_kind"] == "verifier_error"
    assert "is not a directory" in row["failure_reason"]
    assert binary.await_count == 0


async def test_an_unhydrated_lfs_pointer_masks_the_row_as_a_verifier_error(make_server, tmp_path: Path) -> None:
    server = make_server(reward_mode="pairwise")
    pointer = tmp_path / "ref_a" / "task_w1_t1" / "repeat_0" / "report.md"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 123456\n")

    row = await _verify(server, tmp_path, {"report.md": "x"})

    assert row["mask_sample"] is True
    assert row["failure_kind"] == "verifier_error"
    assert "unhydrated Git LFS pointer" in row["failure_reason"]
