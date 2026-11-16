# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run whose latest resume crossed operational configuration drift says so in every recovery decision.

The resume writes the drifted settings to the run's resume-drift journal; the collector reads its last line, and the
decision reason (what `gym eval resubmit --recover --inspect`, a recovery and a chain continuation print) ends with
"resumed across operational drift: <settings>".
"""

import json
from pathlib import Path

from nemo_gym.orchestration.recovery import decide_from_collected, resume_drift_note
from nemo_gym.rollouts.rollout_recovery import resume_drift_path_for, validate_resume
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect, ident, write_job_dir
from tests.unit_tests.rollouts.test_resume_drift import BASE, VLLM, current_manifest, merged, saved_run


def drift_journal(job: Path, *lines: list[str]) -> None:
    path = job / "artifacts" / "rollouts_manifest.resume-drift.jsonl"
    path.write_text("".join(json.dumps({"operational_keys": keys}) + "\n" for keys in lines))


def decide(job: Path):
    return decide_from_collected(collect(job), max_infra_retries=3, max_walltime_s=None, live_job=False)


def test_a_drift_line_gives_the_sentence_in_the_decision_reason(tmp_path: Path) -> None:
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[])
    drift_journal(job, ["servers.p.responses_api_models.vllm_model.request_timeout_s", "collection.max_retries"])

    collected = collect(job)
    decision = decide(job)

    assert collected.resume_drift_keys == [
        "servers.p.responses_api_models.vllm_model.request_timeout_s",
        "collection.max_retries",
    ]
    assert decision.reason.endswith(
        "; resumed across operational drift: collection.max_retries, "
        "servers.p.responses_api_models.vllm_model.request_timeout_s"
    )


def test_no_journal_and_a_latest_resume_without_drift_give_no_sentence(tmp_path: Path) -> None:
    job = write_job_dir(tmp_path / "job", n=2, results=[ident(0, reward=1.0)], failures=[])
    assert collect(job).resume_drift_keys is None
    assert "operational drift" not in decide(job).reason

    drift_journal(job, ["collection.max_retries"], [])  # a later resume crossed no drift
    assert collect(job).resume_drift_keys is None
    assert "operational drift" not in decide(job).reason


def test_the_sentence_sorts_and_caps_the_settings() -> None:
    keys = [f"servers.s.k{index:02d}" for index in range(25, 0, -1)]
    note = resume_drift_note(keys)
    assert note.startswith("resumed across operational drift: servers.s.k01, servers.s.k02,")
    assert note.endswith("servers.s.k20 and 5 more")


def test_a_resume_without_drift_after_one_with_drift_records_an_empty_line(tmp_path: Path) -> None:
    path, materialized = saved_run(tmp_path, BASE)
    drifted = merged(BASE, {f"{VLLM}.request_timeout_s": 1200})
    assert validate_resume(path, current_manifest(tmp_path, drifted), materialized) is not None
    # The next resume on the configuration it was saved with crosses no drift; the journal's last line says so.
    assert validate_resume(path, current_manifest(tmp_path, BASE), materialized) is not None

    lines = [json.loads(line) for line in resume_drift_path_for(path).read_text().splitlines()]
    assert [line["operational_keys"] for line in lines] == [
        ["servers.policy_model.responses_api_models.vllm_model.request_timeout_s"],
        [],
    ]


def test_an_unchanged_run_never_gets_a_drift_journal(tmp_path: Path) -> None:
    path, materialized = saved_run(tmp_path, BASE)
    assert validate_resume(path, current_manifest(tmp_path, BASE), materialized) is not None
    assert not resume_drift_path_for(path).exists()


def test_inspect_reports_the_drift_of_a_submitted_run(tmp_path: Path, slurm_bin) -> None:
    from nemo_gym.orchestration.submit import inspect_recovery
    from tests.unit_tests.orchestration.test_eval_resubmit import _handle
    from tests.unit_tests.orchestration.test_resubmit_walltime import _timed_run

    job_dir = _timed_run(tmp_path)
    drift_journal(job_dir, ["servers.policy_model.responses_api_models.vllm_model.request_timeout_s"])

    out = inspect_recovery(*_handle(job_dir))

    assert out.decision.reason.endswith(
        "resumed across operational drift: servers.policy_model.responses_api_models.vllm_model.request_timeout_s"
    )
