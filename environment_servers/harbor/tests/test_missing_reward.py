# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A dataset's missing_reward.py: loading it beside dataset.toml, and what each verdict does to the episode."""

import json
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from environment_servers.harbor import verifier
from environment_servers.harbor.tests.conftest import (
    SEPARATE_IN_TASK_IMAGE_TOML,
    TASK_TOML,
    AgentSandbox,
    FakeSandbox,
    make_server,
    run_episode,
)
from environment_servers.harbor.tests.test_multi_turn import THREE_STEPS, multi_turn_server, request_for
from environment_servers.harbor.tests.test_verifier_filesystem import (  # noqa: F401
    TRAJECTORY,
    LocalSandbox,
    verifier_tree,
)
from environment_servers.harbor.verifier import ARTIFACTS_DIR, VERIFIER_LOGS_DIR
from nemo_gym.sandbox import SandboxExecResult


VERDICT = """
from nemo_gym.data.tasks.harbor import MissingRewardVerdict

def classify_missing_reward(evidence):
    return {body}
"""


def dataset_classifier(tmp_path: Path, body: str) -> None:
    """missing_reward.py beside the dataset.toml of the taskset ``make_server`` serves."""
    folder = tmp_path / "datasets" / "ds"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "missing_reward.py").write_text(VERDICT.format(body=body))


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("raise RuntimeError('boom')", "failed to import"),
        ("x = 1", "does not export"),
        ("classify_missing_reward = 1", "is not callable"),
    ],
    ids=["import-fails", "no-export", "not-callable"],
)
async def test_a_classifier_that_does_not_load_refuses_the_tasksets_episodes(tmp_path, monkeypatch, source, fragment):
    folder = tmp_path / "datasets" / "ds"
    folder.mkdir(parents=True)
    (folder / "missing_reward.py").write_text(source)
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    response = await run_episode(server, task)
    assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
    assert fragment in response.failure.failure_reason
    assert not any("test.sh" in call["command"] for call in sandbox.execs)


@pytest.mark.parametrize(
    ("body", "mask", "kind"),
    [
        (
            "MissingRewardVerdict('task', 'bench:judge_failed', 'the deliverable is broken')",
            False,
            "bench:judge_failed",
        ),
        ("MissingRewardVerdict('infrastructure', 'bench:crashed', 'grader crashed')", True, "bench:crashed"),
        ("None", False, "harbor:missing_reward"),
        ("1 / 0", True, "verifier_error"),
        ("MissingRewardVerdict('maybe', 'bench:x', None)", True, "verifier_error"),
        # A shared kind on a task 0 would score a fault of ours, and an invented bare kind is no kind at all.
        ("MissingRewardVerdict('task', 'judge_failed', 'mislabelled')", True, "verifier_error"),
        ("MissingRewardVerdict('infrastructure', 'crashed', 'not namespaced')", True, "verifier_error"),
    ],
    ids=["task", "infrastructure", "none", "raises", "not-a-verdict", "task-shared-kind", "bare-kind"],
)
async def test_each_verdict_scores_a_missing_reward(tmp_path, monkeypatch, body, mask, kind):
    dataset_classifier(tmp_path, body)
    server, task, _, _ = make_server(tmp_path, monkeypatch, FakeSandbox(verifier_files={"test-stdout.txt": "no"}))
    result = (await run_episode(server, task)).result
    assert result.reward == 0 and result.mask_sample is mask and result.failure_kind == kind


JUDGE_FAILED = "MissingRewardVerdict('judge_failed', 'bench:judge_refused', 'the judge service refused the call')"


async def test_a_judge_failure_is_routed_as_gyms_judge_failed_class(tmp_path, monkeypatch):
    dataset_classifier(tmp_path, JUDGE_FAILED)
    server, task, _, _ = make_server(tmp_path, monkeypatch, FakeSandbox(verifier_files={}))
    result = (await run_episode(server, task)).result.model_dump()
    assert result["mask_sample"] is True and result["evaluation_completed"] is False
    assert result["failure_kind"] == "judge_failed" and result["_ng_failure_class"] == "judge_failed"
    assert "bench:judge_refused" in result["_ng_failure_judge_error"]


async def test_a_judge_failure_in_a_step_routes_the_multi_step_episode(tmp_path, monkeypatch):
    dataset_classifier(tmp_path, JUDGE_FAILED)
    server, task, sandbox, _ = multi_turn_server(
        tmp_path, monkeypatch, THREE_STEPS, {"plan": "P", "build": "B", "check": "C"}
    )
    pending = [{"reward.txt": "1\n"}, {}, {"reward.txt": "1\n"}]

    def next_files():
        sandbox.verifier_files = pending.pop(0)

    server.server_client.turn_hooks.append(next_files)
    response = await server.run_request(request_for(task))
    result = response.result.model_dump()
    assert [turn["failure_kind"] for turn in result["turns"]] == [None, "judge_failed"]
    assert result["mask_sample"] is True and result["_ng_failure_class"] == "judge_failed"


@pytest.mark.parametrize(
    ("files", "test_result"),
    [
        ({"reward.txt": "0.5\n"}, None),
        ({"reward.json": '{"reward": "x"}'}, None),
        ({}, SandboxExecResult(stdout=None, stderr="timeout", return_code=124, error_type="timeout")),
    ],
    ids=["valid", "invalid", "timeout"],
)
async def test_only_a_finished_run_without_a_reward_file_reaches_the_classifier(
    tmp_path, monkeypatch, files, test_result
):
    sources: list[str] = []
    download = verifier.download_dir

    async def recording(sandbox, source, target, **kwargs):
        sources.append(source)
        return await download(sandbox, source, target, **kwargs)

    monkeypatch.setattr(verifier, "download_dir", recording)
    dataset_classifier(tmp_path, "1 / 0")
    sandbox = FakeSandbox(verifier_files=dict(files))
    if test_result is not None:
        sandbox.test_result = test_result
    server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox)
    result = (await run_episode(server, task)).result
    assert result.failure_kind != "verifier_error"
    assert ARTIFACTS_DIR not in sources and sources.count(VERIFIER_LOGS_DIR) == 2


async def test_evidence_that_does_not_transfer_masks_the_sample_without_the_classifier(tmp_path, monkeypatch):
    download = verifier.download_dir

    async def reward_files_only(sandbox, source, target, **kwargs):
        if kwargs.get("names") is None:
            raise verifier.SandboxTransferError("the sandbox went away")
        return await download(sandbox, source, target, **kwargs)

    monkeypatch.setattr(verifier, "download_dir", reward_files_only)
    dataset_classifier(tmp_path, "MissingRewardVerdict('task', 'bench:seen', 'classified')")
    server, task, _, _ = make_server(tmp_path, monkeypatch, FakeSandbox(verifier_files={}))
    result = (await run_episode(server, task)).result
    assert result.mask_sample is True and result.failure_kind == "provider_unavailable"


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_the_classifier_reads_the_artifacts_test_sh_left_without_links(tmp_path, monkeypatch, verifier_tree):  # noqa: F811
    root, logs, _ = verifier_tree
    artifacts = logs / "artifacts"
    monkeypatch.setattr(verifier, "ARTIFACTS_DIR", str(artifacts))
    outside = root / "private"
    outside.write_text("must not be copied")
    dataset_classifier(
        tmp_path,
        "MissingRewardVerdict('task', 'bench:seen', "
        "repr(sorted(p.name for p in evidence.artifacts_dir.rglob('*'))) + ' ' + evidence.metadata['family'])",
    )
    toml = TASK_TOML + '\n[metadata]\nfamily = "repurpose"\n'
    server, task, _, _ = make_server(tmp_path, monkeypatch, toml=toml)
    (task.path / "tests" / "test.sh").write_text(
        f"mkdir -p {shlex.quote(str(artifacts / 'nested'))}\n"
        f"echo '{json.dumps({'answer': 1})}' > {shlex.quote(str(artifacts / 'deliverable.json'))}\n"
        f"echo 2 > {shlex.quote(str(artifacts / 'nested' / 'part.json'))}\n"
        f"ln -s {shlex.quote(str(outside))} {shlex.quote(str(artifacts / 'link'))}\n"
    )
    session = SimpleNamespace(task=task, taskset="ds", sandbox=LocalSandbox(), workdir=str(root), entrypoint=None)
    outcome = await server._run_verifier(session, TRAJECTORY)
    assert outcome["failure_kind"] == "bench:seen" and outcome["mask_sample"] is False
    assert outcome["failure_reason"] == "['deliverable.json', 'nested', 'part.json'] repurpose"
    await server.shutdown()


async def test_a_classifier_edited_after_materialization_refuses_the_episode(tmp_path, monkeypatch):
    """The snapshot carries the dataset's missing_reward.py into the task digest, so an edit changes the digest the
    rows and the mapping were materialized under."""
    dataset_classifier(tmp_path, "None")
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch, FakeSandbox(verifier_files={}))
    dataset_classifier(tmp_path, JUDGE_FAILED)
    response = await run_episode(server, task)
    assert response.result is None and response.failure.stage == "seed" and response.failure.terminal
    assert "changed since materialization" in response.failure.failure_reason
    assert not any("test.sh" in call["command"] for call in sandbox.execs)


# The loader admits a separate verifier in the task image only with a declared image.
SEPARATE_IN_IMAGE = SEPARATE_IN_TASK_IMAGE_TOML.replace(
    "[environment]\n", '[environment]\ndocker_image = "ubuntu:24.04"\n'
)


async def test_a_separate_verifier_classifies_from_its_own_sandbox(tmp_path, monkeypatch):
    dataset_classifier(tmp_path, JUDGE_FAILED)
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
    )
    server, task, _, _ = make_server(tmp_path, monkeypatch, AgentSandbox(), toml=SEPARATE_IN_IMAGE)
    verifier_sandbox = FakeSandbox(verifier_files={})

    async def create_verifier(task, *, episode_key=None):
        server._sandboxes[f"{episode_key}/verifier"] = verifier_sandbox
        return verifier_sandbox

    monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
    result = (await run_episode(server, task)).result.model_dump()
    assert result["verifier_mode"] == "separate" and result["_ng_failure_class"] == "judge_failed"
    assert verifier_sandbox.stopped
