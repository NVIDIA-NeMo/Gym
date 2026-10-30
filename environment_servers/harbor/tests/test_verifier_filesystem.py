# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import shlex
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from environment_servers.harbor import session, verifier
from environment_servers.harbor.tests.conftest import (
    SEPARATE_TOML,
    TASK_TOML,
    AgentSandbox,
    FakeSandbox,
    make_server,
    run_episode,
)
from environment_servers.harbor.tests.test_review_regressions import during_turn
from environment_servers.harbor.verifier import verifier_command
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.tasks.harbor import load_task


# What run() hands the verifier: the exported ATIF transcript, written to /logs/agent before test.sh runs.
TRAJECTORY = b'{"schema_version": "ATIF-v1.7"}\n'


class LocalSandbox:
    """Run production transfer and verifier commands on a pytest-owned filesystem tree."""

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        return SandboxExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
        )

    async def upload(self, source, target):
        shutil.copyfile(source, target)

    async def download(self, source, target):
        shutil.copyfile(source, target)


@pytest.fixture
def verifier_tree(tmp_path, monkeypatch):
    root = tmp_path / "sandbox"
    root.mkdir()
    logs, tests = root / "logs", root / "tests"
    monkeypatch.setattr(verifier, "TESTS_DIR", str(tests))
    monkeypatch.setattr(verifier, "VERIFIER_LOGS_DIR", str(logs / "verifier"))
    monkeypatch.setattr(session, "VERIFIER_LOGS_DIR", str(logs / "verifier"))
    monkeypatch.setattr(verifier, "AGENT_LOGS_DIR", str(logs / "agent"))
    monkeypatch.setattr(session, "AGENT_LOGS_DIR", str(logs / "agent"))
    monkeypatch.setattr(verifier, "VERIFIER_STDOUT", str(logs / "verifier" / "test-stdout.txt"))
    return root, logs, tests


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
@pytest.mark.parametrize("linked", ["none", "tests", "verifier", "logs"])
async def test_linked_verifier_roots_cannot_preserve_planted_success(tmp_path, monkeypatch, verifier_tree, linked):
    root, logs, tests = verifier_tree
    planted = root / "planted"
    planted.mkdir()
    (planted / "reward.json").write_text('{"reward": 1}')
    (planted / "helper.sh").write_text("echo 1")
    if linked == "logs":
        (planted / "verifier").mkdir()
        (planted / "verifier" / "reward.json").write_text('{"reward": 1}')
        logs.symlink_to(planted, target_is_directory=True)
    else:
        logs.mkdir()
        if linked != "none":
            target = tests if linked == "tests" else logs / "verifier"
            target.symlink_to(planted, target_is_directory=True)
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    (task.path / "tests" / "test.sh").write_text(
        f"#!/bin/bash\nif test -f {shlex.quote(str(tests / 'helper.sh'))}; then\n"
        f"  bash {shlex.quote(str(tests / 'helper.sh'))}\nelse\n  echo 0\nfi "
        f"> {shlex.quote(str(logs / 'verifier' / 'reward.txt'))}\n"
    )
    outcome = await server._run_verifier(
        SimpleNamespace(task=task, sandbox=LocalSandbox(), workdir=str(root), entrypoint=None), TRAJECTORY
    )
    assert outcome["reward"] == 0
    assert not outcome.get("mask_sample", False)
    assert not tests.is_symlink() and not logs.is_symlink() and not (logs / "verifier").is_symlink()
    assert not (tests / "helper.sh").exists()
    assert (planted / "reward.json").read_text() == '{"reward": 1}'
    await server.shutdown()


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
@pytest.mark.parametrize("left_behind", ["declared_input", "undeclared_input", "planted_reward", "planted_report"])
async def test_the_shared_verifier_keeps_declared_sidecar_inputs_alone_and_removes_everything_else(
    tmp_path, monkeypatch, verifier_tree, left_behind
):
    """Review (M22, then round 38 P1): kumo's ``verifier`` sidecar appends ``/logs/verifier/actions.jsonl`` during
    the agent's run, shared with ``main`` by the planner, and ``main``'s ``tests/score.py`` reads it; emptying the
    whole folder before ``test.sh`` deleted that input, so a test that needs it measured 0 for 1. Keeping everything
    but the grader's own four names kept an agent-planted test report too: an all-pass ``results.xml`` left behind by
    a solution that exits during pytest's import, which a ``test.sh`` that reads the report scores as a pass. The
    folder is emptied again now, except the inputs the task declares (``[verifier].inputs`` from dataset.toml): a
    declared ``actions.jsonl`` is read by ``test.sh``, an undeclared one is gone like a planted ``reward.json`` or
    ``results.xml``, in a single-step shared task as in a step's shared verification."""
    root, logs, tests = verifier_tree
    verifier_dir = logs / "verifier"
    verifier_dir.mkdir(parents=True)
    reward = shlex.quote(str(verifier_dir / "reward.txt"))
    if left_behind == "declared_input":
        (tmp_path / "datasets" / "ds").mkdir(parents=True)
        (tmp_path / "datasets" / "ds" / "dataset.toml").write_text(
            '[gym.tasks."hello".verifier]\ninputs = ["actions.jsonl"]\n'
        )
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    assert task.config.verifier.inputs == (["actions.jsonl"] if left_behind == "declared_input" else [])
    if left_behind in ("declared_input", "undeclared_input"):
        (verifier_dir / "actions.jsonl").write_text('{"action": "click"}\n')
        actions = shlex.quote(str(verifier_dir / "actions.jsonl"))
        (task.path / "tests" / "test.sh").write_text(
            f"#!/bin/bash\nif test -s {actions}; then echo 1; else echo 0; fi > {reward}\n"
        )
    elif left_behind == "planted_reward":
        (verifier_dir / "reward.json").write_text('{"reward": 1}')
        (task.path / "tests" / "test.sh").write_text(f"#!/bin/bash\necho 0 > {reward}\n")
    else:
        # The report pytest would have written had it run; the solution exited during import, so it writes none.
        (verifier_dir / "results.xml").write_text('<testsuite tests="3" failures="0" errors="0"/>\n')
        report = shlex.quote(str(verifier_dir / "results.xml"))
        (task.path / "tests" / "test.sh").write_text(
            f"#!/bin/bash\nif test -f {report} && ! grep -q 'failures=\"[1-9]' {report}; then echo 1; else echo 0; fi "
            f"> {reward}\n"
        )

    outcome = await server._run_verifier(
        SimpleNamespace(task=task, sandbox=LocalSandbox(), workdir=str(root), entrypoint=None), TRAJECTORY
    )

    assert not outcome.get("mask_sample", False), outcome
    if left_behind == "declared_input":
        assert outcome["reward"] == 1, outcome
        assert (verifier_dir / "actions.jsonl").read_text() == '{"action": "click"}\n'
    else:
        assert outcome["reward"] == 0, outcome
        assert not any((verifier_dir / name).exists() for name in ("actions.jsonl", "reward.json", "results.xml"))
    await server.shutdown()


async def separate_verification(tmp_path, monkeypatch, verifier_tree, artifacts: list[str]) -> dict:
    """One separate-mode verification on the pytest-owned tree, with real shell and tar on both sides.

    The agent's sandbox and the fresh verifier sandbox are the same host tree: every artifact is collected
    from it, stopping the agent's sandbox empties the tree (the verifier starts fresh, as its own sandbox
    would), and the restores bring back what the agent left, planted files included, the way they would into
    a real verifier sandbox. ``artifacts`` are host paths under the tree; the trusted ``test.sh`` writes reward 0.
    """
    root, logs, tests = verifier_tree

    class AgentSandbox(LocalSandbox):
        async def stop(self):
            for child in root.iterdir():
                shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()

    monkeypatch.setattr(verifier, "ARTIFACTS_DIR", str(logs / "artifacts"))
    entries = ", ".join(f'"{source}"' for source in artifacts)
    toml = (
        'schema_version = "1.4"\n\n'
        f"artifacts = [{entries}]\n\n"
        '[verifier]\ntimeout_sec = 300.0\nenvironment_mode = "separate"\n\n'
        '[verifier.environment]\ndocker_image = "org/verifier:1"\n\n'
        "[agent]\ntimeout_sec = 120.0\n\n"
        '[environment]\ndocker_image = "org/agent:1"\n'
    )
    server, task, _, _ = make_server(tmp_path, monkeypatch, toml=toml)
    (task.path / "tests" / "test.sh").write_text(
        f"#!/bin/bash\necho 0 > {shlex.quote(str(logs / 'verifier' / 'reward.txt'))}\n"
    )
    agent = AgentSandbox()

    async def create_verifier(task, *, episode_key=None):
        return LocalSandbox()

    async def nothing():
        return None

    monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)
    session = SimpleNamespace(
        task=task,
        sandbox=agent,
        workdir=str(root),
        entrypoint=None,
        session_id="r1",
        service=lambda name: agent,
        stop_main=agent.stop,
        stop_agent_side=nothing,
    )
    outcome = await server._run_verifier(session, TRAJECTORY)
    await server.shutdown()
    return outcome


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_the_uploaded_transcript_is_readable_by_a_non_root_verifier_user_under_umask_077(
    tmp_path, monkeypatch, verifier_tree
):
    """The host copy of the transcript is created under the server's umask and the archive carries that mode, which
    root tar keeps: under umask 077 `/logs/agent/trajectory.json` lands root-owned 0600, and a non-root
    `[verifier].user` cannot read it. The reserved file alone is opened to every reader after the upload."""
    _, logs, _ = verifier_tree
    server, _, _, _ = make_server(tmp_path, monkeypatch)
    logs_dir = tmp_path / "verifier_logs"
    logs_dir.mkdir()
    previous = os.umask(0o077)
    try:
        await server._upload_trajectory(LocalSandbox(), TRAJECTORY, logs_dir)
    finally:
        os.umask(previous)
    uploaded = logs / "agent" / "trajectory.json"
    assert uploaded.read_bytes() == TRAJECTORY
    mode = stat.S_IMODE(uploaded.stat().st_mode)
    assert mode & 0o444 == 0o444, f"trajectory.json is {oct(mode)}: a non-root verifier user cannot read it"
    # The host copy beside the verifier logs is the diagnostic, written as the server's umask says.
    host = logs_dir / "trajectory.json"
    assert host.read_bytes() == TRAJECTORY and stat.S_IMODE(host.stat().st_mode) == 0o600


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
@pytest.mark.parametrize("leading", ["/", "//"], ids=["logs", "double_slash_logs"])
async def test_a_restored_artifact_over_the_logs_cannot_replace_the_graders_reward(
    tmp_path, monkeypatch, verifier_tree, leading
):
    """The reviewer's case: ``artifacts = ["/logs"]`` is valid and brought the agent's ``/logs/verifier/reward.json``
    into the fresh verifier sandbox right before ``test.sh`` ran, and its reward of 1 beat the ``reward.txt`` of 0
    that the trusted tests wrote, unmasked. The restore still brings the agent's folders back (Harbor restores
    ``/logs`` whole), but ``/logs/verifier`` is emptied again after it, so the grader's result is the only one.
    Spelled ``//logs`` the entry names the same directory and is held to the same rule."""
    root, logs, tests = verifier_tree
    for folder in ("verifier", "agent", "artifacts"):
        (logs / folder).mkdir(parents=True)
    (logs / "verifier" / "reward.json").write_text('{"reward": 1}')
    (logs / "agent" / "agent.log").write_text("agent output")
    (logs / "artifacts" / "state.db").write_text("db")

    outcome = await separate_verification(tmp_path, monkeypatch, verifier_tree, [leading + str(logs).lstrip("/")])

    assert outcome["reward"] == 0 and not outcome.get("mask_sample", False), outcome
    assert "failure_kind" not in outcome and outcome["verifier_mode"] == "separate"
    assert not (logs / "verifier" / "reward.json").exists()
    assert (logs / "verifier" / "reward.txt").read_text().strip() == "0"
    # The agent's other folders came back with the restore, and the transcript was written after it.
    assert (logs / "agent" / "agent.log").read_text() == "agent output"
    assert (logs / "artifacts" / "state.db").read_text() == "db"
    assert (logs / "agent" / "trajectory.json").read_bytes() == TRAJECTORY
    # The planted file was collected, as the host diagnostic every artifact leaves, under /logs's own staging root.
    staged = Path(outcome["verifier_logs_dir"]) / "artifacts" / "01" / str(logs).lstrip("/")
    assert (staged / "verifier" / "reward.json").read_text() == '{"reward": 1}'


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
@pytest.mark.parametrize(
    "shape", ["tests", "double_slash_tests", "verifier_dir", "verifier_file", "file_through_link", "dir_through_link"]
)
async def test_an_artifact_that_reaches_the_tests_or_the_verifier_outputs_is_collected_but_never_restored(
    tmp_path, monkeypatch, verifier_tree, caplog, shape
):
    """An artifact at ``/tests`` would replace the trusted test script, one at or inside ``/logs/verifier`` would
    plant a reward, and one whose path runs through a link an earlier artifact restored (``/app/out -> /tests``)
    would do the first by another name. Each is collected on the host and refused at the restore with a warning,
    the trusted ``test.sh`` runs as uploaded and the sample is scored on what it wrote."""
    root, logs, tests = verifier_tree
    rigged = f"#!/bin/bash\necho 1 > {shlex.quote(str(logs / 'verifier' / 'reward.txt'))}\n"
    if shape in ("tests", "double_slash_tests", "file_through_link", "dir_through_link"):
        tests.mkdir()
        (tests / "test.sh").write_text(rigged)
    else:
        (logs / "verifier").mkdir(parents=True)
        (logs / "verifier" / "reward.json").write_text('{"reward": 1}')
    if shape.endswith("through_link"):
        (root / "app").mkdir()
        (root / "app" / "out").symlink_to(tests, target_is_directory=True)
    artifacts = {
        "tests": [str(tests)],
        "double_slash_tests": ["/" + str(tests)],
        "verifier_dir": [str(logs / "verifier")],
        "verifier_file": [str(logs / "verifier" / "reward.json")],
        "file_through_link": [str(root / "app"), str(root / "app" / "out" / "test.sh")],
        "dir_through_link": [str(root / "app"), str(root / "app" / "out")],
    }[shape]

    with caplog.at_level("WARNING"):
        outcome = await separate_verification(tmp_path, monkeypatch, verifier_tree, artifacts)

    assert outcome["reward"] == 0 and not outcome.get("mask_sample", False), outcome
    assert "failure_kind" not in outcome
    assert f"artifact {'/' + artifacts[-1].lstrip('/')!r} not restored" in caplog.text
    assert (tests / "test.sh").read_text().startswith("#!/bin/bash\necho 0 > ")
    assert not (logs / "verifier" / "reward.json").exists()
    if shape.endswith("through_link"):
        # The first artifact was restored, link and all; only the one that ran through the link was refused.
        assert (root / "app" / "out").is_symlink() and f"artifact {artifacts[0]!r} not restored" not in caplog.text
        assert "physically is" in caplog.text
    else:
        assert "would replace the grader's" in caplog.text
    # Every artifact was collected: the refused one is on the host under its own staging root.
    staged = Path(outcome["verifier_logs_dir"]) / "artifacts" / f"{len(artifacts):02d}" / artifacts[-1].lstrip("/")
    assert staged.is_file() if shape in ("verifier_file", "file_through_link") else staged.is_dir()


@pytest.mark.parametrize("literal", ["9" * 400, "NaN"], ids=["overflow", "nan"])
async def test_non_finite_numeric_reward_is_measured_zero(tmp_path, monkeypatch, literal):
    """An integer too large for a float and a JSON NaN are both numbers that cannot be a score."""
    sandbox = FakeSandbox(verifier_files={"reward.json": '{"reward": ' + literal + "}"})
    server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox)
    result = (await run_episode(server, task)).result
    assert result.reward == 0 and result.mask_sample is False
    assert result.failure_kind == "harbor:invalid_reward"
    assert sum("test.sh" in call["command"] and "timeout" in call["command"] for call in sandbox.execs) == 1


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_unsafe_diagnostic_link_does_not_mask_valid_reward(tmp_path, monkeypatch, verifier_tree):
    root, logs, _ = verifier_tree
    outside = root / "private"
    outside.write_text("must not be copied")
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    (task.path / "tests" / "test.sh").write_text(
        f"#!/bin/bash\necho 1 > {shlex.quote(str(logs / 'verifier/reward.txt'))}\n"
        f"ln -s {shlex.quote(str(outside))} {shlex.quote(str(logs / 'verifier/diagnostic-link'))}\n"
        "echo useful-output\n"
    )
    outcome = await server._run_verifier(
        SimpleNamespace(task=task, sandbox=LocalSandbox(), workdir=str(root), entrypoint=None), TRAJECTORY
    )
    assert outcome["reward"] == 1 and not outcome.get("mask_sample", False)
    artifacts = Path(outcome["verifier_logs_dir"])
    assert not (artifacts / "diagnostic-link").exists()
    assert "useful-output" in (artifacts / "test-stdout.txt").read_text()
    assert all("must not be copied" not in path.read_text() for path in artifacts.rglob("*") if path.is_file())
    await server.shutdown()


async def test_timeout_retains_logs_and_is_measured_zero(tmp_path, monkeypatch):
    sandbox = FakeSandbox(
        verifier_files={"test-stdout.txt": "verifier reached checkpoint"},
        test_result=SandboxExecResult(stdout=None, stderr="timeout", return_code=124, error_type="timeout"),
    )
    server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox)
    result = (await run_episode(server, task)).result
    assert result.reward == 0 and result.mask_sample is False
    assert result.failure_kind == "harbor:verifier_timeout"
    assert result.verifier_return_code == 124
    assert "checkpoint" in (Path(result.verifier_logs_dir) / "test-stdout.txt").read_text()
    assert sum("test.sh" in call["command"] and "timeout" in call["command"] for call in sandbox.execs) == 1


async def test_optional_diagnostic_transfer_failure_preserves_score(tmp_path, monkeypatch):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    download = sandbox.download
    calls = 0

    async def fail_diagnostics(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("diagnostic transfer failed")
        await download(source, target)

    monkeypatch.setattr(sandbox, "download", fail_diagnostics)
    outcome = await server._run_verifier(
        SimpleNamespace(task=task, sandbox=sandbox, workdir="/app", entrypoint=None), TRAJECTORY
    )
    assert calls == 2
    assert outcome["reward"] == 1 and not outcome.get("mask_sample", False)
    await server.shutdown()


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_unsafe_reward_link_still_masks_transfer_failure(tmp_path, monkeypatch, verifier_tree):
    root, logs, _ = verifier_tree
    outside = root / "untrusted-reward"
    outside.write_text("1")
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    (task.path / "tests/test.sh").write_text(
        f"ln -s {shlex.quote(str(outside))} {shlex.quote(str(logs / 'verifier/reward.txt'))}\n"
    )
    outcome = await server._run_verifier(
        SimpleNamespace(task=task, sandbox=LocalSandbox(), workdir=str(root), entrypoint=None), TRAJECTORY
    )
    assert outcome["reward"] == 0 and outcome["mask_sample"] is True
    assert not list(server._artifacts_dir.rglob("reward.txt"))
    await server.shutdown()


@pytest.mark.parametrize("identifier", ["normal", "dots", "long"])
async def test_episode_ids_never_name_verifier_folders(tmp_path, monkeypatch, identifier):
    """Rollout IDs are caller input; verifier folders are reserved under the artifact root regardless of them."""
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    root = tmp_path / "work" / "artifacts"
    server._artifacts_dir = root
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "reward.txt"
    sentinel.write_text("untouched")
    rollout = {"normal": "r1", "dots": "a..b..", "long": "x" * 300}[identifier]
    response = await run_episode(server, task, rollout=rollout)
    assert response.episode_id.rollout_id == rollout
    result = response.result
    assert result.reward == 1 and result.mask_sample is False
    logs = Path(result.verifier_logs_dir).resolve()
    assert logs.parent == root.resolve() and rollout not in logs.name
    assert (logs / "reward.txt").read_text() == "1\n"
    assert sentinel.read_text() == "untouched"
    assert sum("test.sh" in call["command"] and "timeout" in call["command"] for call in sandbox.execs) == 1


async def test_artifact_root_redirected_after_startup_cannot_receive_verifier_files(tmp_path, monkeypatch):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    during_turn(server, lambda: server._artifacts_dir.symlink_to(outside, target_is_directory=True))
    result = (await run_episode(server, task)).result
    assert result.mask_sample is True
    assert result.failure_kind == "persistence_failed"
    assert str(server._artifacts_dir) in result.failure_reason
    assert list(outside.iterdir()) == []
    assert not any("test.sh" in call["command"] for call in sandbox.execs)
    assert sandbox.stopped


async def test_reused_episode_id_does_not_read_old_verifier_output(tmp_path, monkeypatch):
    sandbox = FakeSandbox(verifier_files={"test-stdout.txt": "no reward was written"})
    server, task, _, _ = make_server(tmp_path, monkeypatch, sandbox)
    stale = server._artifacts_dir / "r1"
    stale.mkdir(parents=True)
    (stale / "reward.json").write_text('{"reward": 1}')
    result = (await run_episode(server, task)).result
    assert result.reward == 0 and result.mask_sample is False
    assert result.failure_kind == "harbor:missing_reward"
    assert Path(result.verifier_logs_dir) != stale
    assert (stale / "reward.json").read_text() == '{"reward": 1}'


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
@pytest.mark.parametrize("umask", [0o002, 0o077])
async def test_verifier_logs_preserve_directory_permissions_after_transfers(
    tmp_path, monkeypatch, verifier_tree, umask
):
    root, logs, _ = verifier_tree
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    server._artifacts_dir.mkdir()
    server._artifacts_dir.chmod(0o2775)
    (task.path / "tests/test.sh").write_text(
        f"echo 1 > {shlex.quote(str(logs / 'verifier/reward.txt'))}\necho useful-output\n"
    )
    previous_umask = os.umask(umask)
    try:
        expected = server._artifacts_dir / "ordinary-directory"
        expected.mkdir()
        outcome = await server._run_verifier(
            SimpleNamespace(task=task, sandbox=LocalSandbox(), workdir=str(root), entrypoint=None), TRAJECTORY
        )
    finally:
        os.umask(previous_umask)
        await server.shutdown()
    assert outcome["reward"] == 1 and not outcome.get("mask_sample", False)
    artifacts = Path(outcome["verifier_logs_dir"])
    assert artifacts.parent == server._artifacts_dir
    assert stat.S_IMODE(artifacts.stat().st_mode) == stat.S_IMODE(expected.stat().st_mode)
    assert (artifacts / "reward.txt").read_text() == "1\n"
    assert (artifacts / "test-stdout.txt").read_text() == "useful-output\n"


async def test_artifact_directory_creation_failure_is_a_storage_failure(tmp_path, monkeypatch):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    during_turn(server, lambda: server._artifacts_dir.write_text("a file blocks the output directory"))
    result = (await run_episode(server, task)).result
    assert result.mask_sample is True
    assert result.failure_kind == "persistence_failed"
    assert str(server._artifacts_dir) in result.failure_reason
    assert server._artifacts_dir.read_text() == "a file blocks the output directory"
    assert not any("test.sh" in call["command"] for call in sandbox.execs)


def _run_index(commands: list[str]) -> int:
    """The test.sh run itself (the `timeout` wrapper), not the `chmod +x` before it or a probe for the file."""
    return next(i for i, c in enumerate(commands) if "timeout" in c and "test.sh" in c)


@pytest.mark.skipif(shutil.which("tar") is None, reason="tar is required")
async def test_test_sh_alone_is_made_executable_as_root_and_the_package_modes_stay(
    tmp_path, monkeypatch, verifier_tree
):
    """Harbor's verifier rule: upload the package as it is, `chmod +x` the test script as root because it is a
    `.sh`, run it as the verifier user, chown nothing, widen nothing. A package built under umask 077 (0600
    files) ends up with a 0700 test.sh and every other file as it was, readable by root alone."""
    root, logs, tests = verifier_tree

    class Recording(LocalSandbox):
        def __init__(self):
            self.execs: list[dict] = []

        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
            self.execs.append({"command": command, "user": user})
            return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

    toml = TASK_TOML.replace("[verifier]\n", '[verifier]\nuser = "tester"\n')
    server, task, _, _ = make_server(tmp_path, monkeypatch, toml=toml)
    package = task.path / "tests"
    (package / "test.sh").write_text(f"#!/bin/bash\necho 1 > {shlex.quote(str(logs / 'verifier/reward.txt'))}\n")
    (package / "helper.sh").write_text("echo helper\n")
    (package / "data.txt").write_text("fixture\n")
    for name, mode in (("test.sh", 0o600), ("helper.sh", 0o600), ("data.txt", 0o400)):
        (package / name).chmod(mode)
    # Review (round 14): names the task digest skips are not shipped; a `tests/helper.pyc` would be importable by
    # a test while outside the digest, the manifest check and the server's own check.
    (package / "helper.pyc").write_bytes(b"\x00")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "test.cpython-313.pyc").write_bytes(b"\x00")
    sandbox = Recording()
    previous_umask = os.umask(0o077)
    try:
        outcome = await server._run_verifier(
            SimpleNamespace(task=task, sandbox=sandbox, workdir=str(root), entrypoint=None), TRAJECTORY
        )
    finally:
        os.umask(previous_umask)
        await server.shutdown()

    assert outcome["reward"] == 1 and not outcome.get("mask_sample", False)
    commands = [c["command"] for c in sandbox.execs]
    unpack = next(i for i, c in enumerate(commands) if "tar -xzf" in c)
    chmod = commands.index(f"chmod +x {tests}/test.sh")
    run = _run_index(commands)
    # Upload, then the chmod as its own root exec, then the run as the verifier user with no chmod in it.
    assert unpack < chmod < run
    assert sandbox.execs[chmod]["user"] == "root" and sandbox.execs[run]["user"] == "tester"
    assert commands[run] == verifier_command(120)
    modes = {name: stat.S_IMODE((tests / name).stat().st_mode) for name in ("test.sh", "helper.sh", "data.txt")}
    assert modes == {"test.sh": 0o700, "helper.sh": 0o600, "data.txt": 0o400}
    assert sorted(path.name for path in tests.iterdir()) == ["data.txt", "helper.sh", "test.sh"]


@pytest.mark.parametrize("image_owns_tests", [False, True])
async def test_separate_verifier_chmods_test_sh_as_root_then_runs_it_as_the_verifier_user(
    tmp_path, monkeypatch, image_owns_tests
):
    """The same rule in a separate verifier, with the upload (Harbor's default) and without it (the image owns
    /tests, Harbor's skip_tests_upload): the chmod runs either way, as Harbor runs it either way."""

    class Verifier(FakeSandbox):
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
            if image_owns_tests and command.startswith("if [ -d /tests/test.sh ]"):
                self.execs.append({"command": command, "user": user})
                return SandboxExecResult(stdout="file\n", stderr="", return_code=0)
            return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

    agent = AgentSandbox()
    server, task, _, _ = make_server(tmp_path, monkeypatch, agent)
    (task.path / "task.toml").write_text(SEPARATE_TOML.replace('user = "root"', 'user = "tester"'))
    task = load_task(task.path)
    server.config.tasksets["ds"].tasks["hello"] = task.digest
    verifier = Verifier()

    async def create_verifier(task, *, episode_key=None):
        server._sandboxes[f"{episode_key}/verifier"] = verifier
        return verifier

    monkeypatch.setattr(server, "_create_verifier_sandbox", create_verifier)

    result = (await run_episode(server, task)).result

    assert result is not None and result.reward == 1.0 and result.verifier_mode == "separate"
    commands = [c["command"] for c in verifier.execs]
    uploads = [i for i, c in enumerate(commands) if "tar -xzf" in c and "-C /tests;" in c]
    chmod = commands.index("chmod +x /tests/test.sh")
    run = _run_index(commands)
    assert (uploads == []) is image_owns_tests
    assert all(i < chmod for i in uploads) and chmod < run
    assert verifier.execs[chmod]["user"] == "root" and verifier.execs[run]["user"] == "tester"
    assert commands[run] == verifier_command(300)


@pytest.mark.parametrize("linked", ["none", "tests", "verifier"])
async def test_after_a_turns_verification_the_next_agent_reads_none_of_its_grading_files(
    tmp_path, monkeypatch, verifier_tree, linked
):
    # The sandbox as a verified turn leaves it: the uploaded tests (the hidden ones included), the reward, the
    # server's transcript, and what the agent wrote under /logs/agent itself. An agent may also have replaced a
    # grading root with a link to a folder of its own before the verification.
    root, logs, tests = verifier_tree
    planted = root / "planted"
    planted.mkdir()
    (planted / "keep.txt").write_text("the agent's own")
    if linked == "tests":
        tests.symlink_to(planted, target_is_directory=True)
    else:
        tests.mkdir()
        (tests / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
        (tests / "test.patch").write_text("hidden tests\n")
    (logs / "agent").mkdir(parents=True)
    (logs / "agent" / "trajectory.json").write_bytes(TRAJECTORY)
    (logs / "agent" / "commits.txt").write_text("written by the agent in its turn\n")
    if linked == "verifier":
        (logs / "verifier").symlink_to(planted, target_is_directory=True)
    else:
        (logs / "verifier").mkdir()
        (logs / "verifier" / "reward.txt").write_text("1\n")
    server, _, _, _ = make_server(tmp_path, monkeypatch)

    await server._reset_shared_verifier_dirs(LocalSandbox())

    assert sorted(path.name for path in tests.iterdir()) == []
    assert sorted(path.name for path in (logs / "verifier").iterdir()) == []
    assert sorted(path.name for path in (logs / "agent").iterdir()) == ["commits.txt"]
    assert not tests.is_symlink() and not (logs / "verifier").is_symlink()
    # A linked root is replaced, never followed: the folder it pointed at keeps its contents.
    assert (planted / "keep.txt").read_text() == "the agent's own"
    await server.shutdown()
