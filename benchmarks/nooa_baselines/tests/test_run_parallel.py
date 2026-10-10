# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path
from threading import Event, Lock

import pytest

from benchmarks.nooa_baselines import run_parallel


@pytest.fixture
def launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "source-manifest.json").write_text("{}\n")
    monkeypatch.setenv("SLURM_JOB_ID", "test-123")
    monkeypatch.setattr(
        sys, "argv", ["run_parallel", "--run-root", str(tmp_path), "--swe-model-dir", str(tmp_path / "slow-model")]
    )
    return tmp_path


def outcome(root: Path) -> dict:
    return json.loads(next(root.glob("controller-*/progress.json")).read_text())["benchmarks"]


def test_canary_failure_is_isolated_and_ready_replica_advances_without_other_model(
    launcher: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tb_full = Event()
    lock = Lock()
    phases: dict[str, list[str]] = {name: [] for name in ("swe", "tb", "gdp")}

    def wait_for_model(model_dir: Path, *, timeout_seconds: int) -> None:
        if model_dir.name == "slow-model":
            # A global canary/full barrier would deadlock this independent model wait.
            assert tb_full.wait(5), "TB full phase did not progress while SWE's model was unavailable"

    def logged(command: list[str], *, path: Path, env: dict[str, str], cwd: Path) -> None:
        if "--phase" not in command:
            return
        benchmark = command[command.index("--benchmark") + 1]
        phase = command[command.index("--phase") + 1]
        with lock:
            phases[benchmark].append(phase)
        if benchmark == "swe" and phase == "canary":
            raise subprocess.CalledProcessError(1, command)
        if benchmark == "tb" and phase == "full":
            tb_full.set()

    monkeypatch.setattr(run_parallel, "wait_for_model", wait_for_model)
    monkeypatch.setattr(run_parallel, "run_logged", logged)
    with pytest.raises(SystemExit) as stopped:
        run_parallel.main()
    assert stopped.value.code == 1
    assert phases == {"swe": ["canary"], "tb": ["canary", "full"], "gdp": ["canary", "full"]}
    result = outcome(launcher)
    assert result["swe"]["completed"] is False
    assert result["swe"]["error_type"] == "CalledProcessError"
    assert result["tb"] == result["gdp"] == {"completed": True}


def test_gdp_canary_provider_failure_does_not_block_peers(launcher: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def logged(command: list[str], *, path: Path, env: dict[str, str], cwd: Path) -> None:
        seen.append(path.name)
        if path.name == "gdp-canary.log":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(run_parallel, "wait_for_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_parallel, "run_logged", logged)
    with pytest.raises(SystemExit) as stopped:
        run_parallel.main()
    assert stopped.value.code == 1
    assert set(seen) == {"gdp-canary.log", "swe-canary.log", "swe-full.log", "tb-canary.log", "tb-full.log"}
    assert outcome(launcher)["gdp"]["error_type"] == "CalledProcessError"
    assert all(outcome(launcher)[name] == {"completed": True} for name in ("swe", "tb"))


def test_explicit_replica_overlay_and_concurrency_forwarded_without_slurm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_parallel",
            "--run-root",
            str(tmp_path),
            "--benchmarks",
            "gdp",
            "--model-dir",
            str(tmp_path / "endpoint"),
            "--gdp-config",
            str(tmp_path / "provider.yaml"),
            "--gdp-concurrency",
            "32",
        ],
    )
    models, commands = [], []
    monkeypatch.setattr(run_parallel, "wait_for_model", lambda path, **kwargs: models.append(path))
    monkeypatch.setattr(run_parallel, "run_logged", lambda command, **kwargs: commands.append(command))
    with pytest.raises(SystemExit) as stopped:
        run_parallel.main()
    assert stopped.value.code == 0 and models == [tmp_path / "endpoint"]
    assert len(commands) == 2
    assert "--concurrency" not in commands[0]
    assert commands[1][-2:] == ["--concurrency", "32"]
    assert all(command[command.index("--config") + 1] == str(tmp_path / "provider.yaml") for command in commands)


def test_existing_component_log_is_preserved(tmp_path: Path) -> None:
    log = tmp_path / "previous.log"
    log.write_text("existing attempt evidence\n")
    with pytest.raises(FileExistsError):
        run_parallel.run_logged(["must-not-start"], path=log, env={}, cwd=tmp_path)
    assert log.read_text() == "existing attempt evidence\n"
