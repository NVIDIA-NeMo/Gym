# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors.script_templates import render_export_tail
from nemo_gym.orchestration.executors.slurm.script import build_sbatch_script, reverify_steps


COMPUTE = {"cluster": {"type": "slurm", "account": "acct", "hostname": "host"}}
KUBERNETES = {
    "k8s": {"type": "kubernetes", "namespace": "ns", "gpus_per_node": 8, "context": "ctx", "pvc_name": "pvc"}
}
JOB_ID = "gym-job-20261114T101500Z-abc123"
BENCH_DIR = Path("/jobs") / JOB_ID / "tb"
JUDGE = {"config": ["resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml"]}


def _submit(benchmark: dict[str, Any], *, compute: dict | None = None, exports: dict | None = None) -> SubmitConfig:
    return SubmitConfig.model_validate(
        {
            "services": {"svc": {"container": "gym:latest", "type": "vllm", "model": "org/model"}},
            "compute": compute or COMPUTE,
            "driver": {"container": "gym:latest", "benchmarks": {"tb": benchmark}, "gym_install": {"ref": "abc"}},
            "job": {"output_path": "/jobs", **({"export": exports} if exports else {})},
        }
    )


def _script(config: SubmitConfig) -> str:
    compute = next(iter(config.compute.values()))
    return build_sbatch_script(config, "tb", config.driver.benchmarks["tb"], compute, BENCH_DIR)


class TestReverifyConfig:
    def test_a_step_needs_a_config(self) -> None:
        with pytest.raises(ValidationError):
            _submit({"reverify": {"judge": {"config": []}}})

    def test_step_names_are_snake_case(self) -> None:
        with pytest.raises(ValidationError, match="lowercase snake_case"):
            _submit({"reverify": {"Second-Pass": JUDGE}})

    def test_a_step_cannot_overwrite_the_materialized_inputs(self) -> None:
        with pytest.raises(ValidationError, match="would overwrite the run"):
            _submit({"reverify": {"materialized_inputs": JUDGE}})

    def test_a_command_benchmark_cannot_reverify(self) -> None:
        with pytest.raises(ValidationError, match="both `command` and `reverify`"):
            _submit({"command": "echo hi", "reverify": {"second_pass": JUDGE}})

    def test_kubernetes_compute_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sets `reverify`, which is not supported on Kubernetes"):
            _submit({"reverify": {"second_pass": JUDGE}}, compute=KUBERNETES)


class TestReverifyCommand:
    def test_the_step_reads_the_runs_rollouts_and_writes_its_own_file(self) -> None:
        config = _submit({"reverify": {"second_pass": {**JUDGE, "overrides": {"concurrency_hint": 4}}}})

        [(name, output, command)] = reverify_steps(config.driver.benchmarks["tb"], BENCH_DIR)

        assert name == "second_pass"
        assert output == f"{BENCH_DIR}/artifacts/rollouts_second_pass.jsonl"
        assert command.startswith(f"NEMO_GYM_REVERIFY_SOURCE_ROLLOUTS={BENCH_DIR}/artifacts/rollouts.jsonl gym eval")
        for expected in (
            f"--config {JUDGE['config'][0]}",
            f"--inputs {BENCH_DIR}/artifacts/rollouts_materialized_inputs.jsonl",
            f"--rollouts {BENCH_DIR}/artifacts/rollouts.jsonl",
            f"--output {output}",
            "+concurrency_hint=4",
        ):
            assert expected in command

    def test_the_job_script_runs_the_step_after_the_eval_and_before_the_exports(self) -> None:
        script = _script(_submit({"reverify": {"second_pass": JUDGE}}, exports={"atif": {"agent_version": "test"}}))

        assert 'exec "$@"' not in script
        assert "gym eval reverify" in script
        assert script.index("gym eval reverify") < script.index("nemo_gym.orchestration.export_step atif")

    def test_a_step_alone_still_gets_the_tail_and_the_success_marker(self) -> None:
        script = _script(_submit({"reverify": {"second_pass": JUDGE}}))

        assert 'exec "$@"' not in script
        assert "GYM_EVAL_SUCCEEDED_MARKER" in script


# A real bash run of the tail, with a `gym` and a `python3` that record what they are asked to do.
FAKE_TOOL = """#!/bin/bash
echo "$(basename "$0") $*" >> "$FAKE_GYM_LOG"
if [ "$(basename "$0")" = gym ] && [ -n "${FAKE_REVERIFY_RC:-}" ]; then exit "$FAKE_REVERIFY_RC"; fi
"""


@pytest.fixture
def tail_env(tmp_path: Path) -> dict[str, Any]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("gym", "python3"):
        (bin_dir / tool).write_text(FAKE_TOOL)
        (bin_dir / tool).chmod(0o755)
    run = tmp_path / "run.sh"
    run.write_text('#!/bin/bash\necho run >> "$FAKE_GYM_LOG"\nexit "${FAKE_RUN_RC:-0}"\n')
    run.chmod(0o755)
    bench = tmp_path / "bench"
    (bench / "artifacts").mkdir(parents=True)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_GYM_LOG": str(tmp_path / "calls.log")}
    return {"bench": bench, "run": run, "env": env, "log": tmp_path / "calls.log"}


def _run_tail(tail_env: dict[str, Any], **env: str) -> list[str]:
    output = str(tail_env["bench"] / "artifacts" / "rollouts_judge.jsonl")
    tail = render_export_tail(
        tail_env["bench"], [("atif", ["+label=x"])], [("judge", output, f"gym eval reverify --output {output}")]
    )
    completed = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{tail}", "--", str(tail_env["run"])],
        env={**tail_env["env"], **env},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    calls = tail_env["log"].read_text().splitlines() if tail_env["log"].exists() else []
    return [str(completed.returncode), *calls]


class TestReverifyTail:
    def test_a_successful_run_is_reverified_then_exported(self, tail_env: dict[str, Any]) -> None:
        rc, *calls = _run_tail(tail_env)

        assert rc == "0"
        output = tail_env["bench"] / "artifacts" / "rollouts_judge.jsonl"
        assert calls == [
            "run",
            f"gym eval reverify --output {output}",
            "python3 -m nemo_gym.orchestration.export_step atif +label=x",
        ]

    def test_an_existing_output_is_resumed(self, tail_env: dict[str, Any]) -> None:
        (tail_env["bench"] / "artifacts" / "rollouts_judge.jsonl").write_text("{}\n")

        _, _, reverify, _ = _run_tail(tail_env)

        assert reverify.endswith("--resume")

    def test_a_failed_step_keeps_the_exit_code_and_the_exports(self, tail_env: dict[str, Any]) -> None:
        rc, *calls = _run_tail(tail_env, FAKE_REVERIFY_RC="2")

        assert rc == "0"
        assert calls[-1].startswith("python3 -m nemo_gym.orchestration.export_step atif")

    def test_a_failed_run_is_not_reverified(self, tail_env: dict[str, Any]) -> None:
        rc, *calls = _run_tail(tail_env, FAKE_RUN_RC="3")

        assert rc == "3"
        assert calls == ["run"]
