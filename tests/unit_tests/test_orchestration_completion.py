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
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.completion import main, validate_completion
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.path_utils import failures_path_for, materialized_inputs_path_for


def _row(task=0, repeat=0, **extra):
    return {"_ng_task_index": task, "_ng_rollout_index": repeat, **extra}


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize(
    ("results", "completed"),
    [
        ([_row(reward=0), _row(1, reward=1)], 2),
        ([_row(), _row(), _row(99)], 1),
        ([_row(), _row(1, _ng_failure_class="agent_run_error", reward=0)], 1),
        ([_row(), _row(1, _ng_no_persist=True)], 1),
        ([], 0),
    ],
    ids=["complete-with-zero-reward", "duplicates-and-unrelated", "failed", "not-persisted", "empty"],
)
def test_coverage_uses_actual_distinct_results(tmp_path, capsys, results, completed):
    output = tmp_path / "rollouts.jsonl"
    _write_rows(materialized_inputs_path_for(output), [_row(), _row(1)])
    _write_rows(output, results)
    # Even a scored, terminal failure in the sidecar cannot replace a result.
    _write_rows(failures_path_for(output), [_row(1, reward=0, _ng_failure_terminal=True)])
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}

    assert main([str(output)]) == (0 if completed == 2 else 1)

    captured = capsys.readouterr()
    if completed == 2:
        assert "EVAL SUCCEEDED: 2/2" in captured.out
    else:
        assert f"EVAL FAILED: {completed}/2 samples completed; {2 - completed} missing" in captured.err
        assert str(output) in captured.err
        assert str(materialized_inputs_path_for(output)) in captured.err
        assert "Partial artifacts retained" in captured.err
    assert before == {path: path.read_bytes() for path in tmp_path.iterdir()}


@pytest.mark.parametrize(
    ("artifact", "content"),
    [("inputs", None), ("results", None), ("results", "{invalid"), ("inputs", "{}\n")],
    ids=["missing-inputs", "missing-results", "invalid-json", "missing-identity"],
)
def test_invalid_artifacts_fail_with_location(tmp_path, capsys, artifact, content):
    output = tmp_path / "rollouts.jsonl"
    inputs = materialized_inputs_path_for(output)
    _write_rows(inputs, [_row()])
    _write_rows(output, [_row()])
    invalid = inputs if artifact == "inputs" else output
    if content is None:
        invalid.unlink()
    else:
        invalid.write_text(content)
    assert main([str(output)]) == 1
    assert str(invalid) in capsys.readouterr().err


def test_duplicate_expected_identity_is_invalid(tmp_path):
    output = tmp_path / "rollouts.jsonl"
    _write_rows(materialized_inputs_path_for(output), [_row(), _row()])
    _write_rows(output, [_row()])
    with pytest.raises(ValueError, match="Duplicate expected sample identity"):
        validate_completion(output)


def test_valid_empty_evaluation(tmp_path):
    output = tmp_path / "rollouts.jsonl"
    _write_rows(materialized_inputs_path_for(output), [])
    _write_rows(output, [])
    assert validate_completion(output) == 0


@pytest.mark.parametrize(
    ("completed", "run_exit", "prepare_exit", "install", "otel_enabled", "expected_exit"),
    [
        (2, 0, 0, False, False, 0),
        (1, 0, 0, False, False, 1),
        (1, 23, 0, False, True, 23),
        (2, 0, 27, False, False, 27),
        (2, 0, 0, True, True, 0),
        (1, 0, 0, True, True, 1),
    ],
    ids=["complete", "partial", "run-error", "prepare-error", "install-complete", "install-partial"],
)
def test_generated_batch_script_exit_and_artifacts(
    tmp_path, completed, run_exit, prepare_exit, install, otel_enabled, expected_exit
):
    """Execute the real generated shell and validator, stubbing only external commands."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "logs").mkdir()
    (bin_dir / "python").symlink_to(sys.executable)

    def executable(name, body):
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n" + dedent(body))
        path.chmod(0o755)

    executable(
        "srun",
        """
        import subprocess, sys
        args = sys.argv[1:]
        log = None
        while args[0].startswith('--'):
            option = args.pop(0)
            if option.startswith('--output='):
                log = option.split('=', 1)[1]
        with open(log, 'w') as handle:
            if args[0] != 'bash':
                sys.exit(0)  # Model and collector services are local stubs.
            result = subprocess.run(args, stdout=handle, stderr=handle)
        sys.exit(result.returncode)
        """,
    )
    executable(
        "gym",
        """
        import json, os, sys
        from pathlib import Path
        from nemo_gym.path_utils import materialized_inputs_path_for
        if sys.argv[1:3] == ['eval', 'prepare']:
            sys.exit(int(os.environ['TEST_PREPARE_EXIT']))
        assert sys.argv[1:3] == ['eval', 'run']
        paths = [a.split('=', 1)[1] for a in sys.argv[3:] if a.startswith('+output_jsonl_fpath=')]
        assert len(paths) == 1, paths
        output = Path(paths[0])
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = [{'_ng_task_index': i, '_ng_rollout_index': 0, 'reward': 0} for i in range(2)]
        materialized_inputs_path_for(output).write_text(''.join(json.dumps(r) + '\\n' for r in rows))
        output.write_text(''.join(json.dumps(r) + '\\n' for r in rows[:int(os.environ['TEST_COMPLETED'])]))
        output.with_suffix('.metrics.json').write_text('{"partial_metrics": true}')
        sys.exit(int(os.environ['TEST_RUN_EXIT']))
        """,
    )
    # The installation variant also stays local: no actual download, clone or install.
    executable("curl", "pass\n")
    executable("uv", "pass\n")
    executable("git", "import sys\nfrom pathlib import Path\nif sys.argv[1] == 'clone': Path(sys.argv[-1]).mkdir()\n")
    executable(
        "mktemp",
        "import os\nfrom pathlib import Path\npath = Path(os.environ['TEST_INSTALL_DIR'])\npath.mkdir()\nprint(path)\n",
    )
    executable("sleep", "pass\n")
    executable("pkill", "import os\nfrom pathlib import Path\nPath(os.environ['TEST_CLEANUP']).touch()\n")

    run = {
        "disable_aggregation": True,
        "disable_health_check": True,
        "count_failure_classes_as_zero": ["agent_run_error"],
    }
    if install:
        run["output_jsonl_fpath"] = str(tmp_path / "custom dir/it's rollouts.jsonl")
    config = SubmitConfig.model_validate(
        {
            "services": {"policy": {"type": "vllm", "container": "model:latest", "model": "test"}}
            if otel_enabled
            else {},
            "compute": {"cluster": {"type": "slurm", "account": "test"}},
            "driver": {
                "benchmarks": {"test": {"run": run, "prepare": {"split": "benchmark"} if prepare_exit else {}}},
                **({"gym_install": {"ref": "main"}} if install else {}),
                **({"policy_model": "policy"} if otel_enabled else {}),
            },
            "job": {"output_path": str(tmp_path)},
            "otel": {"endpoint": "https://example.test", "service_name": "test", "token_env": "TEST_OTEL_TOKEN"},
        }
    )
    script = build_sbatch_script(config, "test", config.driver.benchmarks["test"], config.compute["cluster"], tmp_path)
    # Avoid sourcing a real user's uv environment in the stub installation path.
    script = "source() { :; }\nexport -f source\nkill() { return 1; }\n" + script
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "TEST_COMPLETED": str(completed),
        "TEST_RUN_EXIT": str(run_exit),
        "TEST_PREPARE_EXIT": str(prepare_exit),
        "TEST_INSTALL_DIR": str(tmp_path / "install"),
        "TEST_CLEANUP": str(tmp_path / "collector-stopped"),
        "TEST_OTEL_TOKEN": "test-token",
        "SLURM_JOB_ID": "123",
    }
    result = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == expected_exit, result.stderr
    assert (tmp_path / "collector-stopped").exists() == otel_enabled
    log = (tmp_path / "logs/driver.log").read_text()
    if expected_exit:
        assert "EVAL FAILED: benchmark test. See logs/driver.log" in result.stderr
    if not run_exit and not prepare_exit:
        assert f"{completed}/2 samples completed" in log
        assert ("EVAL FAILED:" if expected_exit else "EVAL SUCCEEDED:") in log
    else:
        assert "samples completed" not in log  # The validator must not mask an earlier error.
    if not prepare_exit:
        output = tmp_path / run.get("output_jsonl_fpath", "artifacts/rollouts.jsonl")
        assert len(output.read_text().splitlines()) == completed
        assert output.with_suffix(".metrics.json").read_text() == '{"partial_metrics": true}'
