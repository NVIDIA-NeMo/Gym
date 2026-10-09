# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The low-concurrency launcher stays byte-identical unless LATENCY_METRICS=1 is set, and refuses a resume that
Gym would mishandle."""

import json
import subprocess
from pathlib import Path

import pytest

from nemo_gym import PARENT_DIR


LAUNCHER = PARENT_DIR / "benchmarks/nemotron_3.5_super/sbatch_external_vllm_low_concurrency.sh"
GOLDEN_DIR = Path(__file__).parent / "data"

# Every value the script reads from the environment is pinned so the output is deterministic.
PINNED_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/tmp",
    "USER": "tester",
    "DRY_RUN": "1",
    "MODEL": "/models/m",
    "MODEL_NAME": "m",
    "CONTAINER": "/containers/c.sqsh",
    "MOUNTS": "/a:/a",
    "SLURM_COMMENT": "c",
    "SBATCH_ACCOUNT": "acct",
    "SBATCH_PARTITION": "p",
    "SBATCH_GRES": "gpu:4",
    "EXPERIMENT_NAME": "exp",
}
SHARDS = f"{PARENT_DIR}/benchmarks/nemotron_3.5_super/split50/shards"
GUARD_ENV = {
    "SHARD_DIR": SHARDS,
    "VLLM_CONFIG": f"{PARENT_DIR}/benchmarks/nemotron_3.5_super/vllm_configs/nemotron_3.5_super_low_concurrency.sh",
}
STORED = {
    "model": "/models/m",
    "model_name": "m",
    "container": "c.sqsh",
    "vllm_config": "nemotron_3.5_super_low_concurrency.sh",
    "concurrency": 48,
    "num_replicas": 2,
    "max_num_seqs": 192,
    "sandbox_timeout_s": 7200,
}


def launch(bench: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(LAUNCHER), bench], env={**PINNED_ENV, **env}, cwd=PARENT_DIR, capture_output=True, text=True
    )


def _section(out: str, name: str) -> str:
    return out.split(f"================= {name} =================")[1].split("=================")[0]


@pytest.mark.parametrize("bench", ["tb", "swe"])
def test_default_dry_run_matches_golden(bench: str) -> None:
    done = launch(bench)
    assert done.returncode == 0, done.stderr
    assert done.stdout == (GOLDEN_DIR / f"launcher_dry_run_{bench}.golden").read_text()
    assert "++resume_from_cache=true" in done.stdout and "slurm_job_id_" not in done.stdout
    assert "metrics/run_info.json" not in done.stdout and "timeline.py" not in done.stdout


def test_latency_flag_adds_run_info_metadata_capture_and_the_post_run_steps() -> None:
    out = launch("swe", LATENCY_METRICS="1").stdout
    eval_0 = _section(out, "eval_command_0")
    batch = out.split("================= batch_command =================")[1]
    assert eval_0.count("metrics/run_info.json") == 2  # existence guard and target
    assert "collector_status=0" in eval_0 and '"${GYM_MODEL_PARAMS[@]}" || collector_status=$?' in eval_0
    assert '"shard_config": "benchmarks/nemotron_3.5_super/split50/shards/swe_r0.yaml"' in eval_0
    assert "++model_call_capture_payloads=metadata" in eval_0
    assert (
        "swebench_verified_opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent.latency_metrics=true"
        in eval_0
    )
    cli = "python benchmarks/nemotron_3.5_super/analysis/timeline.py"
    assert (
        eval_0.index("run_info.json")
        < eval_0.index("gym eval run")
        < eval_0.index(f"{cli} build --run-dir results/$experiment_name")
        < eval_0.index("--tokenize-url http://")
        < eval_0.index(f"{cli} check --run-dir results/$experiment_name")
        < eval_0.index(f"{cli} analyze --run-dir results/$experiment_name")
        < eval_0.index('exit "$collector_status"')
    )
    assert batch.count(f"{cli} report --experiment-dir results/exp'") == 1
    assert batch.index('wait "$pid"') < batch.index("timeline.py report") < batch.index('kill "$watchdog"')
    assert "otlp" not in out and "run_sessions" not in out and "release_before_record" not in out
    tb = launch("tb", LATENCY_METRICS="1").stdout
    assert (
        "latency_metrics=true" not in tb and "++model_call_capture_payloads=metadata" in tb
    )  # Terminus always observes


def _fake_gym(tmp_path: Path, *, run_info: dict | None, gym_files: bool = True) -> Path:
    """A directory that passes the launcher's checkout check, holding an experiment ``exp`` with two replicas."""
    gym = tmp_path / "gym"
    (gym / "nemo_gym").mkdir(parents=True)
    (gym / "benchmarks").mkdir()
    for i in range(2):
        replica = gym / "results" / "exp" / f"replica_{i}"
        (replica / "metrics").mkdir(parents=True)
        if gym_files:
            (replica / "rollouts.jsonl").write_text("{}\n")
            (replica / "rollouts_materialized_inputs.jsonl").write_text("{}\n")
        if run_info is not None:
            (replica / "metrics" / "run_info.json").write_text(
                json.dumps({**run_info, "replica": i, "shard_config": f"{SHARDS}/tb_r{i}.yaml"})
            )
    return gym


def test_resume_guard_accepts_the_same_knobs_and_refuses_different_ones_or_a_bare_replica(tmp_path) -> None:
    gym = _fake_gym(tmp_path, run_info=STORED)
    done = launch("tb", GYM=str(gym), **GUARD_ENV)
    assert done.returncode == 0, done.stderr
    assert "Resuming results/exp" in done.stdout
    roundabout = f"{PARENT_DIR}/benchmarks/../benchmarks/nemotron_3.5_super/split50/shards"
    assert launch("tb", GYM=str(gym), **{**GUARD_ENV, "SHARD_DIR": roundabout}).returncode == 0
    done = launch("tb", GYM=str(gym), CONCURRENCY="32", **GUARD_ENV)
    assert done.returncode == 1 and "resume refused: concurrency differs" in done.stderr
    assert "stored 48" in done.stderr and "now '32'" in done.stderr
    bare = _fake_gym(tmp_path / "b", run_info=None, gym_files=False)
    done = launch("tb", GYM=str(bare), **GUARD_ENV)
    assert done.returncode == 1 and "replica_0/rollouts.jsonl is missing" in done.stderr
    fresh = tmp_path / "f" / "gym"
    (fresh / "nemo_gym").mkdir(parents=True)
    (fresh / "benchmarks").mkdir()
    done = launch("tb", GYM=str(fresh), **GUARD_ENV)
    assert done.returncode == 0 and "Resuming" not in done.stdout
