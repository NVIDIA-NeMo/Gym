# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the submitted commands without allocating a cluster or calling a model."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors.slurm.script import _build_vllm_command, build_sbatch_script


CONFIG_DIR = Path(__file__).resolve().parents[3] / "benchmarks/indic/arena_proxy_v3/submit"


@pytest.fixture
def deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    settings = {
        "SLURM_ACCOUNT": "test-account",
        "SHARED_ROOT": str(tmp_path),
        "RUN_DIR": str(tmp_path / "run"),
        "MODEL_PATH": str(tmp_path / "checkpoint"),
        "MODEL_KEY": "test-model",
        "IMAGE": str(tmp_path / "vllm-0.26.0.sqsh"),
        "GYM_DRIVER_IMAGE": str(tmp_path / "gym.sqsh"),
        "GYM_GIT_REPO": "https://github.com/NVIDIA-NeMo/Gym.git",
        "GYM_GIT_REF": "a" * 40,
        "GENERATION_JOB_ID": "12345",
        "GENERATION_DIR": str(tmp_path / "generation"),
        "JUDGE_KEY_FILE": str(tmp_path / "judge-key"),
    }
    for name in (
        "CONTEXT",
        "TOP_K",
        "TOP_P",
        "TEMPERATURE",
        "MAX_OUTPUT_TOKENS",
        "TP",
        "DP",
        "GPU_UTIL",
        "CONCURRENCY",
        "JUDGE_CONCURRENCY",
        "GPU_PARTITION",
        "CPU_PARTITION",
        "ARENA_JUDGE_BASE_URL",
        "ARENA_JUDGE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    return settings


def _config(name: str) -> SubmitConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        resolved = OmegaConf.to_container(compose(config_name=name), resolve=True)
    return SubmitConfig.model_validate({key: value for key, value in resolved.items() if not key.startswith("_")})


@pytest.mark.parametrize("name", ["generate", "gemma31b", "judge"])
def test_submit_renders_valid_shell_and_the_same_published_install(deployment, tmp_path, name):
    config = _config(name)
    benchmark_name, benchmark = next(iter(config.driver.benchmarks.items()))
    script = build_sbatch_script(config, benchmark_name, benchmark, config.compute["cluster"], tmp_path / "job")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True, capture_output=True)
    assert config.driver.gym_install.ref == deployment["GYM_GIT_REF"]
    assert config.driver.container == deployment["GYM_DRIVER_IMAGE"]
    assert config.driver.env["INPUT_JSONL"] == deployment["RUN_DIR"] + "/inputs/arena_proxy_v3_benchmark.jsonl"
    assert "ARENA_JUDGE_API_KEY" not in config.driver.env
    if name == "judge":
        assert config.services == {}
        assert "#SBATCH --dependency=afterok:12345" in script
        assert "#SBATCH --kill-on-invalid-dep=yes" in script
        assert "#SBATCH --cpus-per-task=4" in script
        assert "--gpus-per-node" not in script
        assert "vllm serve" not in script
    else:
        assert "#SBATCH --gpus-per-node=8" in script
        assert "#SBATCH --exclusive=exclusive\n" in script


@pytest.mark.parametrize(
    ("name", "context", "top_k", "parser", "expert_parallel"),
    [("generate", "163840", "-1", "nemotron_v3", True), ("gemma31b", "131072", "64", "gemma4", False)],
)
def test_reference_service_and_sampling_settings(deployment, name, context, top_k, parser, expert_parallel):
    config = _config(name)
    argv = shlex.split(_build_vllm_command(config.services["policy"]))
    for flag, value in {
        "--served-model-name": "test-model",
        "--host": "127.0.0.1",
        "--tensor-parallel-size": "8",
        "--max-model-len": context,
        "--gpu-memory-utilization": "0.85",
        "--max-num-seqs": "256",
        "--reasoning-parser": parser,
        "--tool-call-parser": "gemma4" if name == "gemma31b" else "qwen3_coder",
    }.items():
        assert argv[argv.index(flag) + 1] == value
    assert json.loads(argv[argv.index("--default-chat-template-kwargs") + 1]) == {"enable_thinking": True}
    assert ("--enable-expert-parallel" in argv) is expert_parallel
    assert "--enable-auto-tool-choice" in argv
    assert "--trust-remote-code" in argv
    assert config.services["policy"].number_of_instances == 1
    assert config.driver.env["CONTEXT"] == context
    assert config.driver.env["TOP_K"] == top_k
    assert config.driver.env["TEMPERATURE"] == "1.0"
    assert config.driver.env["TOP_P"] == "0.95"
    assert config.driver.env["MAX_OUTPUT_TOKENS"] == "120000"


def _run_command(config: SubmitConfig, tmp_path: Path, *, fail_preflight: bool = False):
    """Record child invocations, keeping fake credential values out of the log."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    recorder = (
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
tool = Path(sys.argv[0]).name
with open(os.environ["CALL_LOG"], "a") as stream:
    stream.write(json.dumps({"tool": tool, "args": sys.argv[1:],
        "judge_key_present": bool(os.environ.get("ARENA_JUDGE_API_KEY")),
        "context": os.environ.get("CONTEXT"), "top_k": os.environ.get("TOP_K")}) + "\\n")
sys.exit(31 if os.environ.get("FAIL_PREFLIGHT") and tool == "python" else 0)
"""
    )
    for path in (bin_dir / "python", bin_dir / "gym", tmp_path / "resources_servers/arena_judge/.venv/bin/python"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(recorder)
        path.chmod(0o700)
    benchmark = next(iter(config.driver.benchmarks.values()))
    env = {
        "PATH": f"{bin_dir}:{os.defpath}",
        "SLURM_JOB_ID": "12345",
        "NEMO_GYM_BENCH_DIR": str(tmp_path / "job"),
        "CALL_LOG": str(tmp_path / "calls.jsonl"),
        **config.driver.env,
    }
    for key, name in (
        ("policy_base_url", "NEMO_GYM_POLICY_BASE_URL"),
        ("policy_model_name", "NEMO_GYM_POLICY_MODEL_NAME"),
        ("policy_api_key", "NEMO_GYM_POLICY_API_KEY"),
    ):
        if key in benchmark.run:
            env[name] = str(benchmark.run[key])
    if fail_preflight:
        env["FAIL_PREFLIGHT"] = "1"
    inputs = Path(env["INPUT_JSONL"])
    inputs.parent.mkdir(parents=True, exist_ok=True)
    inputs.write_text("{}\n")
    if config.services == {}:
        for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
            path = Path(env["GENERATION_DIR"]) / "artifacts" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", benchmark.command],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    log = Path(env["CALL_LOG"])
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


def test_generation_runs_preflight_then_collects_saved_answers(deployment, tmp_path):
    result, calls = _run_command(_config("generate"), tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    assert calls[0]["args"][:2] == ["-m", "benchmarks.indic.arena_proxy_v3.generation_preflight"]
    argv = calls[1]["args"]
    assert argv[:2] == ["eval", "run"]
    for flag, value in {
        "--model": "test-model",
        "--temperature": "1.0",
        "--top-p": "0.95",
        "--num-repeats": "1",
        "--max-output-tokens": "120000",
        "--concurrency": "256",
    }.items():
        assert argv[argv.index(flag) + 1] == value
    assert "+skip_verification=true" in argv and "+require_complete=true" in argv
    assert "--resume" in argv
    assert not any(call["judge_key_present"] for call in calls)


def test_failed_preflight_prevents_generation(deployment, tmp_path):
    result, calls = _run_command(_config("generate"), tmp_path, fail_preflight=True)
    assert result.returncode == 31
    assert [call["tool"] for call in calls] == ["python"]


@pytest.mark.parametrize("mode", [0o600, 0o644])
def test_cpu_command_reuses_saved_answers_and_enforces_key_permissions(deployment, tmp_path, mode):
    key = Path(deployment["JUDGE_KEY_FILE"])
    key.write_text("unit-test-judge-credential\n")
    key.chmod(mode)
    config = _config("judge")
    assert "unit-test-judge-credential" not in config.model_dump_json()
    result, calls = _run_command(config, tmp_path)
    assert "unit-test-judge-credential" not in result.stdout + result.stderr
    if mode != 0o600:
        assert result.returncode != 0
        assert not calls
        return
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    argv = calls[0]["args"]
    assert argv[:2] == ["eval", "reverify"]
    assert argv[argv.index("--inputs") + 1].endswith("generation/artifacts/rollouts_materialized_inputs.jsonl")
    assert argv[argv.index("--rollouts") + 1].endswith("generation/artifacts/rollouts.jsonl")
    assert argv[argv.index("--concurrency") + 1] == "12"
    assert "--resume" in argv
    assert calls[0]["judge_key_present"]
    assert not calls[1]["judge_key_present"]
    assert calls[1]["args"][:2] == ["-m", "benchmarks.indic.arena_proxy_v3.summarize"]
