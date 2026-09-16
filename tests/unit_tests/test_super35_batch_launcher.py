# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real launch scripts without Slurm jobs, installs, or model calls."""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollout_collection import RolloutCollectionConfig


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = Path("benchmarks/nemotron_3.5_super")
SUBMIT = BENCHMARK / "submit_batch.sh"
SERVING = BENCHMARK / "vllm_configs/batched.sh"


def run_shell(script: str, root: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run only local shell logic with a bounded lifetime."""
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def read_args(path: Path) -> list[str]:
    """Read argument boundaries recorded by the command stubs."""
    return path.read_bytes().decode().split("\0")[:-1]


@pytest.fixture
def checkout(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Copy just the published entrypoint's files, with fake model and Slurm binaries."""
    for path in (
        SUBMIT,
        SERVING,
        BENCHMARK / "sbatch_external_vllm.sh",
        BENCHMARK / "batch_configs/core.yaml",
        BENCHMARK / "batch_configs/swe.yaml",
    ):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config.json", "chat_template.jinja", "ultra_v3_reasoning_parser.py"):
        (checkpoint / name).touch()
    (tmp_path / "container.sqsh").touch()
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stubs = {
        "preflight": (
            '#!/bin/bash\nprintf "%s\\0" "$@" > "$CAPTURE/preflight-args"\n'
            'cat > "$CAPTURE/preflight.py"\nexit "${PREFLIGHT_STATUS:-0}"\n'
        ),
        "sbatch": (
            "#!/bin/bash\nset -euo pipefail\n"
            'if [[ "$*" != *--dependency=* ]]; then\n'
            '  printf "%s\\0" "$@" > "$CAPTURE/sbatch-args"\n'
            '  printf "%s" "$eval_command" > "$CAPTURE/eval-command"\n'
            '  printf "%s" "$GYM_BATCH_ARGS" > "$CAPTURE/gym-batch-args"\n'
            '  printf "%s\\0" "$SBATCH_ACCOUNT" "$SBATCH_PARTITION" "$MODEL" "$MODEL_NAME" '
            '"$MOUNTS" "$VLLM_CONFIG" "${ROLLOUTS_FPATH:-}" > "$CAPTURE/launch-env"\n'
            "fi\nprintf '12345\\n'\n"
        ),
    }
    for name, contents in stubs.items():
        stub = stub_dir / name
        stub.write_text(contents)
        stub.chmod(0o755)
    # Never inherit real credentials, Slurm settings, or exported shell functions.
    env = {
        "PATH": f"{stub_dir}:{os.environ['PATH']}",
        "USER": "batch-test",
        "MODEL": str(checkpoint),
        "MODEL_NAME": "test-model",
        "CONTAINER": str(tmp_path / "container.sqsh"),
        "SBATCH_ACCOUNT": "test-account",
        "SBATCH_PARTITION": "batch_long",
        "SBATCH_TIME": "20:00:00",
        "EXPERIMENT_NAME": "batch-test",
        "GYM_PYTHON": str(stub_dir / "preflight"),
        "CAPTURE": str(tmp_path),
    }
    return tmp_path, env


EVAL_STUBS = r"""
set -euo pipefail
source() {
    if [[ "$1" != /opt/Gym_venv/bin/activate ]]; then builtin source "$@"; fi
}
cd() { if [[ "$1" != /opt/Gym ]]; then builtin cd "$@"; fi; }
uv() { echo UV_SYNC; return "${UV_STATUS:-0}"; }
python() { echo DRIVER_CHECK; return "${DRIVER_STATUS:-0}"; }
getent() { printf '127.0.0.1 router\n'; }
gym() {
    printf '%s\0' "$@" > "$CAPTURE/$2-args"
    if [[ "$1 $2" == 'env prefetch' ]]; then return "${PREFETCH_STATUS:-0}"; fi
}
"""


def run_evaluation(root: Path, env: dict[str, str], **settings: str) -> subprocess.CompletedProcess:
    """Execute the generated evaluation command with its external effects stubbed."""
    return run_shell(
        EVAL_STUBS + (root / "eval-command").read_text(),
        root,
        {
            **env,
            "ROUTER_NODE": "router",
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_USER": "batch-test",
            "GYM_BATCH_ARGS": (root / "gym-batch-args").read_text(),
            **settings,
        },
    )


@pytest.mark.parametrize("batch", ["core", "swe"])
def test_submit_forwards_settings_to_prefetch_prepare_and_run(checkout, batch: str) -> None:
    """The full path uses caller settings and preserves Hydra argument boundaries."""
    root, env = checkout
    env.update(ROLLOUTS_FPATH="results/existing.jsonl", MOUNTS="/shared/data:/shared/data:ro")
    config = root / "custom.yaml"
    config.touch()
    overrides = [
        "++resume_from_cache=true",
        "++num_samples_in_parallel_by_agent={one:2,two:3}",
        '++description="literal $(touch INJECTED); spaces and braces {a,b}"',
    ]
    args = ["--config", str(BENCHMARK / f"batch_configs/{batch}.yaml"), "--config", str(config), *overrides]
    # Invoke from outside the checkout; its location, not the caller's cwd, determines the root.
    command = shlex.join(["bash", str(root / SUBMIT), batch, "--config", str(config), *overrides])
    result = run_shell(command, root.parent, env)
    assert result.returncode == 0, result.stderr
    assert read_args(root / "preflight-args") == ["-", *args]
    assert "--time=20:00:00" in read_args(root / "sbatch-args")
    assert "--nodes=4" in read_args(root / "sbatch-args")
    account, partition, model, name, mounts, serving, output = read_args(root / "launch-env")
    assert (account, partition, model, name) == ("test-account", "batch_long", env["MODEL"], "test-model")
    assert mounts == (
        f"{root}:{root},{root}:/opt/Gym,{model}:{model}:ro,"
        f"{root}/results/batch-test/uv_venvs:/opt/uv_venvs,/shared/data:/shared/data:ro"
    )
    assert serving == str(root / SERVING)
    assert output == "results/existing.jsonl"
    result = run_evaluation(root, env)
    assert result.returncode == 0, result.stderr
    for name, prefix in (
        ("prefetch", ["env", "prefetch"]),
        ("prepare", ["eval", "prepare"]),
        ("run", ["eval", "run"]),
    ):
        captured = read_args(root / f"{name}-args")
        assert captured[: 2 + len(args)] == [*prefix, *args]
    assert "++output_jsonl_fpath=results/existing.jsonl" in read_args(root / "run-args")
    assert "++skip_venv_if_present=false" in read_args(root / "prefetch-args")
    assert not (root / "INJECTED").exists()


@pytest.mark.parametrize("check_only", [False, True])
def test_defaults_do_not_enable_resume_or_select_a_stable_output(checkout, check_only: bool) -> None:
    """A name alone never opts into cached results; --check never submits or creates a run."""
    root, env = checkout
    env.pop("SBATCH_TIME")
    args = ["bash", str(SUBMIT), "swe", *(["--check"] if check_only else [])]
    result = run_shell(shlex.join(args), root, env)
    assert result.returncode == 0, result.stderr
    if check_only:
        assert not (root / "sbatch-args").exists()
        assert not (root / "results").exists()
    else:
        assert "--time=04:00:00" in read_args(root / "sbatch-args")
        result = run_evaluation(root, env)
        assert result.returncode == 0, result.stderr
        args = read_args(root / "run-args")
        assert not any(arg.startswith("++resume_from_cache=") for arg in args)
        assert any(arg.startswith("++output_jsonl_fpath=results/batch-test/slurm_job_id_12345/date_") for arg in args)


@pytest.mark.parametrize("missing", ["MODEL", "CONTAINER", "SBATCH_ACCOUNT", "SBATCH_PARTITION"])
def test_missing_required_setting_fails_before_submission(checkout, missing: str) -> None:
    """No checkpoint, image, account, or partition is silently inherited from the pilot."""
    root, env = checkout
    env.pop(missing)
    result = run_shell(f"bash {SUBMIT} swe", root, env)
    assert result.returncode != 0
    assert missing in result.stderr
    assert not (root / "sbatch-args").exists()


@pytest.mark.parametrize(
    "settings",
    [{"EXPERIMENT_NAME": "../escape"}, {"NUM_PREFILL_NODES": "0"}, {"MODEL_NAME": "model;touch INJECTED"}],
)
def test_invalid_launch_settings_fail_before_submission(checkout, settings: dict[str, str]) -> None:
    """Reject values that cannot safely enter the generated shell or result path."""
    root, env = checkout
    result = run_shell(f"bash {SUBMIT} swe", root, {**env, **settings})
    assert result.returncode != 0
    assert not (root / "preflight-args").exists()
    assert not (root / "sbatch-args").exists()
    assert not (root / "INJECTED").exists()


def test_preflight_failure_does_not_submit_or_create_run(checkout) -> None:
    """Invalid Gym configuration must fail before any GPU allocation is requested."""
    root, env = checkout
    result = run_shell(f"bash {SUBMIT} core", root, {**env, "PREFLIGHT_STATUS": "42"})
    assert result.returncode == 42
    assert not (root / "sbatch-args").exists()
    assert not (root / "results").exists()


@pytest.mark.parametrize("failure", ["UV_STATUS", "DRIVER_STATUS", "PREFETCH_STATUS"])
def test_dependency_failure_prevents_evaluation(checkout, failure: str) -> None:
    """Never evaluate using partially installed parent or server environments."""
    root, env = checkout
    result = run_shell(f"bash {SUBMIT} swe", root, env)
    assert result.returncode == 0, result.stderr
    result = run_evaluation(root, env, **{failure: "43"})
    assert result.returncode == 43
    assert not (root / "prepare-args").exists()
    assert not (root / "run-args").exists()


def test_serving_workers_do_not_install_gym_dependencies(checkout) -> None:
    """Both worker pools retain the pilot's four-GPU topology and untouched Python environment."""
    root, env = checkout
    result = run_shell(
        EVAL_STUBS
        + f'\nbuiltin source {SERVING}\nprintf "%s\\n" "${{VLLM_PREFILL_ARGS[*]}}" "${{VLLM_DECODE_ARGS[*]}}"',
        root,
        env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("--tensor-parallel-size 4") == 2
    assert "UV_SYNC" not in result.stdout
    assert "DRIVER_CHECK" not in result.stdout
    assert not (root / "prefetch-args").exists()


@pytest.mark.parametrize("batch", ["core", "swe"])
@pytest.mark.parametrize("valid", [True, False])
def test_check_runs_real_preflight_without_submitting(checkout, batch: str, valid: bool) -> None:
    """The actual embedded Python validates config and credentials without the user's env.yaml."""
    root, env = checkout
    # An explicit file prevents falling back to credentials in the installed checkout.
    (root / "env.yaml").write_text("{}\n")
    env.update(
        GYM_PYTHON=sys.executable,
        PYTHONPATH=str(ROOT),
        OPENSANDBOX_DOMAIN="sandbox.example",
        OPENSANDBOX_API_KEY="fixture-key" if valid else "dummy",
        NV_INFERENCE_API_KEY="fixture-key",
    )
    result = run_shell(f"bash {SUBMIT} {batch} --check", root, env)
    assert (result.returncode == 0) is valid, result.stderr
    if valid:
        assert "PASS: Gym configuration resolves" in result.stdout
    else:
        assert "Missing connection setting: sandbox.opensandbox.connection.api_key" in result.stderr
    assert "fixture-key" not in result.stdout + result.stderr
    assert not (root / "sbatch-args").exists()
    assert not (root / "results").exists()


@pytest.mark.parametrize("batch, expected_members", [("core", 11), ("swe", 2)])
def test_recipes_resolve_without_local_pilot_files_or_credentials(
    monkeypatch, batch: str, expected_members: int
) -> None:
    """Resolve the real configs offline with synthetic credentials, preserving repeat policy."""
    monkeypatch.chdir(ROOT)
    parser = GlobalConfigDictParser()
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {
            "config_paths": [
                str(BENCHMARK / f"batch_configs/{batch}.yaml"),
                str(BENCHMARK / "sandbox_utils.yaml"),
                str(BENCHMARK / "policy_model_override.yaml"),
            ],
            "nv_inference_api_key": "fixture-key",
            "sandbox": {"opensandbox": {"connection": {"domain": "sandbox.example", "api_key": "fixture-key"}}},
        },
    )
    config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial, skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True
        )
    )
    rollout = RolloutCollectionConfig.model_validate(
        {**OmegaConf.to_container(config, resolve=True), "input_jsonl_fpath": "unused", "output_jsonl_fpath": "unused"}
    )
    members = [
        server
        for server in parser.filter_for_server_instance_configs(config)
        if any(dataset.type == "benchmark" for dataset in server.datasets or [])
    ]
    assert len(members) == expected_members
    assert not any("swebench_pro" in server.name for server in members)
    assert config.model_endpoint_readiness_timeout_seconds == 1800
    sampling = config.policy_model.responses_api_models.vllm_model.sampling_overrides
    assert sampling.temperature == 1.0 and sampling.top_p == 0.95
    if batch == "swe":
        assert {server.name for server in members} == {
            "swebench_verified_opencode_sandboxed_agent",
            "swebench_multilingual_opencode_sandboxed_agent",
        }
        assert rollout.num_repeats == 1 and rollout.num_repeats_add_seed is False
        assert all(dataset.num_repeats == 3 for server in members for dataset in server.datasets)
    else:
        assert rollout.num_repeats == {
            "apex_shortlist_math_with_judge_simple_agent": 16,
            "lmarena_v2_benchmark_agent": 3,
            "_default": 1,
        }
        assert rollout.num_repeats_add_seed == {
            "apex_shortlist_math_with_judge_simple_agent": True,
            "lmarena_v2_benchmark_agent": True,
            "_default": False,
        }
        assert (
            config.tau2_benchmark_agent.responses_api_agents.tau2.user_model_server.name
            == "Qwen3-235B-A22B-Instruct-2507-FP8"
        )
