# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real launch scripts without Slurm jobs, installs, or model calls."""

import os
import runpy
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
BUILDER = BENCHMARK / "build_eval_container.sh"


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
        BUILDER,
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
uv() { echo UNEXPECTED_INSTALL; return 99; }
python() {
    printf '%s\0' "$@" > "$CAPTURE/dependency-check-args"
    cat > "$CAPTURE/dependency-check.py"
    return "${DRIVER_STATUS:-0}"
}
getent() { printf '127.0.0.1 router\n'; }
gym() {
    printf '%s\0' "$@" > "$CAPTURE/$2-args"
    if [[ "$1 $2" == 'env prefetch' ]]; then return 99; fi
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
def test_submit_forwards_settings_to_dependency_check_prepare_and_run(checkout, batch: str) -> None:
    """The full path uses caller settings and preserves Hydra argument boundaries."""
    root, env = checkout
    env.update(ROLLOUTS_FPATH="results/existing.jsonl", MOUNTS="/shared/data:/shared/data:ro")
    config = root / "custom.yaml"
    config.touch()
    overrides = [
        "++resume_from_cache=true",
        "++num_repeats_add_seed={_default:false,one:true}",
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
    assert mounts == f"{root}:{root},{root}:/opt/Gym,{model}:{model}:ro,/shared/data:/shared/data:ro"
    assert not (root / "results/batch-test/uv_venvs").exists()
    assert serving == str(root / SERVING)
    assert output == "results/existing.jsonl"
    result = run_evaluation(root, env)
    assert result.returncode == 0, result.stderr
    for name, prefix in (
        ("prepare", ["eval", "prepare"]),
        ("run", ["eval", "run"]),
    ):
        captured = read_args(root / f"{name}-args")
        assert captured[: 2 + len(args)] == [*prefix, *args]
    assert "++output_jsonl_fpath=results/existing.jsonl" in read_args(root / "run-args")
    assert read_args(root / "dependency-check-args") == ["-", *args]
    assert "+uv_venv_dir=/opt/uv_venvs" in read_args(root / "run-args")
    assert "+skip_venv_if_present=true" in read_args(root / "run-args")
    assert not (root / "prefetch-args").exists()
    assert "UNEXPECTED_INSTALL" not in result.stdout
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


def test_dependency_failure_prevents_evaluation(checkout) -> None:
    """An unusable prebuilt image fails before preparation; it is not repaired in the job."""
    root, env = checkout
    result = run_shell(f"bash {SUBMIT} swe", root, env)
    assert result.returncode == 0, result.stderr
    result = run_evaluation(root, env, DRIVER_STATUS="43")
    assert result.returncode == 1
    assert "Rebuild CONTAINER" in result.stderr
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
    assert "UNEXPECTED_INSTALL" not in result.stdout
    assert not (root / "dependency-check-args").exists()
    assert not (root / "prefetch-args").exists()


@pytest.mark.parametrize("batch, expected_venvs", [("core", 14), ("swe", 3)])
@pytest.mark.parametrize("missing_file", [None, "bin/python", "bin/activate"])
def test_prebuilt_check_uses_resolved_server_environments(
    checkout, monkeypatch, capsys, batch: str, expected_venvs: int, missing_file: str | None
) -> None:
    """Check the real recipes' deduplicated image paths, rejecting absent or incomplete venvs."""
    root, env = checkout
    result = run_shell(f"bash {SUBMIT} {batch}", root, env)
    assert result.returncode == 0, result.stderr
    result = run_evaluation(root, env)
    assert result.returncode == 0, result.stderr

    (root / "env.yaml").write_text("{}\n")
    monkeypatch.chdir(root)
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "sandbox.example")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "fixture-key")
    monkeypatch.setenv("NV_INFERENCE_API_KEY", "fixture-key")
    monkeypatch.setattr(sys, "argv", read_args(root / "dependency-check-args"))
    image_root = Path("/opt/uv_venvs")
    missing_venv = image_root / "responses_api_models/vllm_model/.venv"
    checked = set()
    original_exists = Path.exists

    def image_file_exists(path: Path) -> bool:
        if path.is_relative_to(image_root):
            checked.add(path.parent.parent)
            return missing_file is None or path != missing_venv / missing_file
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", image_file_exists)
    if missing_file:
        with pytest.raises(SystemExit, match=f"Missing prebuilt server environments:\\n{missing_venv}"):
            runpy.run_path(str(root / "dependency-check.py"), run_name="__main__")
    else:
        runpy.run_path(str(root / "dependency-check.py"), run_name="__main__")
        assert f"{expected_venvs} server environment paths checked; no packages installed" in capsys.readouterr().out
    assert len(checked) == expected_venvs


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
    assert rollout.num_samples_in_parallel == (512 if batch == "core" else 1024)
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
        assert config.tau2_benchmark_agent.responses_api_agents.tau2.user_model_server.name == "gpt-5_2-2025-12-11"
        # Check the resolved consumer bindings, not just unused model definitions.
        reference = OmegaConf.load(ROOT / BENCHMARK / "eval_container_config.yaml")
        reference.nv_inference_api_key = "fixture-key"
        reference.judge_model = OmegaConf.load(ROOT / "resources_servers/arena/configs/lmarena_v2.yaml").judge_model
        bindings = [
            (config.tau2_benchmark_agent.responses_api_agents.tau2.user_model_server, "gpt-5_2-2025-12-11"),
            (
                config.tau2_banking_knowledge_bm25_grep_artificial_analysis_agent.responses_api_agents.tau2.user_model_server,
                "gpt-5_4-mini-2026-03-17",
            ),
            (
                config.hle_equivalence_llm_judge_resources_server.resources_servers.equivalence_llm_judge.judge_model_server,
                "hle_benchmark_equivalence_judge_model",
            ),
            (
                config.aalcr_benchmark_resources_server.resources_servers.aalcr.judge_model_server,
                "Qwen3-235B-A22B-Instruct-2507-FP8",
            ),
            (
                config.omniscience_omniscience_resources_server.resources_servers.omniscience.judge_model_server,
                "genrm_model",
            ),
            (config.lmarena_v2_benchmark_resources_server.resources_servers.arena.judge_model_server, "judge_model"),
        ]
        for binding, reference_name in bindings:
            models = config[binding.name].responses_api_models
            expected_models = reference[reference_name].responses_api_models
            assert set(models) == set(expected_models)
            backend = next(iter(models))
            prefix = "openai_" if backend == "openai_model" else ""
            for field in ("model", "base_url"):
                assert models[backend][prefix + field] == expected_models[backend][prefix + field]
            assert models[backend][prefix + "api_key"] == "fixture-key"
        assert "aalcr_batch_judge" not in config
        for name in ("hle_batch_judge", "genrm_model"):
            model = config[name].responses_api_models.openai_model
            assert model.max_concurrent_requests == 32
            assert "reasoning_effort" not in model.get("extra_body", {})
        # GPT-4o's structured verdict must match the labels consumed by the scorer.
        hle = config.hle_equivalence_llm_judge_resources_server.resources_servers.equivalence_llm_judge
        hle_reference = OmegaConf.load(ROOT / BENCHMARK / "benchmark_configs/hle_no_tools.yaml")
        hle_reference = (
            hle_reference.hle_benchmark_equivalence_llm_judge_resources_server.resources_servers.equivalence_llm_judge
        )
        for field in ("judge_responses_create_params", "judge_equal_label", "judge_not_equal_label"):
            assert hle[field] == hle_reference[field]
        assert hle.judge_endpoint_max_concurrency == 32


def test_shared_container_config_with_core_suite_covers_both_batches(monkeypatch) -> None:
    """Existing build settings plus the core suite cover both recipes without a new inventory."""
    monkeypatch.chdir(ROOT)
    parser = GlobalConfigDictParser()
    components = {}
    paths_by_recipe = {
        "core": [str(BENCHMARK / "batch_configs/core.yaml")],
        "swe": [str(BENCHMARK / "batch_configs/swe.yaml")],
        "container": [str(BENCHMARK / "core_text.yaml"), str(BENCHMARK / "eval_container_config.yaml")],
    }
    for recipe, paths in paths_by_recipe.items():
        config = parser.parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=OmegaConf.merge(
                    GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                    {
                        "config_paths": paths,
                        "nv_inference_api_key": "fixture-key",
                    },
                ),
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )
        components[recipe] = {
            (server.SERVER_TYPE, next(iter(getattr(server, server.SERVER_TYPE))))
            for server in parser.filter_for_server_instance_configs(config)
        }
        if recipe == "container":
            # Build-time interpolation must work without an endpoint or credentials.
            OmegaConf.to_container(config, resolve=True)
    assert components["core"] | components["swe"] <= components["container"]
    assert len(components["core"] | components["swe"]) == 16


@pytest.mark.parametrize("extra_configs", [[], ["extra with spaces.yaml", "literal $(touch INJECTED).yaml"]])
@pytest.mark.parametrize("skip_prepare", ["0", "1"])
def test_container_builder_forwards_configs_to_prepare_and_setup(checkout, extra_configs, skip_prepare: str) -> None:
    """Builds preserve config arguments and install dependencies without starting servers."""
    root, env = checkout
    wheel = root / "router.whl"
    wheel.touch()
    env.update(
        INPUT_CONTAINER="base.sqsh",
        OUTPUT_CONTAINER=str(root / "built.sqsh"),
        MOUNTS="",
        GYM_CONFIG="base with spaces.yaml",
        VLLM_ROUTER_WHEEL=str(wheel),
        SKIP_PREPARE=skip_prepare,
    )
    extras = [arg for path in extra_configs for arg in ("--config", path)]
    # Capture the actual generated container script without allocating nodes or building an image.
    capture = 'srun() { cat > "$CAPTURE/build-command"; touch "$OUTPUT_CONTAINER.partial"; }\n'
    result = run_shell(capture + shlex.join(["source", str(BUILDER), *extras]), root, env)
    assert result.returncode == 0, result.stderr
    # Execute that script with system changes stubbed; only Gym argument recording has effects.
    stubs = "\n".join(
        f"{command}() {{ :; }}"
        for command in ("source", "cd", "uv", "git", "apt-get", "gdown", "python3", "mkdir", "rm")
    )
    stubs += '\ngym() { printf "%s\\0" "$@" > "$CAPTURE/$2-args"; }\n'
    result = run_shell(stubs + (root / "build-command").read_text(), root, env)
    assert result.returncode == 0, result.stderr
    configs = [*extras, "--config", env["GYM_CONFIG"]]
    assert read_args(root / "prefetch-args") == [
        "env",
        "prefetch",
        *configs,
        "++uv_venv_dir=/opt/uv_venvs",
    ]
    assert not (root / "start-args").exists()
    if skip_prepare == "0":
        assert read_args(root / "prepare-args") == ["eval", "prepare", "+num_prepare_benchmark_processes=4", *configs]
    else:
        assert not (root / "prepare-args").exists()
    assert not (root / "INJECTED").exists()


@pytest.mark.parametrize("args", [["--config"], ["--unexpected", "config.yaml"]])
def test_container_builder_rejects_invalid_config_arguments(checkout, args: list[str]) -> None:
    """Malformed extra configs fail before launching a build or touching existing output."""
    root, env = checkout
    env.update(
        INPUT_CONTAINER="base.sqsh", OUTPUT_CONTAINER=str(root / "built.sqsh"), MOUNTS="", GYM_CONFIG="base.yaml"
    )
    output = root / "built.sqsh"
    output.write_text("existing image")
    result = run_shell(shlex.join(["bash", str(BUILDER), *args]), root, env)
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert output.read_text() == "existing image"
