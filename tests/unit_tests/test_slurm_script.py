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

from pathlib import Path

import pytest

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors.script_templates import render_driver_entrypoint, render_gym_cmd
from nemo_gym.orchestration.executors.slurm_script import (
    _build_vllm_command,
    _render_directives,
    _render_pool_directives,
    _render_service_command,
    _resolve_env,
    build_sbatch_script,
)
from nemo_gym.orchestration.executors.utils import flatten_run_args as _flatten_run_args


# ---------------------------------------------------------------------------
# flatten_run_args
# ---------------------------------------------------------------------------


def test_scalar_values():
    assert _flatten_run_args({"temperature": 0.05, "top_p": 0.9}) == [
        "+temperature=0.05",
        "+top_p=0.9",
    ]


def test_nested_dict():
    assert _flatten_run_args({"responses_create_params": {"max_concurrent": 92, "temperature": 0.05}}) == [
        "+responses_create_params.max_concurrent=92",
        "+responses_create_params.temperature=0.05",
    ]


def test_list_value():
    assert _flatten_run_args({"config_paths": ["benchmarks/gsm8k/config.yaml", "benchmarks/foo/config.yaml"]}) == [
        "'+config_paths=[benchmarks/gsm8k/config.yaml,benchmarks/foo/config.yaml]'",
    ]


def test_empty():
    assert _flatten_run_args({}) == []


def test_value_with_spaces_is_quoted():
    assert _flatten_run_args({"name": "my model"}) == ["'+name=my model'"]


def test_deeply_nested():
    assert _flatten_run_args({"a": {"b": {"c": 1}}}) == ["+a.b.c=1"]


# ---------------------------------------------------------------------------
# _render_pool_directives
# ---------------------------------------------------------------------------


def test_render_pool_directives_basic(pool):
    lines = _render_pool_directives("main", pool)
    assert "#SBATCH --partition=batch  # pool: main" in lines
    assert "#SBATCH --nodes=1" in lines
    assert "#SBATCH --ntasks-per-node=4" in lines


def test_render_pool_directives_gpus(pool):
    pool.gpus_per_node = 4
    lines = _render_pool_directives("main", pool)
    assert "#SBATCH --gpus-per-node=4" in lines


def test_render_pool_directives_extra_args(pool):
    pool.extra_args["gres"] = "shard:8"
    lines = _render_pool_directives("main", pool)
    assert "#SBATCH --gres=shard:8" in lines


# ---------------------------------------------------------------------------
# _render_directives
# ---------------------------------------------------------------------------


def test_render_directives_job_name(compute, bench_dir):
    out = _render_directives(compute, bench_dir, "gsm8k")
    assert "#SBATCH --job-name=gym-gsm8k" in out


def test_render_directives_account(compute, bench_dir):
    out = _render_directives(compute, bench_dir, "gsm8k")
    assert "#SBATCH --account=my-account" in out


def test_render_directives_walltime(compute, bench_dir):
    compute.walltime = "01:00:00"
    out = _render_directives(compute, bench_dir, "gsm8k")
    assert "#SBATCH --time=01:00:00" in out


def test_render_directives_no_walltime(compute, bench_dir):
    compute.walltime = None
    out = _render_directives(compute, bench_dir, "gsm8k")
    assert "--time" not in out


def test_render_directives_chdir(compute, bench_dir):
    out = _render_directives(compute, bench_dir, "gsm8k")
    assert f"#SBATCH --chdir={bench_dir}" in out


# ---------------------------------------------------------------------------
# _render_service_command
# ---------------------------------------------------------------------------


def test_render_service_command_contains_srun():
    out = _render_service_command("vllm_model", "vllm:latest", "vllm serve model")
    assert "srun --overlap --no-container-mount-home" in out
    assert "--container-image=vllm:latest" in out
    assert "vllm serve model" in out
    assert out.endswith("VLLM_MODEL_PID=$!")


def test_render_service_command_backgrounded():
    out = _render_service_command("vllm_model", "vllm:latest", "vllm serve model")
    assert "& " in out or out.split("\n")[1].endswith(" &")


def test_render_service_command_log_file():
    out = _render_service_command("my_service", "img:latest", "cmd")
    assert "--output=logs/my_service.log" in out


# ---------------------------------------------------------------------------
# _build_vllm_command
# ---------------------------------------------------------------------------


def test_build_vllm_command_basic(vllm_service):
    cmd = _build_vllm_command(vllm_service)
    assert "vllm serve" in cmd
    assert "--port 8000" in cmd
    assert "--tensor-parallel-size 1" in cmd


def test_build_vllm_command_trust_remote_code(vllm_service):
    vllm_service.trust_remote_code = True
    cmd = _build_vllm_command(vllm_service)
    assert "--trust-remote-code" in cmd


def test_build_vllm_command_no_trust_remote_code_by_default(vllm_service):
    cmd = _build_vllm_command(vllm_service)
    assert "--trust-remote-code" not in cmd


# ---------------------------------------------------------------------------
# render_gym_cmd
# ---------------------------------------------------------------------------


def test_render_gym_cmd_subcommand():
    out = render_gym_cmd("eval run", "GYM_CMD", ["+foo=bar"])
    assert out.startswith("GYM_CMD=(")
    assert "gym eval run" in out
    assert "+foo=bar" in out


def test_render_gym_cmd_prepare():
    out = render_gym_cmd("eval prepare", "GYM_PREPARE_CMD", [])
    assert "gym eval prepare" in out
    assert "GYM_PREPARE_CMD=(" in out


# ---------------------------------------------------------------------------
# render_driver_entrypoint
# ---------------------------------------------------------------------------


def test_render_driver_entrypoint_no_install_no_prepare():
    out = render_driver_entrypoint(None, None, None)
    assert out == '"${GYM_CMD[@]}"'


def test_render_driver_entrypoint_with_gym_install():
    out = render_driver_entrypoint("https://github.com/NVIDIA-NeMo/gym", "main", None)
    assert "git clone" in out
    assert "git checkout main" in out
    assert "uv pip install -e . --system" in out
    assert 'exec "$@"' in out
    assert '"${GYM_CMD[@]}"' in out


def test_render_driver_entrypoint_with_prepare():
    out = render_driver_entrypoint(None, None, "gym eval prepare +foo=bar")
    assert "gym eval prepare +foo=bar" in out
    assert 'exec "$@"' in out


def test_render_driver_entrypoint_install_and_prepare():
    out = render_driver_entrypoint("https://github.com/NVIDIA-NeMo/gym", "v1.0", "gym eval prepare")
    assert "git clone" in out
    assert "git checkout v1.0" in out
    assert "gym eval prepare" in out
    assert 'exec "$@"' in out


# ---------------------------------------------------------------------------
# build_sbatch_script (integration)
# ---------------------------------------------------------------------------


def test_build_sbatch_script_contains_shebang(submit_config, bench_dir):
    benchmark = submit_config.driver.benchmarks["gsm8k"]
    compute = next(iter(submit_config.compute.values()))
    script = build_sbatch_script(submit_config, "gsm8k", benchmark, compute, bench_dir)
    assert script.startswith("#!/bin/bash")


def test_build_sbatch_script_contains_vllm_srun(submit_config, bench_dir):
    benchmark = submit_config.driver.benchmarks["gsm8k"]
    compute = next(iter(submit_config.compute.values()))
    script = build_sbatch_script(submit_config, "gsm8k", benchmark, compute, bench_dir)
    assert "vllm serve" in script
    assert "srun --overlap" in script


def test_build_sbatch_script_driver_output_flag(submit_config, bench_dir):
    benchmark = submit_config.driver.benchmarks["gsm8k"]
    compute = next(iter(submit_config.compute.values()))
    script = build_sbatch_script(submit_config, "gsm8k", benchmark, compute, bench_dir)
    assert "--output=logs/driver.log" in script


def test_build_sbatch_script_output_jsonl_fpath(submit_config, bench_dir):
    benchmark = submit_config.driver.benchmarks["gsm8k"]
    compute = next(iter(submit_config.compute.values()))
    script = build_sbatch_script(submit_config, "gsm8k", benchmark, compute, bench_dir)
    assert "+output_jsonl_fpath=artifacts/rollouts.jsonl" in script


def test_build_sbatch_script_policy_model_flags(submit_config_with_policy, bench_dir):
    config = submit_config_with_policy
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    assert "--model-type openai_model" in script
    assert "+policy_base_url=" in script
    assert "+policy_model_name=" in script


# ---------------------------------------------------------------------------
# _resolve_env
# ---------------------------------------------------------------------------


def test_resolve_env_literal():
    out = _resolve_env({"FOO": "bar", "BAZ": "qux"})
    assert "FOO=bar" in out
    assert "BAZ=qux" in out
    assert out.startswith("env ")


def test_resolve_env_value_with_spaces():
    out = _resolve_env({"MSG": "hello world"})
    assert "MSG='hello world'" in out


def test_resolve_env_empty():
    assert _resolve_env({}) == ""


def test_resolve_env_invalid_key_raises():
    with pytest.raises(ValueError, match="Invalid environment variable name"):
        _resolve_env({"X; rm -rf /": "v"})


def test_resolve_env_valid_keys():
    out = _resolve_env({"_VALID_KEY": "a", "key1": "b", "KEY_123": "c"})
    assert "_VALID_KEY=a" in out
    assert "key1=b" in out
    assert "KEY_123=c" in out


def test_resolve_env_invalid_key_with_spaces():
    with pytest.raises(ValueError, match="Invalid environment variable name"):
        _resolve_env({"KEY WITH SPACE": "v"})


def test_resolve_env_invalid_key_with_hyphen():
    with pytest.raises(ValueError, match="Invalid environment variable name"):
        _resolve_env({"KEY-NAME": "v"})


def test_resolve_env_invalid_key_starts_with_digit():
    with pytest.raises(ValueError, match="Invalid environment variable name"):
        _resolve_env({"1KEY": "v"})


def test_resolve_env_invalid_key_with_equals():
    with pytest.raises(ValueError, match="Invalid environment variable name"):
        _resolve_env({"KEY=BAD": "v"})


def test_resolve_env_invalid_key_with_dollar():
    with pytest.raises(ValueError, match="Invalid environment variable name"):
        _resolve_env({"$KEY": "v"})


def test_resolve_env_value_with_semicolons_is_quoted():
    out = _resolve_env({"KEY": "val;rm -rf /"})
    assert "KEY='val;rm -rf /'" in out


def test_resolve_env_value_with_newline_is_quoted():
    out = _resolve_env({"KEY": "line1\nline2"})
    assert "KEY='line1\nline2'" in out


# ---------------------------------------------------------------------------
# build_sbatch_script — env injection
# ---------------------------------------------------------------------------


def test_build_sbatch_script_resolved_tp_in_vllm_cmd(bench_dir):
    config = SubmitConfig.model_validate(
        {
            "services": {
                "vllm_model": {
                    "type": "vllm",
                    "container": "vllm:latest",
                    "model": "org/model",
                    "tensor_parallel_size": 8,
                }
            },
            "compute": {"cluster": {"type": "slurm", "account": "my-account", "hostname": "foo"}},
            "driver": {"container": "python:3.12", "benchmarks": {"gsm8k": {}}},
            "job": {"output_path": "/remote/jobs"},
        }
    )
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    assert "--tensor-parallel-size 8" in script


def test_build_sbatch_script_service_env_before_driver_env(bench_dir):
    config = SubmitConfig.model_validate(
        {
            "services": {
                "vllm_model": {
                    "type": "vllm",
                    "container": "vllm:latest",
                    "model": "org/model",
                    "env": {"SVC_KEY": "svc_val"},
                }
            },
            "compute": {"cluster": {"type": "slurm", "account": "my-account", "hostname": "foo"}},
            "driver": {"container": "python:3.12", "benchmarks": {"gsm8k": {}}, "env": {"DRV_KEY": "drv_val"}},
            "job": {"output_path": "/remote/jobs"},
        }
    )
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    svc_env_idx = script.index("SVC_KEY=svc_val")
    drv_env_idx = script.index("DRV_KEY=drv_val")
    svc_srun_idx = script.index("srun --overlap --no-container-mount-home --container-image=vllm:latest")
    drv_srun_idx = script.index("srun --overlap --no-container-mount-home --container-image=python:3.12")
    # Service env prefix appears before service srun; driver env prefix appears before driver srun.
    assert svc_env_idx < svc_srun_idx
    assert drv_env_idx < drv_srun_idx
    # Service env prefix appears before driver env prefix.
    assert svc_env_idx < drv_env_idx


def test_render_service_command_with_env():
    out = _render_service_command("svc", "img:latest", "cmd", {"FOO": "bar"})
    assert "FOO=bar" in out
    # env prefix must appear before srun on the same line or earlier
    foo_idx = out.index("FOO=bar")
    srun_idx = out.index("srun")
    assert foo_idx < srun_idx


def test_render_service_command_no_env():
    out = _render_service_command("svc", "img:latest", "cmd")
    assert "export" not in out


def test_build_sbatch_script_service_env(bench_dir):
    config = SubmitConfig.model_validate(
        {
            "services": {
                "vllm_model": {
                    "type": "vllm",
                    "container": "vllm:latest",
                    "model": "org/model",
                    "env": {"HF_TOKEN": "hf_test", "LIT": "val"},
                }
            },
            "compute": {"cluster": {"type": "slurm", "account": "my-account", "hostname": "foo"}},
            "driver": {"container": "python:3.12", "benchmarks": {"gsm8k": {}}},
            "job": {"output_path": "/remote/jobs"},
        }
    )
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    assert "HF_TOKEN=hf_test" in script
    assert "LIT=val" in script


def test_build_sbatch_script_driver_env(bench_dir):
    config = SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": {"type": "slurm", "account": "my-account", "hostname": "foo"}},
            "driver": {
                "container": "python:3.12",
                "benchmarks": {"gsm8k": {}},
                "env": {"WANDB_API_KEY": "wb_secret"},  # pragma: allowlist secret
            },
            "job": {"output_path": "/remote/jobs"},
        }
    )
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    assert "WANDB_API_KEY=wb_secret" in script


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

from nemo_gym.orchestration.api import NodePool, SlurmComputeConfig, VllmServiceConfig


@pytest.fixture
def bench_dir():
    return Path("/remote/jobs/gym-job-20260729/gsm8k")


@pytest.fixture
def pool():
    return NodePool(partition="batch", nodes=1, ntasks_per_node=4)


@pytest.fixture
def compute():
    return SlurmComputeConfig(type="slurm", account="my-account", hostname="foo")


@pytest.fixture
def vllm_service():
    return VllmServiceConfig(type="vllm", container="vllm:latest", model="org/model")


@pytest.fixture
def submit_config():
    return SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": {"type": "slurm", "account": "my-account", "hostname": "foo"}},
            "driver": {"container": "python:3.12", "benchmarks": {"gsm8k": {}}},
            "job": {"output_path": "/remote/jobs"},
        }
    )


@pytest.fixture
def submit_config_with_policy():
    return SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": {"type": "slurm", "account": "my-account", "hostname": "foo"}},
            "driver": {"container": "python:3.12", "policy_model": "vllm_model", "benchmarks": {"gsm8k": {}}},
            "job": {"output_path": "/remote/jobs"},
        }
    )
