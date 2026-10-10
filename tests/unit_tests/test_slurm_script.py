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

import base64
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

import nemo_gym.orchestration.executors.slurm_script as slurm_script_module
from nemo_gym.orchestration.api import NodePool, RayServiceConfig, SubmitConfig, VllmPDTierConfig
from nemo_gym.orchestration.executors.script_templates import (
    render_driver_entrypoint,
    render_gym_cmd,
)
from nemo_gym.orchestration.executors.slurm_script import (
    _RAY_SERVE_GATEWAY_SOURCE_PATH,
    _build_ray_command,
    _build_service_command,
    _build_vllm_command,
    _build_vllm_multi_instance_multi_node_command,
    _build_vllm_ray_command,
    _build_vllm_ray_serve_command,
    _kv_transfer_flag,
    _node_totals,
    _render_directives,
    _render_pool_directives,
    _render_service_command,
    _resolve_env,
    _with_default_capture_dir,
    _with_resume_flag,
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
    lines = _render_pool_directives({"main": pool})
    assert "#SBATCH --partition=batch" in lines
    assert "#SBATCH --nodes=1" in lines
    assert "#SBATCH --ntasks-per-node=4" in lines


def test_render_pool_directives_gpus(pool):
    pool.gpus_per_node = 4
    lines = _render_pool_directives({"main": pool})
    assert "#SBATCH --gpus-per-node=4" in lines


def test_render_pool_directives_extra_args(pool):
    pool.extra_args["gres"] = "shard:8"
    lines = _render_pool_directives({"main": pool})
    assert "#SBATCH --gres=shard:8" in lines


def test_pools_ask_for_the_sum_of_their_nodes_once(pool):
    # One #SBATCH --nodes for the allocation, not one per pool. Emitting them per
    # pool made every pool but the last a no-op while the rest of the executor
    # sized itself on the sum, so a two-pool job asked for one pool's nodes.
    aux = NodePool(partition="batch", nodes=3, ntasks_per_node=4)
    lines = _render_pool_directives({"main": pool, "aux": aux})
    assert [line for line in lines if line.startswith("#SBATCH --nodes")] == ["#SBATCH --nodes=4"]


def test_pools_that_disagree_on_a_whole_allocation_directive_are_refused(pool):
    other = NodePool(partition="interactive", nodes=1, ntasks_per_node=4)
    with pytest.raises(ValueError, match="disagree on partition"):
        _render_pool_directives({"main": pool, "other": other})


def test_pools_that_disagree_on_extra_args_are_refused(pool):
    pool.extra_args["gres"] = "shard:8"
    other = NodePool(partition="batch", nodes=1, ntasks_per_node=4, extra_args={"gres": "shard:4"})
    with pytest.raises(ValueError, match="conflicts with another pool"):
        _render_pool_directives({"main": pool, "other": other})


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
    assert "--output=logs/my_service-$SLURM_JOB_ID.log" in out


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


def test_build_vllm_command_multi_instance():
    service = VllmServiceConfig(
        type="vllm",
        container="vllm:latest",
        model="org/model",
        number_of_instances=4,
    )
    cmd = _build_vllm_command(service)
    assert "--data-parallel-size 4" in cmd


def test_build_vllm_command_single_instance_omits_dp_flag(vllm_service):
    cmd = _build_vllm_command(vllm_service)
    assert "--data-parallel-size" not in cmd


def test_build_vllm_command_pipeline_parallel():
    service = VllmServiceConfig(type="vllm", container="vllm:latest", model="org/model", pipeline_parallel_size=2)
    cmd = _build_vllm_command(service)
    assert "--pipeline-parallel-size 2" in cmd


def test_build_vllm_command_pipeline_parallel_1_omits_flag(vllm_service):
    cmd = _build_vllm_command(vllm_service)
    assert "--pipeline-parallel-size" not in cmd


def test_build_vllm_command_extra_args():
    service = VllmServiceConfig(
        type="vllm",
        container="vllm:latest",
        model="org/model",
        extra_args="--max-model-len 8192",
    )
    cmd = _build_vllm_command(service)
    assert "--max-model-len 8192" in cmd


def test_build_vllm_command_no_extra_args_by_default(vllm_service):
    cmd = _build_vllm_command(vllm_service)
    assert cmd.endswith("--tensor-parallel-size 1")


def test_build_vllm_command_served_model_name():
    service = VllmServiceConfig(
        type="vllm",
        container="vllm:latest",
        model="/checkpoint",
        served_model_name="super-bf16",
    )
    cmd = _build_vllm_command(service)
    assert "--served-model-name super-bf16" in cmd


def test_build_vllm_command_no_served_model_name_by_default(vllm_service):
    cmd = _build_vllm_command(vllm_service)
    assert "--served-model-name" not in cmd


def test_build_vllm_command_served_model_name_quoted_if_needed():
    service = VllmServiceConfig(
        type="vllm",
        container="vllm:latest",
        model="/checkpoint",
        served_model_name="name with spaces",
    )
    cmd = _build_vllm_command(service)
    assert "--served-model-name 'name with spaces'" in cmd


# ---------------------------------------------------------------------------
# _build_vllm_ray_command - single instance, TP/PP spans nodes (uses Ray core)
# ---------------------------------------------------------------------------


def test_build_vllm_ray_command_uses_ray_distributed_executor(vllm_service):
    cmd = _build_vllm_ray_command(vllm_service, total_nodes=2)
    assert "--distributed-executor-backend ray" in cmd
    assert "vllm serve" in cmd


def test_build_vllm_ray_command_wraps_in_symmetric_run(vllm_service):
    cmd = _build_vllm_ray_command(vllm_service, total_nodes=2)
    assert "ray symmetric-run" in cmd
    assert "--min-nodes 2" in cmd
    assert '--address "$RAY_HEAD_NODE_IP"' in cmd


def test_build_vllm_ray_command_not_ray_serve_library():
    # Sanity check the plan constraint: this must not shell out to `serve` / ray.serve.
    service = VllmServiceConfig(type="vllm", container="vllm:latest", model="org/model")
    cmd = _build_vllm_ray_command(service, total_nodes=2)
    assert "ray.serve" not in cmd
    assert "serve.run" not in cmd


def test_build_vllm_ray_command_installs_ray_if_missing(vllm_service):
    # Model-serving images (e.g. vllm/vllm-openai) don't necessarily bundle the ray CLI.
    cmd = _build_vllm_ray_command(vllm_service, total_nodes=2)
    assert 'command -v ray >/dev/null 2>&1 || pip install -q "ray[default]"' in cmd


def test_build_vllm_ray_command_raises_symmetric_run_node_wait_timeout(vllm_service):
    # Default 30s node-join wait is too short for slow image pulls; must be exported before use.
    cmd = _build_vllm_ray_command(vllm_service, total_nodes=2)
    assert "export RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT=" in cmd
    assert cmd.index("export RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT=") < cmd.index("ray symmetric-run")


# ---------------------------------------------------------------------------
# _build_vllm_ray_command - multiple instances (data parallel) span nodes
# ---------------------------------------------------------------------------


def test_build_vllm_ray_command_dp_does_not_use_ray():
    # Multi-node DP uses vLLM's own --data-parallel-address/--headless coordination, not Ray.
    service = VllmServiceConfig(
        type="vllm",
        container="vllm:latest",
        model="org/model",
        number_of_instances=4,
    )
    cmd = _build_vllm_ray_command(service, total_nodes=2)
    assert "ray" not in cmd
    assert "symmetric-run" not in cmd


def test_build_vllm_ray_command_dp_head_and_worker_branches():
    service = VllmServiceConfig(
        type="vllm",
        container="vllm:latest",
        model="org/model",
        number_of_instances=4,
    )
    cmd = _build_vllm_ray_command(service, total_nodes=2)
    assert 'if [ "$SLURM_NODEID" = "0" ]; then' in cmd
    assert "--headless" in cmd
    assert "--data-parallel-size 4" in cmd
    assert "--data-parallel-size-local 2" in cmd
    assert '--data-parallel-address "$HEAD_NODE_IP"' in cmd
    assert "--data-parallel-rpc-port 13345" in cmd
    assert "--data-parallel-start-rank $(( SLURM_NODEID * 2 ))" in cmd


# ---------------------------------------------------------------------------
# _build_vllm_ray_serve_command / Ray Serve gateway selection
# ---------------------------------------------------------------------------


def test_build_vllm_ray_serve_command_single_node_no_ray_bootstrap(vllm_service):
    cmd = _build_vllm_ray_serve_command(vllm_service, total_nodes=1, gpus_per_node_values=[])
    assert "ray_serve_gateway.py" in cmd
    assert "base64 -d" in cmd
    assert "git clone" not in cmd  # no gym_install needed - the gateway's source is embedded
    assert "ray symmetric-run" not in cmd
    assert "vllm serve" not in cmd  # the gateway itself launches vllm serve, not this bash command


def test_build_vllm_ray_serve_command_embeds_actual_gateway_source(vllm_service):
    # The base64 blob must decode back to the real, current ray_serve_gateway.py source.
    cmd = _build_vllm_ray_serve_command(vllm_service, total_nodes=1, gpus_per_node_values=[])
    match = re.search(r"printf '%s' '([A-Za-z0-9+/=]+)' \| base64 -d > ray_serve_gateway\.py", cmd)
    assert match, cmd
    decoded = base64.b64decode(match.group(1)).decode()
    assert decoded == _RAY_SERVE_GATEWAY_SOURCE_PATH.read_text()


def test_build_vllm_ray_serve_command_single_node_ensures_ray_installed(vllm_service):
    # Regression test: the single-node path invokes python3 directly, so ray isn't guaranteed
    # importable there unlike the multi-node path (gated via `ray symmetric-run`/`ray start`).
    cmd = _build_vllm_ray_serve_command(vllm_service, total_nodes=1, gpus_per_node_values=[])
    assert 'command -v ray >/dev/null 2>&1 || pip install -q \\"ray[default]\\"' in cmd
    assert cmd.index("command -v ray") < cmd.index("python3 ray_serve_gateway.py")


def test_build_vllm_ray_serve_command_single_node_ray_guard_does_not_bypass_earlier_failure(vllm_service, tmp_path):
    # Regression test: without parens around the ray guard, && and || precedence would let an
    # earlier step's failure be masked by the ray-install fallback, launching the gateway anyway.
    cmd = _build_vllm_ray_serve_command(vllm_service, total_nodes=1, gpus_per_node_values=[])

    inner = cmd.replace("pip install --quiet aiohttp", "false").replace(
        "python3 ray_serve_gateway.py", "echo GATEWAY_LAUNCHED"
    )
    script = 'command() { [ "$2" = ray ] && return 1 || builtin command "$@"; }\nexport -f command\n' + inner + "\n"
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10, cwd=tmp_path)

    assert result.returncode != 0
    assert "GATEWAY_LAUNCHED" not in result.stdout


def test_build_vllm_ray_serve_command_single_node_model_with_space_survives_quoting(tmp_path):
    # Regression test: shlex.quote() wraps a model name with a space in literal single quotes,
    # which would terminate the outer bash -lc '...' wrapper early if not double-quote-escaped.
    service = VllmServiceConfig(type="vllm", container="vllm:latest", model="org/my model")
    cmd = _build_vllm_ray_serve_command(service, total_nodes=1, gpus_per_node_values=[])

    # Stub pip so the aiohttp install and the ray fallback (taken when ray isn't on PATH) never install for real.
    inner = cmd.replace("python3 ray_serve_gateway.py", "fake_gateway")
    script = (
        'pip() { :; }\nfake_gateway() { for a in "$@"; do echo "ARG:$a"; done; }\nexport -f pip fake_gateway\n'
        + inner
        + "\n"
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    args = [line.removeprefix("ARG:") for line in result.stdout.splitlines()]
    assert "--model" in args, f"corrupted command, got args: {args}"
    assert args[args.index("--model") + 1] == "org/my model"


def test_build_vllm_ray_serve_command_multi_node_wraps_in_symmetric_run(vllm_service):
    cmd = _build_vllm_ray_serve_command(vllm_service, total_nodes=2, gpus_per_node_values=[8])
    assert "ray symmetric-run" in cmd
    assert "--min-nodes 2" in cmd
    assert "ray_serve_gateway.py" in cmd
    assert "base64 -d" in cmd


def test_build_vllm_ray_serve_command_multi_node_raises_queue_length_response_deadline(vllm_service):
    # Default replica queue-length RPC deadline (0.1s) is too tight for cross-node hops; must be
    # exported before `ray start`/`ray symmetric-run` runs so every node's raylet has it from birth.
    cmd = _build_vllm_ray_serve_command(vllm_service, total_nodes=2, gpus_per_node_values=[8])
    assert "export RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S=" in cmd
    assert cmd.index("export RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S=") < cmd.index("ray symmetric-run")


def test_build_vllm_ray_serve_command_multi_node_chain_survives_symmetric_run_entrypoint(vllm_service):
    # Regression test: the whole write-then-install-then-launch chain must reach `ray symmetric-run`
    # as one opaque token, not get split by the outer bash -lc live-parsing its own && operators.
    cmd = _build_vllm_ray_serve_command(vllm_service, total_nodes=2, gpus_per_node_values=[8])

    script = cmd.replace(
        "ray symmetric-run",
        'fake_symmetric_run() { for a in "$@"; do echo "ARG:$a"; done; }; fake_symmetric_run',
    ).replace("if ray symmetric-run --help", "if true")
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)

    assert result.returncode == 0, result.stderr
    args = [line.removeprefix("ARG:") for line in result.stdout.splitlines()]
    assert args[-3:-1] == ["bash", "-c"]
    chain = args[-1]
    assert "base64 -d" in chain
    assert "&&" in chain
    assert "python3 ray_serve_gateway.py" in chain


def test_build_vllm_ray_serve_command_passes_gpus_per_node():
    cmd = _build_vllm_ray_serve_command(
        VllmServiceConfig(type="vllm", container="vllm:latest", model="org/model"),
        total_nodes=2,
        gpus_per_node_values=[8],
    )
    assert "--gpus-per-node 8" in cmd


def test_build_vllm_ray_serve_command_omits_gpus_per_node_when_unknown(vllm_service):
    cmd = _build_vllm_ray_serve_command(vllm_service, total_nodes=1, gpus_per_node_values=[])
    assert "--gpus-per-node" not in cmd


def test_build_vllm_ray_serve_command_passes_flags():
    service = VllmServiceConfig(
        type="vllm",
        container="vllm:latest",
        model="org/model",
        port=9000,
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
        number_of_instances=2,
        trust_remote_code=True,
    )
    cmd = _build_vllm_ray_serve_command(service, total_nodes=4, gpus_per_node_values=[8])
    assert "--model org/model" in cmd
    assert "--port 9000" in cmd
    assert "--tensor-parallel-size 8" in cmd
    assert "--pipeline-parallel-size 2" in cmd
    assert "--number-of-instances 2" in cmd
    assert "--trust-remote-code" in cmd


def test_build_vllm_ray_serve_command_passes_served_model_name_and_extra_args():
    service = VllmServiceConfig(
        type="vllm",
        container="vllm:latest",
        model="org/model",
        served_model_name="my-model",
        extra_args="--max-model-len 8192",
    )
    cmd = _build_vllm_ray_serve_command(service, total_nodes=1, gpus_per_node_values=[])
    assert "--served-model-name my-model" in cmd
    assert "--extra-args '--max-model-len 8192'" in cmd


def test_build_service_command_uses_ray_serve_when_opted_in(vllm_service):
    vllm_service.use_ray_serve = True
    cmd = _build_service_command(vllm_service, total_nodes=1, gpus_per_node_values=[8])
    assert "ray_serve_gateway.py" in cmd


def test_build_service_command_default_ignores_ray_serve_single_node(vllm_service):
    cmd = _build_service_command(vllm_service, total_nodes=1, gpus_per_node_values=[8])
    assert "ray_serve_gateway" not in cmd
    assert "vllm serve" in cmd


def test_build_service_command_mandatory_ray_serve_when_instance_spans_nodes():
    # TP*PP=16 > 8 GPUs/node forces Ray Serve on automatically, with no use_ray_serve set.
    service = VllmServiceConfig(
        type="vllm",
        container="vllm:latest",
        model="org/model",
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
        number_of_instances=2,
    )
    cmd = _build_service_command(service, total_nodes=4, gpus_per_node_values=[8])
    assert "ray_serve_gateway.py" in cmd


def test_build_service_command_default_multi_instance_multi_node_unchanged():
    # tp_pp=8 fits within 8 gpus/node - stays on the existing (non-Ray) multi-node DP path.
    service = VllmServiceConfig(
        type="vllm",
        container="vllm:latest",
        model="org/model",
        tensor_parallel_size=8,
        number_of_instances=4,
    )
    cmd = _build_service_command(service, total_nodes=2, gpus_per_node_values=[8])
    assert "ray_serve_gateway" not in cmd
    assert "--headless" in cmd


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


def test_driver_policy_model_type_defaults_to_openai_model(submit_config, bench_dir):
    submit_config.driver.policy_model = "vllm_model"
    benchmark = submit_config.driver.benchmarks["gsm8k"]
    compute = next(iter(submit_config.compute.values()))
    script = build_sbatch_script(submit_config, "gsm8k", benchmark, compute, bench_dir)
    assert "--model-type openai_model" in script


def test_driver_policy_model_type_is_configurable(submit_config, bench_dir):
    # Every certified NEL run of these benchmarks serves the policy as
    # vllm_model, and lmarena_v3 ships its own vllm_model policy that a second
    # composed server would collide with.
    submit_config.driver.policy_model = "vllm_model"
    submit_config.driver.policy_model_type = "vllm_model"
    benchmark = submit_config.driver.benchmarks["gsm8k"]
    compute = next(iter(submit_config.compute.values()))
    script = build_sbatch_script(submit_config, "gsm8k", benchmark, compute, bench_dir)
    assert "--model-type vllm_model" in script
    assert "--model-type openai_model" not in script


def test_driver_policy_model_type_empty_composes_nothing(submit_config, bench_dir):
    submit_config.driver.policy_model = "vllm_model"
    submit_config.driver.policy_model_type = ""
    benchmark = submit_config.driver.benchmarks["gsm8k"]
    compute = next(iter(submit_config.compute.values()))
    script = build_sbatch_script(submit_config, "gsm8k", benchmark, compute, bench_dir)
    assert "--model-type" not in script


def test_render_driver_entrypoint_no_install_no_prepare():
    out = render_driver_entrypoint(None, None, None)
    assert out == '"${GYM_CMD[@]}"'


def test_render_driver_entrypoint_with_gym_install():
    out = render_driver_entrypoint("https://github.com/NVIDIA-NeMo/gym", "main", None)
    assert "git clone" in out
    # `git -C "$GYM_SRC/gym" checkout`, not `git checkout`: the clone is
    # out-of-tree. See test_gym_install_does_not_clone_into_the_job_directory.
    assert "checkout main" in out
    assert "uv venv --seed .venv" in out
    assert "source .venv/bin/activate" in out
    assert "uv pip install -e ." in out
    assert "--system" not in out
    assert "--break-system-packages" not in out
    assert 'exec "$@"' in out
    assert '"${GYM_CMD[@]}"' in out


def test_render_driver_entrypoint_installs_git_if_missing():
    # The driver container (e.g. a minimal python image) may not bundle git.
    out = render_driver_entrypoint("https://github.com/NVIDIA-NeMo/gym", "main", None)
    assert "command -v git >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq git)" in out
    assert out.index("command -v git") < out.index("git clone")


def test_render_driver_entrypoint_with_prepare():
    out = render_driver_entrypoint(None, None, "gym eval prepare +foo=bar")
    assert "gym eval prepare +foo=bar" in out
    assert 'exec "$@"' in out


def test_render_driver_entrypoint_install_and_prepare():
    out = render_driver_entrypoint("https://github.com/NVIDIA-NeMo/gym", "v1.0", "gym eval prepare")
    assert "git clone" in out
    assert "checkout v1.0" in out
    assert "gym eval prepare" in out
    assert 'exec "$@"' in out


def test_worker_command_drops_api_server_count():
    """vLLM exits on `--api-server-count` in headless mode before loading anything:
    "no API servers are started in headless mode". The flag is valid on the head
    node and arrives via the service's own extra_args, so only the worker branch
    is stripped. A real mmlu-prox run lost all five workers to this.
    """
    service = VllmServiceConfig(
        type="vllm",
        container="vllm:latest",
        model="/checkpoint",
        tensor_parallel_size=4,
        number_of_instances=2,
        extra_args="--api-server-count 1 --enable-prefix-caching",
    )
    block = _build_vllm_multi_instance_multi_node_command(service, total_nodes=2)
    head, worker = block.split("else")

    assert "--api-server-count 1" in head
    assert "--headless" in worker
    assert "--api-server-count" not in worker
    # Stripping must not take the neighbouring flag with it.
    assert "--enable-prefix-caching" in worker


def test_multi_instance_multi_node_command_survives_the_shell():
    """vLLM's JSON flags are single-quoted, and the DP branches are embedded in a
    single-quoted `bash -lc '...'`. Unescaped they end the block early and the
    whole invocation word-splits -- a real mmlu-prox submission died of this with
    "/usr/bin/env: Argument list too long". Run the rendered block through bash
    and check the JSON arrives as one argument.
    """
    service = VllmServiceConfig(
        type="vllm",
        container="vllm:latest",
        model="/checkpoint",
        tensor_parallel_size=4,
        number_of_instances=2,
        extra_args='--hf-overrides \'{"architectures":["Custom"],"norm_mean":[0.5,0.5]}\'',
    )
    block = _build_vllm_multi_instance_multi_node_command(service, total_nodes=2)

    # Replace `vllm serve` with a printf that dumps one argument per line, so the
    # test observes what the shell actually passed rather than the rendered text.
    # Double quotes, because this substitution happens AFTER rendering and so is
    # not itself escaped -- single quotes here would break the block the test is
    # checking.
    script = "SLURM_NODEID=0 HEAD_NODE_IP=1.2.3.4 " + block.replace("vllm serve", 'printf "%s\\n"', 2)
    argv = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.splitlines()

    assert '{"architectures":["Custom"],"norm_mean":[0.5,0.5]}' in argv


def test_render_driver_entrypoint_prepare_arg_with_spaces_survives_the_shell():
    """A prepare argument containing spaces must reach Hydra as ONE word.

    The entrypoint body is embedded in a single-quoted `bash -c '...'`, so an
    inner single quote ends the outer string instead of nesting. Without
    escaping, gdpval's real prepare argument word-split and Hydra failed with
    "no viable alternative at input '[{num_tasks:'". Asserting on the rendered
    string would not catch that -- only running it through a shell does.
    """
    arg = "+multistage.stages=[{num_tasks: 45, waivable: [timeout, transient]}]"
    out = render_driver_entrypoint(None, None, f"printf '%s\\n' {shlex.quote(arg)}")

    script = out.replace('exec "$@"', ":").replace('"${GYM_CMD[@]}"', "''")
    printed = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.splitlines()

    assert printed == [arg]


def test_render_driver_entrypoint_no_install_no_prepare_has_no_set_e():
    # The trivial path isn't wrapped in bash -c at all, so there's no
    # preamble for a failure to silently fall through in the first place.
    out = render_driver_entrypoint(None, None, None)
    assert "set -euo pipefail" not in out


def test_render_driver_entrypoint_with_gym_install_sets_e():
    out = render_driver_entrypoint("https://github.com/NVIDIA-NeMo/gym", "main", None)
    assert "set -euo pipefail" in out
    # Must be the first statement, ahead of the clone/checkout/install, so a
    # failure anywhere in the preamble aborts instead of falling through to
    # exec "$@" against whatever was already on disk/PATH.
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    assert lines[0] == "bash -c '"
    assert lines[1] == "set -euo pipefail"


def test_render_driver_entrypoint_with_prepare_sets_e():
    out = render_driver_entrypoint(None, None, "gym eval prepare +foo=bar")
    assert "set -euo pipefail" in out


# ---------------------------------------------------------------------------
# _with_default_capture_dir
# ---------------------------------------------------------------------------


def test_with_default_capture_dir_injects_when_observability_on():
    run = {"observability_enabled": True}
    out = _with_default_capture_dir(run, Path("/remote/jobs/gym-job-20260729/gsm8k"))
    assert out["model_call_capture_dir"] == "/remote/jobs/gym-job-20260729/gsm8k/model-calls"


@pytest.mark.parametrize("enabled", [True, False])
def test_with_default_capture_dir_explicit_value_wins(enabled):
    run = {"observability_enabled": enabled, "model_call_capture_dir": "/custom/path"}
    out = _with_default_capture_dir(run, Path("/remote/jobs/gym-job-20260729/gsm8k"))
    assert out["observability_enabled"] is enabled
    assert out["model_call_capture_dir"] == "/custom/path"


def test_with_default_capture_dir_no_injection_when_observability_off():
    run = {"split": "benchmark", "observability_enabled": False}
    out = _with_default_capture_dir(run, Path("/remote/jobs/gym-job-20260729/gsm8k"))
    assert "model_call_capture_dir" not in out


def test_with_default_capture_dir_does_not_mutate_input():
    run = {}
    _with_default_capture_dir(run, Path("/remote/jobs/gym-job-20260729/gsm8k"))
    assert run == {}


# ---------------------------------------------------------------------------
# _with_resume_flag
# ---------------------------------------------------------------------------


def test_with_resume_flag_injects_when_resumable():
    assert _with_resume_flag({"split": "benchmark"}, True) == {"split": "benchmark", "resume_from_cache": True}


def test_with_resume_flag_no_injection_when_not_resumable():
    run = {"split": "benchmark"}
    assert _with_resume_flag(run, False) == run


def test_with_resume_flag_explicit_value_wins():
    run = {"resume_from_cache": False}
    assert _with_resume_flag(run, True) == {"resume_from_cache": False}


def test_with_resume_flag_does_not_mutate_input():
    run = {"split": "benchmark"}
    _with_resume_flag(run, True)
    assert "resume_from_cache" not in run


# ---------------------------------------------------------------------------
# build_sbatch_script (integration)
# ---------------------------------------------------------------------------


def test_build_sbatch_script_resumable_adds_prologue_and_resume_flag(bench_dir):
    config = SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": {"type": "slurm", "account": "my-account", "hostname": "foo"}},
            "driver": {
                "container": "python:3.12",
                "benchmarks": {"gsm8k": {"resumable": {"max_retries": 5, "max_walltime": "10:00:00"}}},
            },
            "job": {"output_path": "/remote/jobs"},
        }
    )
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    assert "# --- Auto-resume chain ---" in script
    assert '_this_script="$OUTPUT_DIR/job.sh"' in script
    assert "_gym_accumulated >= 36000" in script
    assert "-gt 5" in script
    assert "+resume_from_cache=True" in script


def test_build_sbatch_script_not_resumable_omits_prologue(submit_config, bench_dir):
    benchmark = submit_config.driver.benchmarks["gsm8k"]
    compute = next(iter(submit_config.compute.values()))
    script = build_sbatch_script(submit_config, "gsm8k", benchmark, compute, bench_dir)
    assert "Auto-resume chain" not in script
    assert "resume_from_cache" not in script


def test_build_sbatch_script_auto_default_capture_dir(bench_dir):
    config = SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": {"type": "slurm", "account": "my-account", "hostname": "foo"}},
            "driver": {
                "container": "python:3.12",
                "benchmarks": {"gsm8k": {"run": {}}},
            },
            "job": {"output_path": "/remote/jobs"},
        }
    )
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    assert "+observability_enabled=True" in script
    assert f"+model_call_capture_dir={bench_dir / 'model-calls'}" in script


def test_build_sbatch_script_explicit_capture_dir_wins(bench_dir):
    config = SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": {"type": "slurm", "account": "my-account", "hostname": "foo"}},
            "driver": {
                "container": "python:3.12",
                "benchmarks": {
                    "gsm8k": {"run": {"observability_enabled": True, "model_call_capture_dir": "/custom/path"}}
                },
            },
            "job": {"output_path": "/remote/jobs"},
        }
    )
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    assert "+model_call_capture_dir=/custom/path" in script
    assert "model-calls" not in script


def test_build_sbatch_script_no_capture_dir_when_observability_off(submit_config, bench_dir):
    benchmark = submit_config.driver.benchmarks["gsm8k"]
    benchmark.run["observability_enabled"] = False
    compute = next(iter(submit_config.compute.values()))
    script = build_sbatch_script(submit_config, "gsm8k", benchmark, compute, bench_dir)
    assert "+observability_enabled=False" in script
    assert "model_call_capture_dir" not in script


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


@pytest.mark.parametrize(
    ("run_args", "expected"),
    [({}, True), ({"require_complete": True}, True), ({"require_complete": False}, False)],
)
def test_build_sbatch_script_driver_output_flag(submit_config, bench_dir, run_args, expected):
    benchmark = submit_config.driver.benchmarks["gsm8k"]
    benchmark.run.update(run_args)
    compute = next(iter(submit_config.compute.values()))
    script = build_sbatch_script(submit_config, "gsm8k", benchmark, compute, bench_dir)
    assert "--output=logs/driver-$SLURM_JOB_ID.log" in script
    assert f"+require_complete={expected}" in script
    assert f"+require_complete={not expected}" not in script


def test_build_sbatch_script_output_jsonl_fpath(submit_config, bench_dir):
    benchmark = submit_config.driver.benchmarks["gsm8k"]
    compute = next(iter(submit_config.compute.values()))
    script = build_sbatch_script(submit_config, "gsm8k", benchmark, compute, bench_dir)
    # Absolute: a relative output path lands wherever the container's cwd
    # happens to be, and cwd cannot be moved without breaking a benchmark's
    # cwd-relative prepare_script.
    assert f"+output_jsonl_fpath={bench_dir}/artifacts/rollouts.jsonl" in script


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


def test_resolve_env_runtime_marker_emits_unquoted_shell_reference():
    out = _resolve_env({"FOO": "runtime:NEL_INVOCATION_ID"})
    assert "FOO=${NEL_INVOCATION_ID}" in out
    assert "'" not in out


def test_resolve_env_runtime_marker_alongside_literal():
    out = _resolve_env({"LIT": "val", "RUN": "runtime:NEL_INVOCATION_ID"})
    assert "LIT=val" in out
    assert "RUN=${NEL_INVOCATION_ID}" in out


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
                    "env": {"SVC_KEY": "lit:svc_val"},
                }
            },
            "compute": {"cluster": {"type": "slurm", "account": "my-account", "hostname": "foo"}},
            "driver": {"container": "python:3.12", "benchmarks": {"gsm8k": {}}, "env": {"DRV_KEY": "lit:drv_val"}},
            "job": {"output_path": "/remote/jobs"},
        }
    )
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    svc_env_idx = script.index("SVC_KEY=svc_val")
    drv_env_idx = script.index("DRV_KEY=drv_val")
    svc_srun_idx = script.index("--container-image=vllm:latest")
    drv_srun_idx = script.index("--container-image=python:3.12")
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
                    "env": {"HF_TOKEN": "lit:hf_test", "LIT": "lit:val"},
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
                "env": {"WANDB_API_KEY": "lit:wb_secret"},  # pragma: allowlist secret
            },
            "job": {"output_path": "/remote/jobs"},
        }
    )
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    assert "WANDB_API_KEY=wb_secret" in script


# ---------------------------------------------------------------------------
# mounts: service and driver
# ---------------------------------------------------------------------------


def test_render_service_command_with_mounts():
    out = _render_service_command("svc", "img:latest", "cmd", mounts=["/src:/dst", "/data"])
    assert "--container-mounts=/src:/dst,/data" in out


def test_render_service_command_no_mounts_by_default():
    out = _render_service_command("svc", "img:latest", "cmd")
    assert "--container-mounts" not in out


def test_render_service_command_empty_mounts_omits_flag():
    out = _render_service_command("svc", "img:latest", "cmd", mounts=[])
    assert "--container-mounts" not in out


# ---------------------------------------------------------------------------
# _render_service_command — pre_command
# ---------------------------------------------------------------------------


def test_render_service_command_no_pre_command_by_default():
    out = _render_service_command("svc", "img:latest", "vllm serve model")
    assert "bash -c" not in out
    assert (
        "srun --overlap --no-container-mount-home --container-image=img:latest "
        "--output=logs/svc-$SLURM_JOB_ID.log vllm serve model &" in out
    )


def test_render_service_command_pre_command_wraps_in_bash_c():
    out = _render_service_command("svc", "img:latest", "vllm serve model", pre_command="export FOO=bar")
    assert "bash -c 'export FOO=bar\nexec vllm serve model'" in out


def test_render_service_command_pre_command_still_backgrounded():
    out = _render_service_command("svc", "img:latest", "vllm serve model", pre_command="export FOO=bar")
    assert out.rstrip().endswith("&\nSVC_PID=$!")


def test_render_service_command_pre_command_multi_statement_round_trips():
    # Round-trip through shlex, like bash would: the bash -c argument (after
    # shell-unquoting) must be exactly pre_command + a newline + exec <command>,
    # regardless of what quote characters pre_command itself contains.
    pre_command = "export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')\nunset RAY_ADDRESS"
    out = _render_service_command("svc", "img:latest", "vllm serve model", pre_command=pre_command)
    tokens = shlex.split(out)
    assert tokens[tokens.index("bash") + 1] == "-c"
    assert tokens[tokens.index("bash") + 2] == f"{pre_command}\nexec vllm serve model"
    assert out.count("&\n") == 1  # one srun invocation, not split by the embedded newline


def test_render_service_command_pre_command_quoting_survives_single_quotes():
    # pre_command containing a single quote must not break out of the bash -c
    # quoting or split into a second shell word.
    out = _render_service_command("svc", "img:latest", "cmd", pre_command="echo 'hi'")
    assert out.count("bash -c") == 1
    tokens = shlex.split(out)
    assert tokens[tokens.index("bash") + 2] == "echo 'hi'\nexec cmd"


def test_render_service_command_pre_command_and_extra_args_coexist():
    # extra_args lands inside `command` (already appended by the caller before
    # _render_service_command is invoked); pre_command wraps the whole thing.
    out = _render_service_command(
        "svc", "img:latest", "vllm serve model --max-model-len 8192", pre_command="unset RAY_ADDRESS"
    )
    tokens = shlex.split(out)
    assert tokens[tokens.index("bash") + 2] == "unset RAY_ADDRESS\nexec vllm serve model --max-model-len 8192"


def test_build_sbatch_script_service_mounts(bench_dir):
    config = SubmitConfig.model_validate(
        {
            "services": {
                "vllm_model": {
                    "type": "vllm",
                    "container": "vllm:latest",
                    "model": "org/model",
                    "mounts": ["/lustre/datasets:/data", "/tmp/cache"],
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
    assert "--container-mounts=/lustre/datasets:/data,/tmp/cache" in script


def test_build_sbatch_script_driver_mounts(bench_dir):
    config = SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": {"type": "slurm", "account": "my-account", "hostname": "foo"}},
            "driver": {
                "container": "python:3.12",
                "benchmarks": {"gsm8k": {}},
                "mounts": ["/lustre/checkpoints:/ckpts"],
            },
            "job": {"output_path": "/remote/jobs"},
        }
    )
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    # mounts flag appears on the driver srun line
    driver_srun_line = next(line for line in script.splitlines() if "python:3.12" in line)
    assert "--container-mounts=/lustre/checkpoints:/ckpts" in driver_srun_line


def test_build_sbatch_script_no_service_mounts_by_default(submit_config, bench_dir):
    """Services get no mounts unless configured. The driver and the OTel
    collector are the exceptions and always mount the job directory, because that
    is where their artifacts go -- see
    test_driver_can_write_its_artifacts_into_the_job_directory."""
    benchmark = submit_config.driver.benchmarks["gsm8k"]
    compute = next(iter(submit_config.compute.values()))
    script = build_sbatch_script(submit_config, "gsm8k", benchmark, compute, bench_dir)

    service_lines = [
        line
        for line in script.splitlines()
        if "srun" in line
        and "--output=logs/driver-$SLURM_JOB_ID.log" not in line
        and "--output=logs/otel_collector-$SLURM_JOB_ID.log" not in line
    ]
    assert service_lines, "expected at least one service srun line"
    for line in service_lines:
        assert "--container-mounts" not in line


# ---------------------------------------------------------------------------
# _validate_mounts
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock

from nemo_gym.orchestration.executors.connection import LocalConnection
from nemo_gym.orchestration.executors.slurm import _validate_mounts


def _make_submit_config_with_mounts(driver_mounts=None, service_mounts=None):
    return SubmitConfig.model_validate(
        {
            "services": {
                "vllm_model": {
                    "type": "vllm",
                    "container": "vllm:latest",
                    "model": "org/model",
                    **({"mounts": service_mounts} if service_mounts is not None else {}),
                }
            },
            "compute": {"cluster": {"type": "slurm", "account": "acct", "hostname": "foo"}},
            "driver": {
                "container": "python:3.12",
                "benchmarks": {"gsm8k": {}},
                **({"mounts": driver_mounts} if driver_mounts is not None else {}),
            },
            "job": {"output_path": "/remote/jobs"},
        }
    )


def test_validate_mounts_local_passes_when_src_exists(tmp_path):
    src = str(tmp_path)
    config = _make_submit_config_with_mounts(driver_mounts=[f"{src}:/data"])
    # No exception means all srcs were found — implicitly asserted by the call completing.
    _validate_mounts(config, LocalConnection())


def test_validate_mounts_local_raises_for_missing_src(tmp_path):
    src = str(tmp_path / "nonexistent")
    config = _make_submit_config_with_mounts(driver_mounts=[f"{src}:/data"])
    with pytest.raises(ValueError, match="driver") as exc_info:
        _validate_mounts(config, LocalConnection())
    assert src in str(exc_info.value)


def test_validate_mounts_local_parses_flags_format(tmp_path):
    src = str(tmp_path)
    # Passes because src exists; would raise if the code checked "src:ro" as a path instead of "src".
    config = _make_submit_config_with_mounts(driver_mounts=[f"{src}:/data:ro"])
    _validate_mounts(config, LocalConnection())


def test_validate_mounts_remote_passes_when_all_exist():
    config = _make_submit_config_with_mounts(driver_mounts=["/lustre/data:/data"])
    conn = MagicMock()
    conn.run.return_value = ""  # no __GYM_MISSING lines → all exist
    _validate_mounts(config, conn)
    # Confirms conn.run was called with a check for the correct src.
    (commands,), _ = conn.run.call_args
    assert any("/lustre/data" in cmd for cmd in commands)


def test_validate_mounts_remote_raises_for_missing_src():
    config = _make_submit_config_with_mounts(service_mounts=["/lustre/missing:/data"])
    conn = MagicMock()
    conn.run.return_value = "__GYM_MISSING:/lustre/missing"
    with pytest.raises(ValueError, match="services\\.vllm_model") as exc_info:
        _validate_mounts(config, conn)
    assert "/lustre/missing" in str(exc_info.value)


def test_validate_mounts_no_mounts_passes():
    config = _make_submit_config_with_mounts()
    conn = MagicMock()
    _validate_mounts(config, conn)
    conn.run.assert_not_called()


# ---------------------------------------------------------------------------
# _render_service_command — multi-node flags
# ---------------------------------------------------------------------------


def test_render_service_command_multi_node_adds_nodes_and_ntasks():
    out = _render_service_command("vllm_model", "vllm:latest", "vllm serve model", nodes=4, ntasks=4)
    assert "--nodes=4" in out
    assert "--ntasks=4" in out


def test_render_service_command_multi_node_flags_before_container_image():
    out = _render_service_command("vllm_model", "vllm:latest", "vllm serve model", nodes=4, ntasks=4)
    assert out.index("--nodes=4") < out.index("--container-image=")


def test_render_service_command_single_node_omits_node_flags():
    out = _render_service_command("vllm_model", "vllm:latest", "vllm serve model", nodes=1, ntasks=1)
    assert "--nodes=" not in out
    assert "--ntasks=" not in out


def test_render_service_command_no_nodes_kwarg_omits_node_flags():
    out = _render_service_command("vllm_model", "vllm:latest", "vllm serve model")
    assert "--nodes=" not in out
    assert "--ntasks=" not in out


# ---------------------------------------------------------------------------
# _node_totals
# ---------------------------------------------------------------------------


def test_node_totals_empty_pools():
    compute = SlurmComputeConfig(type="slurm", account="acct")
    assert _node_totals(compute) == (0, 0)


def test_node_totals_single_pool():
    compute = SlurmComputeConfig(
        type="slurm",
        account="acct",
        node_pools={"main": NodePool(partition="gpu", nodes=4, ntasks_per_node=2)},
    )
    assert _node_totals(compute) == (4, 8)


def test_node_totals_multiple_pools():
    compute = SlurmComputeConfig(
        type="slurm",
        account="acct",
        node_pools={
            "gpu": NodePool(partition="gpu", nodes=4, ntasks_per_node=1),
            "cpu": NodePool(partition="cpu", nodes=2, ntasks_per_node=2),
        },
    )
    assert _node_totals(compute) == (6, 8)


# ---------------------------------------------------------------------------
# build_sbatch_script — multi-node srun flags
# ---------------------------------------------------------------------------


def _multi_node_config():
    return SubmitConfig.model_validate(
        {
            "services": {
                "vllm_model": {
                    "type": "vllm",
                    "container": "vllm:latest",
                    "model": "org/model",
                    "tensor_parallel_size": 8,
                }
            },
            "compute": {
                "cluster": {
                    "type": "slurm",
                    "account": "my-account",
                    "hostname": "foo",
                    "node_pools": {"main": {"partition": "gpu", "nodes": 4, "ntasks_per_node": 1}},
                }
            },
            "driver": {"container": "python:3.12", "benchmarks": {"gsm8k": {}}},
            "job": {"output_path": "/remote/jobs"},
        }
    )


def test_build_sbatch_script_multi_node_selects_ray_by_node_count(bench_dir):
    # Node count alone is enough to span the vLLM service via ray - no explicit config needed.
    config = _multi_node_config()
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    vllm_line = next(line for line in script.splitlines() if "vllm:latest" in line)
    assert "--nodes=4" in vllm_line
    assert "ray symmetric-run" in script


def test_build_sbatch_script_multi_node_vllm_srun_gets_node_flags(bench_dir):
    config = _multi_node_config()
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    vllm_line = next(line for line in script.splitlines() if "vllm:latest" in line)
    assert "--nodes=4" in vllm_line
    assert "--ntasks=4" in vllm_line


def test_build_sbatch_script_multi_node_driver_srun_gets_nodes_1(bench_dir):
    config = _multi_node_config()
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    driver_line = next(line for line in script.splitlines() if "python:3.12" in line)
    assert "--nodes=1" in driver_line
    assert "--ntasks=1" in driver_line


def test_build_sbatch_script_multi_node_node_flags_before_container_image(bench_dir):
    config = _multi_node_config()
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    vllm_line = next(line for line in script.splitlines() if "vllm:latest" in line)
    assert vllm_line.index("--nodes=4") < vllm_line.index("--container-image=")


def test_build_sbatch_script_single_node_pool_omits_node_flags_from_srun(bench_dir):
    config = SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {
                "cluster": {
                    "type": "slurm",
                    "account": "my-account",
                    "hostname": "foo",
                    "node_pools": {"main": {"partition": "gpu", "nodes": 1, "ntasks_per_node": 4}},
                }
            },
            "driver": {"container": "python:3.12", "benchmarks": {"gsm8k": {}}},
            "job": {"output_path": "/remote/jobs"},
        }
    )
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    vllm_line = next(line for line in script.splitlines() if "vllm:latest" in line)
    driver_line = next(line for line in script.splitlines() if "python:3.12" in line)
    assert "--nodes=" not in vllm_line
    assert "--nodes=" not in driver_line


def test_build_sbatch_script_non_vllm_service_omits_node_flags_in_multi_node_job(bench_dir):
    # A plain Ray head service doesn't span nodes itself, even when the vLLM service alongside it
    # does - it must not get the whole allocation's --nodes/--ntasks (that would launch
    # `ray start --head` once per node instead of once).
    config = SubmitConfig.model_validate(
        {
            "services": {
                "vllm_model": {
                    "type": "vllm",
                    "container": "vllm:latest",
                    "model": "org/model",
                    "tensor_parallel_size": 8,
                },
                "ray_head": {"type": "ray", "container": "ray:latest"},
            },
            "compute": {
                "cluster": {
                    "type": "slurm",
                    "account": "my-account",
                    "hostname": "foo",
                    "node_pools": {"main": {"partition": "gpu", "nodes": 4, "ntasks_per_node": 1}},
                }
            },
            "driver": {"container": "python:3.12", "benchmarks": {"gsm8k": {}}},
            "job": {"output_path": "/remote/jobs"},
        }
    )
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    vllm_line = next(line for line in script.splitlines() if "vllm:latest" in line)
    ray_line = next(line for line in script.splitlines() if "ray:latest" in line)
    assert "--nodes=4" in vllm_line
    # Once, on the driver's node -- not once per node of the allocation.
    assert '--nodelist="${gym_nodes[0]}" --nodes=1 --ntasks=1' in ray_line


# ---------------------------------------------------------------------------
# build_sbatch_script — ray prelude
# ---------------------------------------------------------------------------


def test_build_sbatch_script_ray_backend_adds_head_node_prelude(bench_dir):
    config = _multi_node_config()
    benchmark = config.driver.benchmarks["gsm8k"]
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gsm8k", benchmark, compute, bench_dir)
    assert "scontrol show hostnames" in script
    assert "ray symmetric-run" in script
    # Must be exported: it's read inside a separate `srun ... bash -lc` subprocess, which only
    # inherits *exported* environment variables, not plain shell variables from the parent script.
    assert 'export RAY_HEAD_NODE_IP="$head_node_ip:6379"' in script


def test_build_sbatch_script_vllm_service_backend_omits_ray_prelude(submit_config, bench_dir):
    benchmark = submit_config.driver.benchmarks["gsm8k"]
    compute = next(iter(submit_config.compute.values()))
    script = build_sbatch_script(submit_config, "gsm8k", benchmark, compute, bench_dir)
    assert "scontrol show hostnames" not in script
    assert "ray symmetric-run" not in script


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


def test_driver_can_write_its_artifacts_into_the_job_directory():
    """A run that cannot reach the job directory completes cleanly and produces
    nothing: `output_jsonl_fpath` is relative, `#SBATCH --chdir` only sets the
    host-side cwd of the batch script, and a Pyxis container starts in whatever
    directory its image declares. Both the mount and the workdir are required.
    """
    config = SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {"hsg": {"type": "slurm", "account": "acct"}},
            "driver": {
                "container": "gym:latest",
                "mounts": ["/host/cache:/cache"],
                "benchmarks": {"gpqa": {}},
            },
            "job": {"output_path": "/jobs"},
        }
    )
    bench_dir = Path("/jobs/gym-job-x/gpqa")

    script = build_sbatch_script(config, "gpqa", config.driver.benchmarks["gpqa"], config.compute["hsg"], bench_dir)

    driver_line = next(line for line in script.splitlines() if "--output=logs/driver-$SLURM_JOB_ID.log" in line)
    assert f"{bench_dir}:{bench_dir}" in driver_line
    # the caller's own mounts must survive alongside the injected one
    assert "/host/cache:/cache" in driver_line
    # The output path is absolute rather than cwd-relative, and cwd is left
    # alone: a benchmark's prepare_script is resolved against cwd, so moving it
    # breaks `gym eval prepare` on a file that exists.
    assert f"+output_jsonl_fpath={bench_dir}/artifacts/rollouts.jsonl" in script
    assert "--container-workdir" not in script


def test_driver_job_dir_is_mounted_even_with_no_configured_mounts():
    config = SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {"hsg": {"type": "slurm", "account": "acct"}},
            "driver": {"container": "gym:latest", "benchmarks": {"gpqa": {}}},
            "job": {"output_path": "/jobs"},
        }
    )
    bench_dir = Path("/jobs/gym-job-x/gpqa")

    script = build_sbatch_script(config, "gpqa", config.driver.benchmarks["gpqa"], config.compute["hsg"], bench_dir)

    driver_line = next(line for line in script.splitlines() if "--output=logs/driver-$SLURM_JOB_ID.log" in line)
    assert f"--container-mounts={bench_dir}:{bench_dir}" in driver_line


def test_gym_install_does_not_clone_into_the_job_directory():
    """The driver's cwd is the job directory. A clone there gives Gym a second
    copy of every built-in asset, and named lookups (`--model-type
    openai_model`) then abort as ambiguous against the installed copy."""
    entrypoint = render_driver_entrypoint(repo="https://github.com/NVIDIA-NeMo/gym", ref="abc123", prepare_cmd=None)

    assert "mktemp -d /tmp/gym-install-" in entrypoint
    assert 'git clone https://github.com/NVIDIA-NeMo/gym "$GYM_SRC/gym"' in entrypoint


def test_gym_install_runs_from_the_install_root():
    """A benchmark's prepare_script is relative to cwd, and a runtime image bakes no
    Gym, so the driver has to run from the clone or `gym eval prepare` cannot find
    its own script. Safe because the clone is outside the job directory and the
    driver's output path is absolute."""
    entrypoint = render_driver_entrypoint(repo="https://github.com/NVIDIA-NeMo/gym", ref="abc123", prepare_cmd=None)

    # Assert the behaviour, not the line layout: the `cd` is chained onto the
    # checkout with && rather than standing on its own line.
    assert 'cd "$GYM_SRC/gym"' in entrypoint
    assert entrypoint.index('cd "$GYM_SRC/gym"') < entrypoint.index('exec "$@"')
    # The only `cd` is into the clone -- nothing else may move cwd.
    assert entrypoint.count("cd ") == entrypoint.count('cd "$GYM_SRC/gym"')


# ---------------------------------------------------------------------------
# driver command
# ---------------------------------------------------------------------------


def _command_config(tmp_path, command, **driver):
    return SubmitConfig.model_validate(
        {
            "services": {
                "policy": {
                    "type": "vllm",
                    "container": "img",
                    "model": "/ckpt",
                    "port": 8000,
                    "served_model_name": "super",
                    "tensor_parallel_size": 4,
                }
            },
            "compute": {
                "hsg": {
                    "type": "slurm",
                    "account": "acct",
                    "node_pools": {"gpu": {"partition": "batch", "nodes": 1, "gpus_per_node": 4}},
                }
            },
            "driver": {
                "container": "efb:latest",
                "policy_model": "policy",
                "benchmarks": {"osworld": {"command": command}},
                **driver,
            },
            "job": {"output_path": str(tmp_path / "jobs")},
        }
    )


def _render_command(tmp_path, command, **driver):
    config = _command_config(tmp_path, command, **driver)
    return build_sbatch_script(
        config,
        "osworld",
        config.driver.benchmarks["osworld"],
        config.compute["hsg"],
        tmp_path / "jobs" / "x" / "osworld",
    )


def test_a_command_replaces_the_gym_run_invocation(tmp_path):
    script = _render_command(tmp_path, "run-the-harness --flag")

    assert "GYM_CMD" not in script
    assert "gym eval run" not in script
    assert "run-the-harness --flag" in script


def test_a_command_is_told_where_to_write_and_how_to_reach_the_policy(tmp_path):
    # These replace the run arguments the command no longer receives; without them
    # the script has no way to find the served model or the job directory.
    bench_dir = tmp_path / "jobs" / "x" / "osworld"
    script = _render_command(tmp_path, "harness")

    assert f"NEMO_GYM_BENCH_DIR={bench_dir}" in script
    assert "NEMO_GYM_POLICY_BASE_URL=http://localhost:8000/v1" in script
    assert "NEMO_GYM_POLICY_MODEL_NAME=super" in script
    assert "NEMO_GYM_POLICY_API_KEY=dummy" in script


def test_a_command_keeps_the_driver_env_the_config_declared(tmp_path):
    script = _render_command(tmp_path, "harness", env={"AWS_ACCESS_KEY_ID": "lit:AKIAFAKE"})

    assert "AWS_ACCESS_KEY_ID=AKIAFAKE" in script
    assert "NEMO_GYM_BENCH_DIR=" in script


def test_a_command_still_runs_prepare_first(tmp_path):
    config = SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {"hsg": {"type": "slurm", "account": "acct"}},
            "driver": {
                "container": "efb:latest",
                "benchmarks": {
                    "osworld": {"prepare": {"config_paths": ["benchmarks/osworld/config.yaml"]}, "command": "harness"}
                },
            },
            "job": {"output_path": str(tmp_path / "jobs")},
        }
    )
    script = build_sbatch_script(
        config, "osworld", config.driver.benchmarks["osworld"], config.compute["hsg"], tmp_path / "jobs" / "osworld"
    )

    prepare_at = script.index("gym eval prepare")
    assert prepare_at < script.index("harness")


def test_a_multi_line_command_survives_quoting(tmp_path):
    command = "set -- harness --name 'a b'\nexec \"$@\""
    script = _render_command(tmp_path, command)

    assert "--name '\"'\"'a b'\"'\"'" in script
    assert 'exec "$@"' in script


def test_a_benchmark_cannot_set_both_command_and_run(tmp_path):
    with pytest.raises(ValueError, match="`command` replaces"):
        SubmitConfig.model_validate(
            {
                "services": {},
                "compute": {"hsg": {"type": "slurm", "account": "acct"}},
                "driver": {
                    "container": "efb:latest",
                    "benchmarks": {"osworld": {"command": "harness", "run": {"split": "benchmark"}}},
                },
                "job": {"output_path": str(tmp_path / "jobs")},
            }
        )


# ---------------------------------------------------------------------------
# node_pool placement
# ---------------------------------------------------------------------------


def _placement_config(tmp_path, services, pools):
    return SubmitConfig.model_validate(
        {
            "services": services,
            "compute": {"hsg": {"type": "slurm", "account": "acct", "node_pools": pools}},
            "driver": {"container": "gym:latest", "benchmarks": {"b": {"run": {}}}},
            "job": {"output_path": str(tmp_path / "jobs")},
        }
    )


def _vllm(port, pool, **extra):
    return {
        "type": "vllm",
        "container": "img",
        "model": "/ckpt",
        "port": port,
        "node_pool": pool,
        **extra,
    }


_TWO_POOLS = {
    "gpu": {"partition": "batch", "nodes": 1, "ntasks_per_node": 1, "gpus_per_node": 4},
    "aux": {"partition": "batch", "nodes": 1, "ntasks_per_node": 1, "gpus_per_node": 4},
}


def _render(tmp_path, services, pools=None):
    config = _placement_config(tmp_path, services, pools or _TWO_POOLS)
    return build_sbatch_script(
        config, "b", config.driver.benchmarks["b"], config.compute["hsg"], tmp_path / "jobs" / "b"
    )


def test_pinned_services_take_contiguous_node_ranges(tmp_path):
    # --nodelist names each pool's hosts; --relative is only a hint Slurm can move a
    # step off (seen on a cluster run: the head landed on the worker's node). Without it both steps
    # start at node 0 and the second one shares the first one's GPUs.
    script = _render(
        tmp_path,
        {
            "policy": _vllm(8000, "gpu", tensor_parallel_size=4),
            "scorer": _vllm(8001, "aux", tensor_parallel_size=4),
        },
    )
    assert 'export GYM_POOL_GPU_NODES="$(IFS=,; echo "${gym_nodes[*]:0:1}")"' in script
    assert 'export GYM_POOL_AUX_NODES="$(IFS=,; echo "${gym_nodes[*]:1:1}")"' in script
    assert '--nodelist="${GYM_POOL_GPU_NODES}" --nodes=1 --ntasks=1' in script.split("# service: policy")[1]
    assert '--nodelist="${GYM_POOL_AUX_NODES}" --nodes=1 --ntasks=1' in script.split("# service: scorer")[1]


def test_a_pinned_single_node_service_is_not_built_as_multi_node(tmp_path):
    # The job spans two nodes, but each service owns one. Sizing them by the
    # allocation total would build both as multi-node Ray deployments.
    script = _render(
        tmp_path,
        {
            "policy": _vllm(8000, "gpu", tensor_parallel_size=4),
            "scorer": _vllm(8001, "aux", tensor_parallel_size=4),
        },
    )
    assert "ray start" not in script
    assert script.count("vllm serve") == 2


def test_an_unpinned_service_keeps_the_whole_allocation(tmp_path):
    script = _render(
        tmp_path,
        {"policy": {"type": "vllm", "container": "img", "model": "/ckpt", "port": 8000, "tensor_parallel_size": 8}},
    )
    policy_line = script.split("# service: policy")[1].splitlines()[1]
    assert "--nodelist" not in policy_line
    assert "--nodes=2" in policy_line


def test_the_driver_runs_on_node_0_of_a_multi_node_job(tmp_path):
    # A multi-node policy serves its API from node 0, and the driver reaches it on
    # localhost. Left unpinned, Slurm may start the driver on any node.
    script = _render(
        tmp_path,
        {"policy": {"type": "vllm", "container": "img", "model": "/ckpt", "port": 8000, "tensor_parallel_size": 8}},
    )
    driver_line = next(line for line in script.splitlines() if "logs/driver-$SLURM_JOB_ID.log" in line)
    assert '--nodelist="${gym_nodes[0]}" --nodes=1 --ntasks=1' in driver_line
    assert 'gym_nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))' in script


def test_a_single_node_job_places_nothing(tmp_path):
    one = {"gpu": {"partition": "batch", "nodes": 1, "ntasks_per_node": 1, "gpus_per_node": 4}}
    script = _render(
        tmp_path,
        {"policy": {"type": "vllm", "container": "img", "model": "/ckpt", "port": 8000, "tensor_parallel_size": 4}},
        one,
    )
    assert "--nodelist" not in script
    assert "gym_nodes=" not in script


def test_an_unknown_node_pool_is_named(tmp_path):
    with pytest.raises(ValueError, match="node_pool 'nope' does not match any node pool"):
        _placement_config(tmp_path, {"policy": _vllm(8000, "nope", tensor_parallel_size=4)}, _TWO_POOLS)


# ---------------------------------------------------------------------------
# ray services
# ---------------------------------------------------------------------------


def test_a_ray_head_blocks_so_slurm_keeps_the_step_alive():
    # `ray start` daemonises and returns; without --block the srun step exits the
    # moment the node is up and the service is torn down again.
    command = _build_ray_command(RayServiceConfig(type="ray", container="img"))
    assert command == "ray start --block --head --port 6379"


def test_a_ray_worker_joins_the_head_and_advertises_its_pools_resources():
    service = RayServiceConfig(
        type="ray", container="img", node_pools=["aux"], num_gpus=0, resources={"aux": {"extra_gpu": 4}}
    )
    command = _build_ray_command(service, pool="aux", address_var="GYM_RAY_ADDRESS_RAY", worker=True)
    assert command == (
        'ray start --block --address "$GYM_RAY_ADDRESS_RAY" --num-gpus 0 --resources=\'{"extra_gpu": 4}\''
    )


def test_a_ray_worker_drops_the_flags_ray_allows_only_on_a_head():
    service = RayServiceConfig(
        type="ray", container="img", extra_args="--include-dashboard=false --port 1 --node-manager-port=8366"
    )
    command = _build_ray_command(service, address_var="A", worker=True)
    assert command == 'ray start --block --address "$A" --node-manager-port=8366'
    assert _build_ray_command(service).endswith("--include-dashboard=false --port 1 --node-manager-port=8366")


def test_the_old_head_and_worker_fields_are_gone():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        RayServiceConfig(type="ray", container="img", mode="worker", address="10.0.0.1:6379")


def test_a_ray_service_names_node_pools_not_a_node_pool():
    with pytest.raises(ValueError, match=r"use node_pools: \[aux\]"):
        RayServiceConfig(type="ray", container="img", node_pool="aux")


def test_ray_resources_for_a_pool_the_service_does_not_span_are_refused():
    with pytest.raises(ValueError, match="sets resources for aux, which it does not span"):
        RayServiceConfig(type="ray", container="img", node_pools=["gpu"], resources={"aux": {"extra_gpu": 4}})


def test_an_unknown_ray_node_pool_is_named(tmp_path):
    with pytest.raises(ValueError, match=r"node_pools \['nope'\] do not match any node pool"):
        _placement_config(tmp_path, {"ray": {"type": "ray", "container": "img", "node_pools": ["nope"]}}, _TWO_POOLS)


# ---------------------------------------------------------------------------
# prefill/decode disaggregation
# ---------------------------------------------------------------------------


_PD_POOLS = {
    "prefill": {"partition": "batch", "nodes": 4, "ntasks_per_node": 1, "gpus_per_node": 4},
    "decode": {"partition": "batch", "nodes": 6, "ntasks_per_node": 1, "gpus_per_node": 4},
}


def _pd_service(**overrides):
    return {
        "type": "vllm_pd",
        "container": "img",
        "model": "/ckpt",
        "served_model_name": "super35",
        "prefill": {"node_pool": "prefill", "tensor_parallel_size": 4, "number_of_instances": 4},
        "decode": {"node_pool": "decode", "tensor_parallel_size": 4, "number_of_instances": 6},
        **overrides,
    }


def _pd_services(**overrides):
    return {"policy": _pd_service(**overrides)}


def _pd_config(tmp_path, services=None, pools=None):
    return SubmitConfig.model_validate(
        {
            "services": services if services is not None else _pd_services(),
            "compute": {"hsg": {"type": "slurm", "account": "acct", "node_pools": pools or _PD_POOLS}},
            "driver": {"container": "img", "policy_model": "policy", "benchmarks": {"b": {"run": {}}}},
            "job": {"output_path": str(tmp_path / "jobs")},
        }
    )


def _pd_script(tmp_path, **kwargs):
    config = _pd_config(tmp_path, **kwargs)
    return build_sbatch_script(
        config, "b", config.driver.benchmarks["b"], config.compute["hsg"], tmp_path / "jobs" / "b"
    )


def _step(script, name):
    """The srun step a service renders to."""
    return script.split(f"# service: {name}\n")[1].split("\n\n")[0]


def test_each_tier_carries_its_own_kv_role(tmp_path):
    script = _pd_script(tmp_path)
    assert '"kv_role":"kv_producer"' in _step(script, "policy-prefill")
    assert '"kv_role":"kv_consumer"' in _step(script, "policy-decode")


def test_the_kv_transfer_settings_come_from_the_pd_service(tmp_path):
    services = _pd_services(kv_connector="OtherConnector", kv_load_failure_policy="recompute")
    script = _pd_script(tmp_path, services=services)
    for tier in ("policy-prefill", "policy-decode"):
        assert '"kv_connector":"OtherConnector"' in _step(script, tier)
        assert '"kv_load_failure_policy":"recompute"' in _step(script, tier)


def test_each_tier_coordinates_on_its_own_head(tmp_path):
    # Pointing both tiers at the allocation's head node would make prefill rank 0 and
    # decode rank 0 try to coordinate through the same address and port.
    script = _pd_script(tmp_path)
    assert "--data-parallel-address ${gym_nodes[0]} --data-parallel-rpc-port 13345" in _step(script, "policy-prefill")
    assert "--data-parallel-address ${gym_nodes[4]} --data-parallel-rpc-port 13346" in _step(script, "policy-decode")


def test_the_tiers_take_separate_node_ranges(tmp_path):
    script = _pd_script(tmp_path)
    assert 'GYM_POOL_PREFILL_NODES="$(IFS=,; echo "${gym_nodes[*]:0:4}")"' in script
    assert 'GYM_POOL_DECODE_NODES="$(IFS=,; echo "${gym_nodes[*]:4:6}")"' in script
    assert '--nodelist="${GYM_POOL_PREFILL_NODES}" --nodes=4 --ntasks=4' in script
    assert '--nodelist="${GYM_POOL_DECODE_NODES}" --nodes=6 --ntasks=6' in script
    assert "#SBATCH --nodes=10" in script


def test_the_router_runs_beside_the_driver(tmp_path):
    # The driver reaches the router on localhost, so both go to node 0.
    script = _pd_script(tmp_path)
    driver = next(line for line in script.splitlines() if "logs/driver-$SLURM_JOB_ID.log" in line)
    assert '--nodelist="${gym_nodes[0]}" --nodes=1 --ntasks=1' in _step(script, "policy")
    assert '--nodelist="${gym_nodes[0]}" --nodes=1 --ntasks=1' in driver


def test_the_router_addresses_both_tier_heads(tmp_path):
    router = _step(_pd_script(tmp_path), "policy")
    assert "vllm-router" in router and "--vllm-pd-disaggregation" in router
    assert '--prefill "http://${gym_nodes[0]}:8001"' in router
    assert '--decode "http://${gym_nodes[4]}:8002"' in router


def test_a_tier_off_the_first_node_is_probed_where_it_runs(tmp_path):
    # The probe runs on the allocation's first node; decode answers on node 4.
    script = _pd_script(tmp_path)
    assert "Waiting for policy-decode at http://${gym_nodes[4]}:8002" in script
    assert "Waiting for policy-prefill at http://localhost:8001" in script
    assert "Waiting for policy at http://localhost:8000" in script


def test_each_tier_exports_its_own_nixl_side_channel(tmp_path):
    script = _pd_script(tmp_path)
    for tier, port in (("policy-prefill", 5600), ("policy-decode", 5601)):
        assert "export VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname)" in _step(script, tier)
        assert f"export VLLM_NIXL_SIDE_CHANNEL_PORT={port}" in _step(script, tier)


def test_the_node_list_is_resolved_for_the_router(tmp_path):
    assert 'gym_nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))' in _pd_script(tmp_path)


@pytest.mark.parametrize("per_node", [False, True], ids=["data_parallel", "server_per_node"])
def test_a_pd_service_renders_the_three_service_form_it_replaced(per_node):
    # The fixtures are the job scripts the earlier form (two tier services wired by hand to a
    # `type: router` service) rendered for SWE-bench Pro's 4 prefill + 6 decode layout.
    shape = (lambda n: {}) if per_node else (lambda n: {"number_of_instances": n})
    pd = _pd_service(
        container="vllm:img",
        server_per_node=per_node,
        prefill={
            "node_pool": "prefill",
            "tensor_parallel_size": 4,
            "extra_args": "--max-model-len 131072",
            **shape(4),
        },
        decode={
            "node_pool": "decode",
            "tensor_parallel_size": 4,
            "env": {"UCX_TLS": "lit:rc,cuda_copy"},
            "pre_command": "export NCCL_DEBUG=WARN",
            **shape(6),
        },
    )
    config = SubmitConfig.model_validate(
        {
            "services": {"policy": pd},
            "compute": {"hsg": {"type": "slurm", "account": "acct", "node_pools": _PD_POOLS}},
            "driver": {"container": "img", "policy_model": "policy", "benchmarks": {"swebench_pro": {"run": {}}}},
            "job": {"output_path": "/jobs"},
        }
    )
    script = build_sbatch_script(
        config,
        "swebench_pro",
        config.driver.benchmarks["swebench_pro"],
        config.compute["hsg"],
        Path("/jobs/swebench_pro"),
    )
    fixture = "pd_swebench_pro_server_per_node.sh" if per_node else "pd_swebench_pro_data_parallel.sh"
    assert script == (Path(__file__).parent / "fixtures" / fixture).read_text()


def test_a_pd_service_deploys_as_two_tiers_and_a_router(tmp_path):
    config = _pd_config(tmp_path)
    assert list(config.deployed_services) == ["policy-prefill", "policy-decode", "policy"]
    assert list(config.services) == ["policy"]


def test_the_tiers_inherit_the_model_and_take_the_next_ports(tmp_path):
    services = _pd_services(port=9000)
    services["policy"]["decode"] |= {"port": 9100, "model": "/other", "container": "other"}
    deployed = _pd_config(tmp_path, services=services).deployed_services
    prefill, decode = deployed["policy-prefill"], deployed["policy-decode"]
    assert (prefill.model, prefill.served_model_name, prefill.container, prefill.port) == (
        "/ckpt",
        "super35",
        "img",
        9001,
    )
    assert (decode.model, decode.served_model_name, decode.container, decode.port) == (
        "/other",
        "super35",
        "other",
        9100,
    )
    assert (prefill.health_check.port, decode.health_check.port) == (9001, 9100)


def test_decode_takes_the_nixl_port_one_above_prefills(tmp_path):
    deployed = _pd_config(tmp_path, services=_pd_services(nixl_side_channel_port=7000)).deployed_services
    assert deployed["policy-prefill"].nixl_side_channel_port == 7000
    assert deployed["policy-decode"].nixl_side_channel_port == 7001


def test_decode_data_parallel_rpc_port_follows_prefills_unless_set(tmp_path):
    services = _pd_services()
    services["policy"]["prefill"]["data_parallel_rpc_port"] = 14000
    assert _pd_config(tmp_path, services=services).deployed_services["policy-decode"].data_parallel_rpc_port == 14001
    services["policy"]["decode"]["data_parallel_rpc_port"] = 15000
    assert _pd_config(tmp_path, services=services).deployed_services["policy-decode"].data_parallel_rpc_port == 15000


def test_the_policy_model_is_the_router(tmp_path):
    run = _pd_config(tmp_path).driver.benchmarks["b"].run
    assert run["policy_base_url"] == "http://localhost:8000/v1"
    assert run["policy_model_name"] == "super35"


def test_a_pd_service_is_recorded_as_written(tmp_path):
    # The resolved config written into the job dir must load back, so the tiers stay nested.
    config = _pd_config(tmp_path)
    dumped = config.model_dump(mode="json")
    assert list(dumped["services"]) == ["policy"]
    for benchmark in dumped["driver"]["benchmarks"].values():
        benchmark["run"] = {}
    reloaded = SubmitConfig.model_validate(dumped)
    assert list(reloaded.deployed_services) == ["policy-prefill", "policy-decode", "policy"]
    assert reloaded.deployed_services["policy-decode"].kv_transfer_config["kv_role"] == "kv_consumer"


def test_a_reloaded_pd_config_keeps_the_derived_tier_values_out(tmp_path):
    # Written back in, a derived value would be refused on reload or fixed at its old value.
    services = _pd_services(server_per_node=True, nixl_side_channel_port=7000)
    for tier in ("prefill", "decode"):
        services["policy"][tier]["number_of_instances"] = 1
    dumped = _pd_config(tmp_path, services=services).model_dump(mode="json")
    for tier in ("prefill", "decode"):
        derived = {"server_per_node", "nixl_side_channel_port", "kv_transfer_config"} & set(
            dumped["services"]["policy"][tier]
        )
        assert not derived
    for benchmark in dumped["driver"]["benchmarks"].values():
        benchmark["run"] = {}
    dumped["services"]["policy"]["nixl_side_channel_port"] = 8000
    decode = SubmitConfig.model_validate(dumped).deployed_services["policy-decode"]
    assert (decode.nixl_side_channel_port, decode.server_per_node) == (8001, True)


def test_a_tiers_missing_mount_is_named_by_its_path_in_the_config(tmp_path):
    services = _pd_services()
    services["policy"]["decode"]["mounts"] = [f"{tmp_path / 'nonexistent'}:/ckpt"]
    with pytest.raises(ValueError, match=r"services\.policy\.decode"):
        _validate_mounts(_pd_config(tmp_path, services=services), LocalConnection())


@pytest.mark.parametrize("tier", ["prefill", "decode"])
@pytest.mark.parametrize("field", ["nixl_side_channel_port", "server_per_node"])
def test_service_wide_settings_are_refused_on_a_tier(tmp_path, tier, field):
    services = _pd_services()
    services["policy"][tier][field] = 1
    with pytest.raises(ValueError, match=f"{field} is set on the {tier} tier. Set it on the vllm_pd service itself"):
        _pd_config(tmp_path, services=services)


def test_tiers_sharing_a_data_parallel_rpc_port_are_refused(tmp_path):
    services = _pd_services()
    services["policy"]["decode"]["data_parallel_rpc_port"] = 13345
    with pytest.raises(ValueError, match="'policy-prefill' and 'policy-decode' share data_parallel_rpc_port 13345"):
        _pd_config(tmp_path, services=services)


def test_a_tier_without_a_node_pool_is_refused(tmp_path):
    services = _pd_services()
    del services["policy"]["prefill"]["node_pool"]
    with pytest.raises(ValueError, match="'policy-prefill' is a prefill/decode tier without a node_pool"):
        _pd_config(tmp_path, services=services)


def test_a_tier_on_an_unknown_pool_is_refused(tmp_path):
    services = _pd_services()
    services["policy"]["decode"]["node_pool"] = "nope"
    with pytest.raises(ValueError, match="'policy-decode' node_pool 'nope' does not match any node pool"):
        _pd_config(tmp_path, services=services)


def test_a_missing_tier_is_refused(tmp_path):
    services = _pd_services()
    del services["policy"]["decode"]
    with pytest.raises(ValueError, match=r"services\.policy\.vllm_pd\.decode\n  Field required"):
        _pd_config(tmp_path, services=services)


def test_a_tier_name_already_taken_is_refused(tmp_path):
    services = _pd_services()
    services["policy-decode"] = {"type": "vllm", "container": "img", "model": "/j", "node_pool": "decode"}
    with pytest.raises(ValueError, match="deploys its tiers as policy-prefill, policy-decode, but services already"):
        _pd_config(tmp_path, services=services)


@pytest.mark.parametrize("field", ["kv_role", "kv_connector", "kv_load_failure_policy"])
def test_kv_settings_are_not_accepted_on_a_tier(tmp_path, field):
    # The P/D service sets each tier's role and owns the connector settings.
    services = _pd_services()
    services["policy"]["prefill"][field] = "x"
    with pytest.raises(ValueError, match=f"prefill.{field}\n  Extra inputs are not permitted"):
        _pd_config(tmp_path, services=services)


@pytest.mark.parametrize("field", ["nixl_side_channel_port", "server_per_node"])
def test_tier_only_settings_are_not_accepted_on_a_plain_vllm_service(tmp_path, field):
    services = {"policy": {"type": "vllm", "container": "img", "model": "/ckpt", field: 1}}
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _pd_config(tmp_path, services=services)


def test_a_router_away_from_the_driver_is_refused(tmp_path):
    # driver.policy_model reaches the router over localhost, and the driver runs on
    # the allocation's first node.
    with pytest.raises(ValueError, match="reaches it over localhost"):
        _pd_config(tmp_path, services=_pd_services(node_pool="decode"))


def test_an_empty_kv_config_renders_no_kv_transfer_flag():
    # `--kv-transfer-config '{}'` would start a tier with no KV role at all.
    tier = VllmPDTierConfig(
        type="vllm",
        container="img",
        model="/ckpt",
        nixl_side_channel_port=5600,
        server_per_node=False,
        kv_transfer_config={},
    )
    assert _kv_transfer_flag(tier) == ""


def test_a_plain_vllm_service_gets_no_kv_transfer_config(tmp_path):
    services = {
        "policy": {"type": "vllm", "container": "img", "model": "/ckpt", "port": 8000, "tensor_parallel_size": 4}
    }
    pools = {"gpu": {"partition": "batch", "nodes": 1, "ntasks_per_node": 1, "gpus_per_node": 4}}
    config = SubmitConfig.model_validate(
        {
            "services": services,
            "compute": {"hsg": {"type": "slurm", "account": "acct", "node_pools": pools}},
            "driver": {"container": "img", "policy_model": "policy", "benchmarks": {"b": {"run": {}}}},
            "job": {"output_path": str(tmp_path / "jobs")},
        }
    )
    script = build_sbatch_script(
        config, "b", config.driver.benchmarks["b"], config.compute["hsg"], tmp_path / "jobs" / "b"
    )
    assert "--kv-transfer-config" not in script
    assert "VLLM_NIXL_SIDE_CHANNEL" not in script
    assert "gym_nodes=" not in script


def _comet_like(tmp_path, pools=None, **ray):
    return _render(
        tmp_path,
        {
            "policy": _vllm(8000, "gpu", tensor_parallel_size=4),
            "ray": {
                "type": "ray",
                "container": "img",
                "node_pools": ["gpu", "aux"],
                "port": 6380,
                "resources": {"aux": {"extra_gpu": 4}},
                **ray,
            },
        },
        pools,
    )


def test_a_ray_service_starts_its_head_beside_the_driver(tmp_path):
    script = _comet_like(tmp_path)
    assert 'export GYM_RAY_ADDRESS_RAY="$(getent hosts ${gym_nodes[0]}' in script
    assert ':6380"' in script
    head = script.split("# service: ray\n")[1].split("\n\n")[0]
    assert '--nodelist="${gym_nodes[0]}" --nodes=1 --ntasks=1' in head
    assert '--head --port 6380 --node-ip-address "${GYM_RAY_ADDRESS_RAY%:*}"' in head
    # The head's own pool advertises nothing here, so it offers no extra_gpu.
    assert "extra_gpu" not in head


def test_every_other_spanned_node_joins_as_a_worker(tmp_path):
    pools = {**_TWO_POOLS, "aux": {**_TWO_POOLS["aux"], "nodes": 2}}
    script = _comet_like(tmp_path, pools)
    assert 'export GYM_RAY_RAY_AUX_WORKERS="$(IFS=,; echo "${gym_nodes[*]:1:2}")"' in script
    # The gpu pool's only node is the head, so it gets no worker step.
    assert "GYM_RAY_RAY_GPU_WORKERS" not in script
    workers = script.split("# service: ray_aux_workers\n")[1].split("\n\n")[0]
    assert '--nodelist="${GYM_RAY_RAY_AUX_WORKERS}" --nodes=2 --ntasks=2' in workers
    assert '--address "$GYM_RAY_ADDRESS_RAY"' in workers
    assert "extra_gpu" in workers


def test_a_ray_service_runs_every_step_in_its_one_container(tmp_path):
    script = _comet_like(tmp_path, container="ray-img:1", mounts=["/x:/x"], pre_command="setup")
    for name in ("ray", "ray_aux_workers"):
        step = script.split(f"# service: {name}\n")[1].split("\n\n")[0]
        assert "--container-image=ray-img:1" in step
        assert "--container-mounts=/x:/x" in step
        assert "setup" in step


def test_a_worker_waits_for_the_head_after_its_own_setup(tmp_path):
    script = _comet_like(tmp_path, pre_command="install-ray")
    workers = script.split("# service: ray_aux_workers\n")[1].split("\n\n")[0]
    assert workers.index("install-ray") < workers.index("until ray status") < workers.index("exec ray start")


def test_the_ray_head_follows_the_driver_to_a_later_pool(tmp_path):
    config = SubmitConfig.model_validate(
        {
            "services": {
                "policy": _vllm(8000, "aux", tensor_parallel_size=4),
                "ray": {"type": "ray", "container": "img", "node_pools": ["gpu", "aux"]},
            },
            "compute": {"hsg": {"type": "slurm", "account": "acct", "node_pools": _TWO_POOLS}},
            "driver": {"container": "gym:latest", "policy_model": "policy", "benchmarks": {"b": {"run": {}}}},
            "job": {"output_path": str(tmp_path / "jobs")},
        }
    )
    script = build_sbatch_script(config, "b", config.driver.benchmarks["b"], config.compute["hsg"], tmp_path / "b")
    assert '--nodelist="${gym_nodes[1]}"' in script.split("# service: ray\n")[1].split("\n\n")[0]
    assert 'export GYM_RAY_ADDRESS_RAY="$(getent hosts ${gym_nodes[1]}' in script
    assert 'export GYM_RAY_RAY_GPU_WORKERS="$(IFS=,; echo "${gym_nodes[*]:0:1}")"' in script
    assert "GYM_RAY_RAY_AUX_WORKERS" not in script


def test_a_single_node_ray_service_is_just_a_head(tmp_path):
    one = {"gpu": {"partition": "batch", "nodes": 1, "ntasks_per_node": 1, "gpus_per_node": 4}}
    script = _render(tmp_path, {"ray": {"type": "ray", "container": "img", "node_pools": ["gpu"]}}, one)
    assert "_workers" not in script
    assert "--nodelist" not in script
    assert 'export GYM_RAY_ADDRESS_RAY="$(getent hosts ${gym_nodes[0]}' in script


def test_a_rendered_ray_service_is_valid_bash(tmp_path):
    script = _comet_like(tmp_path, pre_command="echo 'quoted'", extra_args="--temp-dir=/tmp/ray-$SLURM_JOB_ID")
    result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_the_driver_goes_to_the_policys_node_when_services_are_pinned(tmp_path):
    # The driver reaches the policy on localhost. Unpinned, Slurm put it on the aux
    # node of a real two-node job, where neither the policy nor a local raylet was.
    config = SubmitConfig.model_validate(
        {
            "services": {
                "scorer": _vllm(8001, "gpu", tensor_parallel_size=4),
                "policy": _vllm(8000, "aux", tensor_parallel_size=4),
            },
            "compute": {"hsg": {"type": "slurm", "account": "acct", "node_pools": _TWO_POOLS}},
            "driver": {"container": "gym:latest", "policy_model": "policy", "benchmarks": {"b": {"run": {}}}},
            "job": {"output_path": str(tmp_path / "jobs")},
        }
    )
    script = build_sbatch_script(config, "b", config.driver.benchmarks["b"], config.compute["hsg"], tmp_path / "b")
    driver = next(line for line in script.splitlines() if "--output=logs/driver-$SLURM_JOB_ID.log" in line)
    assert '--nodelist="${gym_nodes[1]}" --nodes=1 --ntasks=1' in driver


def _per_node_services():
    services = _pd_services(server_per_node=True)
    for tier in ("prefill", "decode"):
        services["policy"][tier]["number_of_instances"] = 1
    return services


def test_a_per_node_tier_runs_one_plain_server_on_every_node(tmp_path):
    # One srun over the pool, each task a standalone server: no data-parallel ranks.
    script = _pd_script(tmp_path, services=_per_node_services())
    prefill = _step(script, "policy-prefill")
    assert '--nodelist="${GYM_POOL_PREFILL_NODES}" --nodes=4 --ntasks=4' in prefill
    assert "vllm serve /ckpt --port 8001 --tensor-parallel-size 4" in prefill
    assert "--data-parallel" not in prefill and "--headless" not in prefill


def test_the_router_lists_every_node_of_a_per_node_tier(tmp_path):
    router = _step(_pd_script(tmp_path, services=_per_node_services()), "policy")
    assert [f'--prefill "http://${{gym_nodes[{i}]}}:8001"' in router for i in range(4)] == [True] * 4
    assert [f'--decode "http://${{gym_nodes[{i}]}}:8002"' in router for i in range(4, 10)] == [True] * 6
    assert "gym_nodes[10]" not in router


def test_a_per_node_tier_is_sized_as_one_node(tmp_path):
    # TP4 on a 4-GPU node fills it; judged against the pool total it would warn of idle GPUs.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _pd_config(tmp_path, services=_per_node_services())


def test_server_per_node_needs_one_instance(tmp_path):
    services = _per_node_services()
    services["policy"]["prefill"]["number_of_instances"] = 4
    with pytest.raises(ValueError, match="number_of_instances must be 1"):
        _pd_config(tmp_path, services=services)


def test_an_unpinned_service_in_a_multi_node_job_runs_on_one_node(tmp_path):
    # Without a node count srun starts the router on every node, and each copy
    # tries to bind the hostname the batch script resolved on the first one.
    router = _step(_pd_script(tmp_path, services=_per_node_services()), "policy")
    assert router.count("--nodes=1 --ntasks=1") == 1


def test_the_router_answers_on_localhost(tmp_path):
    # driver.policy_model and the health probe both call http://localhost:<port>;
    # binding $(hostname) would leave nothing listening there.
    router = _step(_pd_script(tmp_path), "policy")
    assert "--host 0.0.0.0" in router and "$(hostname)" not in router


def test_the_policy_host_can_be_routable(tmp_path):
    # An agent in a remote sandbox cannot reach the driver's localhost.
    config = SubmitConfig.model_validate(
        {
            "services": _per_node_services(),
            "compute": {"hsg": {"type": "slurm", "account": "acct", "node_pools": _PD_POOLS}},
            "driver": {
                "container": "img",
                "policy_model": "policy",
                "policy_host": "${oc.env:HEAD_NODE_IP}",
                "benchmarks": {"b": {"run": {}}},
            },
            "job": {"output_path": str(tmp_path / "jobs")},
        }
    )
    assert config.driver.benchmarks["b"].run["policy_base_url"] == "http://${oc.env:HEAD_NODE_IP}:8000/v1"
    assert _pd_config(tmp_path).driver.benchmarks["b"].run["policy_base_url"] == "http://localhost:8000/v1"


# ---------------------------------------------------------------------------
# health probes
# ---------------------------------------------------------------------------


def _driver_on_aux(tmp_path, services):
    config = SubmitConfig.model_validate(
        {
            "services": {"policy": _vllm(8000, "aux", tensor_parallel_size=4), **services},
            "compute": {"hsg": {"type": "slurm", "account": "acct", "node_pools": _TWO_POOLS}},
            "driver": {"container": "gym:latest", "policy_model": "policy", "benchmarks": {"b": {"run": {}}}},
            "job": {"output_path": str(tmp_path / "jobs")},
        }
    )
    return build_sbatch_script(config, "b", config.driver.benchmarks["b"], config.compute["hsg"], tmp_path / "b")


_RAY_WITH_PROBE = {
    "type": "ray",
    "container": "img",
    "node_pools": ["gpu", "aux"],
    "health_check": {"port": 8011, "path": "/"},
}


def test_a_pinned_service_on_a_later_pool_is_probed_where_it_runs(tmp_path):
    # The probe runs on node 0; the aux pool's service answers on node 1.
    script = _render(tmp_path, {"policy": _vllm(8000, "gpu"), "scorer": _vllm(8001, "aux")})
    assert "Waiting for scorer at http://${gym_nodes[1]}:8001" in script
    assert "Waiting for policy at http://localhost:8000" in script


def test_a_ray_head_is_probed_beside_a_non_zero_driver_node(tmp_path):
    script = _driver_on_aux(tmp_path, {"ray": _RAY_WITH_PROBE})
    assert "Waiting for ray at http://${gym_nodes[1]}:8011" in script


def test_the_collector_is_probed_beside_a_non_zero_driver_node(tmp_path):
    script = _driver_on_aux(tmp_path, {})
    assert "Waiting for otel_collector at http://${gym_nodes[1]}:13133" in script


def test_services_on_node_0_are_probed_locally(tmp_path):
    script = _render(tmp_path, {"policy": _vllm(8000, "gpu", tensor_parallel_size=4), "ray": _RAY_WITH_PROBE})
    assert "Waiting for ray at http://localhost:8011" in script
    assert "Waiting for policy at http://localhost:8000" in script


def test_an_unpinned_multi_node_service_is_probed_on_node_0(tmp_path):
    # It spans the allocation and serves its API from node 0, where the probe runs.
    script = _driver_on_aux(tmp_path, {"judge": {"type": "vllm", "container": "img", "model": "/j", "port": 9000}})
    assert "Waiting for judge at http://localhost:9000" in script


def test_an_unpinned_service_is_probed_on_the_drivers_node(tmp_path):
    # The probe runs on node 0, but an unpinned single-node service runs beside the
    # driver, here on the decode pool's first node.
    config = _pd_config(tmp_path)
    config.driver.policy_model = "policy-decode"
    script = build_sbatch_script(config, "b", config.driver.benchmarks["b"], config.compute["hsg"], tmp_path / "b")
    assert '--nodelist="${gym_nodes[4]}"' in _step(script, "policy")
    assert "Waiting for policy at http://${gym_nodes[4]}:8000" in script


def test_a_ray_head_on_node_0_is_probed_locally(tmp_path):
    ray = {
        "type": "ray",
        "container": "img",
        "node_pools": ["gpu", "aux"],
        "health_check": {"port": 8011, "path": "/"},
    }
    script = _render(tmp_path, {"policy": _vllm(8000, "gpu", tensor_parallel_size=4), "ray": ray})
    assert "Waiting for ray at http://localhost:8011" in script


# ---------------------------------------------------------------------------
# GPU packing: services sharing a node get disjoint GPUs
# ---------------------------------------------------------------------------


_ONE_NODE = {"gpu": {"partition": "batch", "nodes": 1, "ntasks_per_node": 1, "gpus_per_node": 4}}


def _step(script, name):
    return script.split(f"# service: {name}\n")[1].split("\n\n")[0]


def _gpus(step):
    # Slurm resets an `env K=V srun` value, so the plan is exported inside the step's command.
    match = (
        re.search(r"export CUDA_VISIBLE_DEVICES=([^\n]*)", shlex.split(step.split("bash -c ", 1)[1])[0])
        if "bash -c " in step
        else None
    )
    return match and match.group(1)


def test_a_judge_and_a_policy_on_one_node_get_disjoint_gpus(tmp_path):
    script = _render(
        tmp_path,
        {"judge": _vllm(8001, None), "policy": _vllm(8000, None, tensor_parallel_size=2)},
        _ONE_NODE,
    )
    assert _gpus(_step(script, "judge")) == "0"
    assert _gpus(_step(script, "policy")) == "1,2"


def test_data_parallel_replicas_take_tp_times_instances(tmp_path):
    eight = {"gpu": {**_ONE_NODE["gpu"], "gpus_per_node": 8}}
    script = _render(
        tmp_path,
        {"policy": _vllm(8000, None, tensor_parallel_size=2, number_of_instances=2), "judge": _vllm(8001, None)},
        eight,
    )
    # One `vllm serve --data-parallel-size 2` puts rank r on visible devices [2r, 2r+2).
    assert _gpus(_step(script, "policy")) == "0,1,2,3"
    assert _gpus(_step(script, "judge")) == "4"


def test_a_lone_service_renders_no_cuda_visible_devices(tmp_path):
    script = _render(tmp_path, {"policy": _vllm(8000, None, tensor_parallel_size=2)}, _ONE_NODE)
    assert "CUDA_VISIBLE_DEVICES" not in script


def test_a_lone_vllm_next_to_a_gpu_ray_service_is_packed(tmp_path):
    # Otherwise Ray advertises the policy's GPUs and schedules work onto them.
    script = _render(
        tmp_path,
        {"policy": _vllm(8000, None, tensor_parallel_size=2), "ray": {"type": "ray", "container": "img"}},
        _ONE_NODE,
    )
    assert _gpus(_step(script, "policy")) == "0,1"
    assert _gpus(_step(script, "ray")) == "2,3"


def test_a_lone_vllm_next_to_a_cpu_only_ray_service_renders_nothing_new(tmp_path):
    script = _render(
        tmp_path,
        {
            "policy": _vllm(8000, None, tensor_parallel_size=2),
            "ray": {"type": "ray", "container": "img", "num_gpus": 0},
        },
        _ONE_NODE,
    )
    assert "CUDA_VISIBLE_DEVICES" not in script


def test_a_lone_ray_service_renders_nothing_new(tmp_path):
    script = _render(tmp_path, {"ray": {"type": "ray", "container": "img"}}, _ONE_NODE)
    assert "CUDA_VISIBLE_DEVICES" not in script


def test_a_manual_cuda_visible_devices_wins_and_reserves_its_gpus(tmp_path):
    script = _render(
        tmp_path,
        {
            "judge": _vllm(8001, None),
            "policy": _vllm(8000, None, tensor_parallel_size=2, env={"CUDA_VISIBLE_DEVICES": "lit:0,1"}),
        },
        _ONE_NODE,
    )
    assert _gpus(_step(script, "policy")) == "0,1"
    assert _gpus(_step(script, "judge")) == "2"


def test_overlapping_manual_gpus_are_refused(tmp_path):
    with pytest.raises(ValueError, match="Services 'a' and 'b' both set GPU 1"):
        _placement_config(
            tmp_path,
            {
                "a": _vllm(8000, None, env={"CUDA_VISIBLE_DEVICES": "lit:1"}),
                "b": _vllm(8001, None, tensor_parallel_size=2, env={"CUDA_VISIBLE_DEVICES": "lit:1,2"}),
            },
            _ONE_NODE,
        )


def test_a_manual_gpu_past_the_node_is_refused(tmp_path):
    with pytest.raises(ValueError, match="has only GPUs 0-3"):
        _placement_config(
            tmp_path,
            {"a": _vllm(8000, None, env={"CUDA_VISIBLE_DEVICES": "lit:4"}), "b": _vllm(8001, None)},
            _ONE_NODE,
        )


def test_services_that_need_more_gpus_than_the_node_has_are_refused(tmp_path):
    with pytest.raises(ValueError, match=r"need 5 GPUs per node \(judge 1, policy 4\), but it has 4") as error:
        _placement_config(
            tmp_path, {"judge": _vllm(8001, None), "policy": _vllm(8000, None, tensor_parallel_size=4)}, _ONE_NODE
        )
    assert "node_pool" in str(error.value)
    assert "share_gpus" in str(error.value)


def test_share_gpus_opts_a_service_out(tmp_path):
    script = _render(
        tmp_path,
        {
            "judge": _vllm(8001, None, share_gpus=True),
            "policy": _vllm(8000, None, tensor_parallel_size=4),
            "scorer": _vllm(8002, None, tensor_parallel_size=4, share_gpus=True),
        },
        _ONE_NODE,
    )
    assert "CUDA_VISIBLE_DEVICES" not in script


def test_an_unreadable_manual_value_is_kept_and_shares(tmp_path):
    script = _render(
        tmp_path,
        {
            "judge": _vllm(8001, None, env={"CUDA_VISIBLE_DEVICES": "runtime:CUDA_VISIBLE_DEVICES"}),
            "policy": _vllm(8000, None, tensor_parallel_size=4),
        },
        _ONE_NODE,
    )
    assert _gpus(_step(script, "judge")) == '"$GYM_CUDA_VISIBLE_DEVICES"'
    assert "CUDA_VISIBLE_DEVICES" not in _step(script, "policy")


def test_a_ray_service_gets_the_leftover_gpus(tmp_path):
    script = _render(
        tmp_path,
        {
            "judge": _vllm(8001, None),
            "policy": _vllm(8000, None, tensor_parallel_size=2),
            "ray": {"type": "ray", "container": "img"},
        },
        _ONE_NODE,
    )
    assert _gpus(_step(script, "ray")) == "3"


def test_a_ray_service_with_its_own_gpus_keeps_them(tmp_path):
    script = _render(
        tmp_path,
        {
            "judge": _vllm(8001, None),
            "policy": _vllm(8000, None, tensor_parallel_size=2),
            "ray": {"type": "ray", "container": "img", "env": {"CUDA_VISIBLE_DEVICES": "lit:0"}},
        },
        _ONE_NODE,
    )
    assert _gpus(_step(script, "ray")) == "0"
    assert _gpus(_step(script, "judge")) == "1"
    assert _gpus(_step(script, "policy")) == "2,3"


def test_pinned_pools_are_packed_per_pool(tmp_path):
    script = _render(
        tmp_path,
        {
            "policy": _vllm(8000, "gpu", tensor_parallel_size=4),
            "judge": _vllm(8001, "aux"),
            "scorer": _vllm(8002, "aux", tensor_parallel_size=2),
            "ray": {"type": "ray", "container": "img", "node_pools": ["gpu", "aux"]},
        },
    )
    # The policy fills its pool, so the ray head beside it sees no GPUs.
    assert _gpus(_step(script, "policy")) == "0,1,2,3"
    assert _gpus(_step(script, "ray")) == "''"
    assert _gpus(_step(script, "judge")) == "0"
    assert _gpus(_step(script, "scorer")) == "1,2"
    assert _gpus(_step(script, "ray_aux_workers")) == "3"


def test_multi_node_data_parallel_is_packed_per_node(tmp_path):
    pools = {"gpu": {"partition": "batch", "nodes": 2, "ntasks_per_node": 1, "gpus_per_node": 4}}
    script = _render(
        tmp_path,
        {
            "policy": _vllm(8000, "gpu", tensor_parallel_size=1, number_of_instances=4),
            "judge": _vllm(8001, "gpu", number_of_instances=2),
        },
        pools,
    )
    assert _gpus(_step(script, "policy")) == "0,1"
    assert _gpus(_step(script, "judge")) == "2"


def test_a_ray_placed_multi_node_service_leaves_its_pool_as_today(tmp_path):
    pools = {"gpu": {"partition": "batch", "nodes": 2, "ntasks_per_node": 1, "gpus_per_node": 4}}
    script = _render(
        tmp_path,
        {"policy": _vllm(8000, "gpu", tensor_parallel_size=8), "judge": _vllm(8001, "gpu", number_of_instances=2)},
        pools,
    )
    assert "CUDA_VISIBLE_DEVICES" not in script


def test_planned_gpus_are_exported_inside_the_step_not_passed_through_srun(tmp_path):
    # On a cluster, Slurm reset `env CUDA_VISIBLE_DEVICES=0 srun ...` to the step's four GPUs.
    script = _render(
        tmp_path,
        {
            "judge": _vllm(8001, None, env={"HF_HOME": "lit:/hf"}),
            "policy": _vllm(8000, None, tensor_parallel_size=2, env={"CUDA_VISIBLE_DEVICES": "lit:2,3"}),
        },
        _ONE_NODE,
    )
    for name in ("judge", "policy"):
        assert "CUDA_VISIBLE_DEVICES" not in _step(script, name).split("srun", 1)[0]
    assert _step(script, "judge").startswith("env HF_HOME=/hf srun")
    assert _gpus(_step(script, "policy")) == "2,3"


def test_a_lone_services_manual_value_is_exported_inside_the_step(tmp_path):
    # Left in the srun env, Slurm silently replaced it with every GPU of the step.
    script = _render(tmp_path, {"policy": _vllm(8000, None, env={"CUDA_VISIBLE_DEVICES": "lit:0"})}, _ONE_NODE)
    assert "CUDA_VISIBLE_DEVICES" not in _step(script, "policy").split("srun", 1)[0]
    assert _gpus(_step(script, "policy")) == "0"


def test_a_runtime_manual_value_exports_the_batch_shells_value(tmp_path):
    script = _render(
        tmp_path, {"policy": _vllm(8000, None, env={"CUDA_VISIBLE_DEVICES": "runtime:MY_GPUS"})}, _ONE_NODE
    )
    step = _step(script, "policy")
    assert step.startswith("env GYM_CUDA_VISIBLE_DEVICES=${MY_GPUS} srun")
    assert _gpus(step) == '"$GYM_CUDA_VISIBLE_DEVICES"'


def test_the_multi_instance_example_renders(tmp_path, monkeypatch):
    raw = yaml.safe_load((Path(__file__).parents[2] / "examples" / "slurm_vllm_multi_instance.yaml").read_text())
    raw["job"]["output_path"] = str(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "x")
    config = SubmitConfig.model_validate(raw)
    compute = next(iter(config.compute.values()))
    script = build_sbatch_script(config, "gpqa", config.driver.benchmarks["gpqa"], compute, tmp_path / "gpqa")
    assert "CUDA_VISIBLE_DEVICES" not in script
    assert "--data-parallel-size 4" in _step(script, "vllm_model")
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0


# ---------------------------------------------------------------------------
# nodes: auto
# ---------------------------------------------------------------------------


def _auto_pool(gpus_per_node=4):
    return {"gpu": {"partition": "batch", "nodes": "auto", "ntasks_per_node": 1, "gpus_per_node": gpus_per_node}}


def _auto_config(tmp_path, services, pools=None, policy=None):
    return SubmitConfig.model_validate(
        {
            "services": services,
            "compute": {"hsg": {"type": "slurm", "account": "acct", "node_pools": pools or _auto_pool()}},
            "driver": {"container": "gym:latest", "policy_model": policy, "benchmarks": {"b": {"run": {}}}},
            "job": {"output_path": str(tmp_path / "jobs")},
        }
    )


def _auto_render(tmp_path, services, pools=None, policy=None):
    config = _auto_config(tmp_path, services, pools, policy)
    return config, build_sbatch_script(
        config, "b", config.driver.benchmarks["b"], config.compute["hsg"], tmp_path / "jobs" / "b"
    )


def test_auto_packs_a_judge_and_a_two_gpu_model_onto_one_node(tmp_path):
    services = {"judge": _vllm(8001, None), "policy": _vllm(8000, None, tensor_parallel_size=2)}
    config, script = _auto_render(tmp_path, services, policy="policy")
    assert config.compute["hsg"].node_pools["gpu"].nodes == 1
    assert "#SBATCH --nodes=1" in script
    # GPU indices follow declaration order on a node.
    assert _gpus(_step(script, "judge")) == "0"
    assert _gpus(_step(script, "policy")) == "1,2"
    # Exactly what the same config with an explicit `nodes: 1` renders.
    _, explicit = _auto_render(tmp_path, services, {"gpu": {**_auto_pool()["gpu"], "nodes": 1}}, policy="policy")
    assert script == explicit


def test_auto_gives_a_service_bigger_than_a_node_whole_nodes_of_its_own(tmp_path):
    # 6 GPUs on 4-GPU nodes: ceil(6 / 4) = 2 whole nodes, nothing else on them.
    config, script = _auto_render(
        tmp_path,
        {
            "judge": _vllm(8001, None, tensor_parallel_size=4),
            "policy": _vllm(8000, None, number_of_instances=6),
        },
    )
    pools = config.compute["hsg"].node_pools
    assert {name: pool.nodes for name, pool in pools.items()} == {"gpu-0": 2, "gpu-1": 1}
    assert config.services["policy"].node_pool == "gpu-0"
    assert config.services["judge"].node_pool == "gpu-1"
    assert "#SBATCH --nodes=3" in script
    assert '--nodelist="${GYM_POOL_GPU_0_NODES}" --nodes=2 --ntasks=2' in _step(script, "policy")
    assert '--nodelist="${GYM_POOL_GPU_1_NODES}" --nodes=1 --ntasks=1' in _step(script, "judge")
    assert "--data-parallel-size-local 3" in _step(script, "policy")
    # Each is alone on its nodes, so neither is narrowed.
    assert "CUDA_VISIBLE_DEVICES" not in script


def test_auto_packs_biggest_first_into_the_first_node_with_room(tmp_path):
    config = _auto_config(
        tmp_path,
        {
            "a": _vllm(8001, None),
            "b": _vllm(8002, None, tensor_parallel_size=3),
            "c": _vllm(8003, None, tensor_parallel_size=2),
            "d": _vllm(8004, None, tensor_parallel_size=2),
        },
    )
    placed = {name: service.node_pool for name, service in config.services.items()}
    assert placed == {"b": "gpu-0", "c": "gpu-1", "d": "gpu-1", "a": "gpu-0"}


def test_auto_puts_the_policy_on_node_0_beside_the_driver(tmp_path):
    config, script = _auto_render(
        tmp_path,
        {"scorer": _vllm(8001, None, tensor_parallel_size=4), "policy": _vllm(8000, None)},
        policy="policy",
    )
    assert config.services["policy"].node_pool == "gpu-0"
    assert config.services["scorer"].node_pool == "gpu-1"
    driver = next(line for line in script.splitlines() if "--output=logs/driver-$SLURM_JOB_ID.log" in line)
    assert '--nodelist="${gym_nodes[0]}"' in driver


def test_explicit_node_counts_are_not_touched(tmp_path):
    config = _placement_config(
        tmp_path,
        {"policy": _vllm(8000, "gpu", tensor_parallel_size=4), "scorer": _vllm(8001, "aux", tensor_parallel_size=4)},
        _TWO_POOLS,
    )
    assert {name: pool.nodes for name, pool in config.compute["hsg"].node_pools.items()} == {"gpu": 1, "aux": 1}
    assert config.services["policy"].node_pool == "gpu"


def test_auto_gives_the_leftovers_to_ray_on_every_node(tmp_path):
    config, script = _auto_render(
        tmp_path,
        {
            "policy": _vllm(8000, None, tensor_parallel_size=3),
            "judge": _vllm(8001, None, tensor_parallel_size=2),
            "ray": {"type": "ray", "container": "img", "node_pools": ["gpu"], "resources": {"gpu": {"x": 1}}},
        },
        policy="policy",
    )
    ray = config.services["ray"]
    assert ray.node_pools == ["gpu-0", "gpu-1"]
    assert set(ray.resources) == {"gpu-0", "gpu-1"}
    assert _gpus(_step(script, "policy")) == "0,1,2"
    assert _gpus(_step(script, "ray")) == "3"
    assert _gpus(_step(script, "judge")) == "0,1"
    assert _gpus(_step(script, "ray_gpu-1_workers")) == "2,3"


def test_auto_keeps_a_manual_cuda_visible_devices(tmp_path):
    config, script = _auto_render(
        tmp_path,
        {
            "policy": _vllm(8000, None, tensor_parallel_size=2),
            "judge": _vllm(8001, None, env={"CUDA_VISIBLE_DEVICES": "lit:0"}),
        },
        policy="policy",
    )
    assert config.compute["hsg"].node_pools["gpu"].nodes == 1
    assert _gpus(_step(script, "judge")) == "0"
    assert _gpus(_step(script, "policy")) == "1,2"


def test_auto_without_gpus_per_node_is_refused(tmp_path):
    pools = {"gpu": {"partition": "batch", "nodes": "auto"}}
    with pytest.raises(ValueError, match="nodes: auto but no gpus_per_node"):
        _auto_config(tmp_path, {"policy": _vllm(8000, None)}, pools)


def test_auto_with_no_vllm_service_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no vLLM service runs on it"):
        _auto_config(tmp_path, {"ray": {"type": "ray", "container": "img", "node_pools": ["gpu"]}})


def test_auto_puts_services_with_clashing_manual_gpus_on_different_nodes(tmp_path):
    config = _auto_config(
        tmp_path,
        {
            "a": _vllm(8001, None, env={"CUDA_VISIBLE_DEVICES": "lit:0"}),
            "b": _vllm(8002, None, env={"CUDA_VISIBLE_DEVICES": "lit:0"}),
        },
    )
    assert config.services["a"].node_pool == "gpu-0"
    assert config.services["b"].node_pool == "gpu-1"


# ---------------------------------------------------------------------------
# GPU packing and nodes: auto beside vllm_pd
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("server_per_node", [False, True])
def test_a_pd_config_renders_the_same_with_and_without_gpu_packing(tmp_path, monkeypatch, server_per_node):
    # A tier owns its nodes: nothing beside it is packed, whatever else shares the pool.
    services = _pd_services(server_per_node=server_per_node)
    if server_per_node:
        for tier in ("prefill", "decode"):
            services["policy"][tier]["number_of_instances"] = 1
    services["judge"] = _vllm(8100, "decode")
    services["ray"] = {"type": "ray", "container": "img", "node_pools": ["prefill"]}
    packed = _pd_script(tmp_path, services=services)
    monkeypatch.setattr(slurm_script_module, "plan_gpus", lambda config: {})
    assert _pd_script(tmp_path, services=services) == packed
    assert "CUDA_VISIBLE_DEVICES" not in packed


def test_auto_refuses_a_pool_a_pd_tier_uses(tmp_path):
    pools = {**_PD_POOLS, "decode": {**_PD_POOLS["decode"], "nodes": "auto"}}
    with pytest.raises(ValueError, match="vllm_pd service 'policy' uses it"):
        _pd_config(tmp_path, pools=pools)


def test_a_pd_router_beside_a_lone_vllm_takes_no_gpus(tmp_path):
    services = {**_pd_services(node_pool="front"), "scorer": _vllm(8200, "front")}
    pools = {"front": {"partition": "batch", "nodes": 1, "ntasks_per_node": 1, "gpus_per_node": 4}, **_PD_POOLS}
    assert "CUDA_VISIBLE_DEVICES" not in _pd_script(tmp_path, services=services, pools=pools)


def test_auto_probes_a_service_it_moves_to_a_second_node_there(tmp_path):
    _, script = _auto_render(
        tmp_path,
        {"policy": _vllm(8000, None, tensor_parallel_size=4), "judge": _vllm(8001, None)},
        policy="policy",
    )
    assert "Waiting for policy at http://localhost:8000" in script
    assert "Waiting for judge at http://${gym_nodes[1]}:8001" in script
