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
"""Golden `job.sh` scripts for every shipped Slurm example and the shapes the examples do not cover.

The placement work moves where sizing and node assignment happen, and must not change what a job runs.
Each case renders one benchmark's script and compares it to `fixtures/job_scripts/<case>.sh`. After a
deliberate change to the rendered script, regenerate with `GYM_UPDATE_GOLDEN=1 pytest <this file>` and review the diff.
"""

import os
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script


_REPO = Path(__file__).parents[2]
_GOLDEN = Path(__file__).parent / "fixtures" / "job_scripts"

_EXAMPLES = [
    "slurm_vllm_1IN",
    "slurm_vllm_multi_instance",
    "slurm_vllm_ray_multi_node",
    "slurm_vllm_ray_multi_instance_multi_node",
]

_BENCHMARK_DRIVER = {"container": "img", "benchmarks": {"bench": {"run": {}}}}


def _slurm(*pools: tuple[str, int, int]) -> dict[str, Any]:
    node_pools = {
        name: {"partition": "batch", "nodes": nodes, "ntasks_per_node": 1, "gpus_per_node": gpus}
        for name, nodes, gpus in pools
    }
    return {"cluster": {"type": "slurm", "account": "acct", "node_pools": node_pools}}


def _vllm(**fields: Any) -> dict[str, Any]:
    return {"type": "vllm", "container": "vllm:img", "model": "org/model", **fields}


_INLINE: dict[str, dict[str, Any]] = {
    "target_and_judge_in_one_pool": {
        "compute": _slurm(("gpu", 2, 8)),
        "services": {
            "target": _vllm(tensor_parallel_size=4, number_of_instances=2, port=8000),
            "judge": _vllm(model="org/judge", tensor_parallel_size=2, port=8001),
        },
        "driver": {**_BENCHMARK_DRIVER, "policy_model": "target"},
    },
    "judge_pinned_to_its_own_pool": {
        "compute": _slurm(("main", 1, 8), ("judge", 1, 8)),
        "services": {
            "target": _vllm(tensor_parallel_size=8, port=8000),
            "judge": _vllm(model="org/judge", tensor_parallel_size=2, node_pool="judge", port=8001),
        },
        "driver": {**_BENCHMARK_DRIVER, "policy_model": "target"},
    },
    "pipeline_parallel_across_two_nodes": {
        "compute": _slurm(("gpu", 2, 8)),
        "services": {"target": _vllm(tensor_parallel_size=8, pipeline_parallel_size=2, port=8000)},
        "driver": {**_BENCHMARK_DRIVER, "policy_model": "target"},
    },
}


@pytest.fixture(autouse=True)
def _fixed_host_environment(monkeypatch):
    monkeypatch.setenv("TP", "4")
    monkeypatch.setenv("HF_TOKEN", "hf-golden")


def _render_all(raw: dict[str, Any]) -> dict[str, str]:
    config = SubmitConfig.model_validate(raw)
    return {
        f"{compute_name}-{bench_name}": build_sbatch_script(
            config, bench_name, bench, compute, Path("/jobs") / bench_name
        )
        for compute_name, compute in config.compute.items()
        for bench_name, bench in config.driver.benchmarks.items()
    }


def _check_golden(case: str, scripts: dict[str, str]) -> None:
    for suffix, script in scripts.items():
        path = _GOLDEN / f"{case}__{suffix}.sh"
        if os.environ.get("GYM_UPDATE_GOLDEN"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(script)
        assert script == path.read_text(), f"{path.name} differs; regenerate with GYM_UPDATE_GOLDEN=1 and review"


@pytest.mark.parametrize("example", _EXAMPLES)
def test_a_shipped_example_renders_its_recorded_script(example):
    raw = OmegaConf.to_container(OmegaConf.load(_REPO / "examples" / f"{example}.yaml"), resolve=True)
    scripts = _render_all(raw)
    assert scripts
    _check_golden(example, scripts)


@pytest.mark.parametrize("case", list(_INLINE))
def test_an_uncovered_shape_renders_its_recorded_script(case):
    raw = {**_INLINE[case], "job": {"output_path": "/jobs"}}
    scripts = _render_all(raw)
    assert scripts
    _check_golden(case, scripts)


@pytest.mark.parametrize("example", _EXAMPLES)
def test_rendering_a_script_twice_gives_the_same_text(example):
    raw = OmegaConf.to_container(OmegaConf.load(_REPO / "examples" / f"{example}.yaml"), resolve=True)
    assert _render_all(raw) == _render_all(raw)
