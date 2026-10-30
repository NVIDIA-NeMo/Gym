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
import shutil
import subprocess
from pathlib import Path

import pytest

from nemo_gym.observability.service_declaration import (
    NOT_PLACED_REASON,
    ServiceDeclaration,
    record_service_declaration,
)
from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors.service_declaration import (
    GPU_LISTING_FAILED_REASON,
    GPU_LISTING_TIMED_OUT_REASON,
    GPUS_UNLISTED_REASON,
    NVIDIA_SMI_FAILED_REASON,
)
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script


needs_shell = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("awk") is None or shutil.which("timeout") is None,
    reason="the job-start step needs bash, awk and timeout",
)

JOB_ID = "4242"
RUN_ID = "gym-job-20261022T100000Z-abc123"


# ---------------------------------------------------------------------------
# what a Slurm job declares at start
# ---------------------------------------------------------------------------


def _pool(nodes, gpus=8):
    return {"partition": "batch", "nodes": nodes, "ntasks_per_node": 1, "gpus_per_node": gpus}


def _vllm(**fields):
    return {"type": "vllm", "container": "vllm:img", "model": "/ckpt", **fields}


def _pd(**fields):
    return {
        "type": "vllm_pd",
        "container": "vllm:img",
        "model": "/ckpt",
        "prefill": {"node_pool": "prefill", "tensor_parallel_size": 4, "number_of_instances": 4},
        "decode": {"node_pool": "decode", "tensor_parallel_size": 4, "number_of_instances": 6},
        **fields,
    }


_PD_POOLS = {"prefill": _pool(4, gpus=4), "decode": _pool(6, gpus=4)}
_PD_PER_NODE = {
    "prefill": {"node_pool": "prefill", "tensor_parallel_size": 4},
    "decode": {"node_pool": "decode", "tensor_parallel_size": 4},
}

# (gym_service, kind, pool_role, node_indices, gpus_per_node, ray_serve); None means absent from the file.
PLACEMENT_CASES = [
    pytest.param(
        {"policy": _vllm(tensor_parallel_size=2, number_of_instances=4)},
        {"gpu": _pool(1)},
        [("policy", "vllm", None, [0], 8, False)],
        id="single_node_data_parallel",
    ),
    pytest.param(
        {"policy": _pd()},
        _PD_POOLS,
        [
            ("policy-prefill", "vllm_pd_tier", "prefill", [0, 1, 2, 3], 4, False),
            ("policy-decode", "vllm_pd_tier", "decode", [4, 5, 6, 7, 8, 9], 4, False),
            ("policy", "vllm_pd_router", "router", [0], 0, None),
        ],
        id="prefill_decode_data_parallel_across_nodes",
    ),
    pytest.param(
        {"policy": _pd(server_per_node=True, **_PD_PER_NODE)},
        _PD_POOLS,
        [
            ("policy-prefill", "vllm_pd_tier", "prefill", [0, 1, 2, 3], 4, False),
            ("policy-decode", "vllm_pd_tier", "decode", [4, 5, 6, 7, 8, 9], 4, False),
            ("policy", "vllm_pd_router", "router", [0], 0, None),
        ],
        id="prefill_decode_server_per_node",
    ),
    pytest.param(
        {"policy": _vllm(tensor_parallel_size=8, pipeline_parallel_size=2)},
        {"gpu": _pool(2)},
        [("policy", "vllm", None, [0, 1], None, False)],
        id="one_replica_across_nodes_leaves_the_split_to_ray",
    ),
    pytest.param(
        {"policy": _vllm(tensor_parallel_size=4, number_of_instances=4)},
        {"gpu": _pool(2)},
        [("policy", "vllm", None, [0, 1], 8, False)],
        id="data_parallel_replicas_split_evenly_across_nodes",
    ),
    pytest.param(
        {"policy": _vllm(tensor_parallel_size=8, number_of_instances=2, use_ray_serve=True)},
        {"gpu": _pool(2)},
        [("policy", "vllm", None, [0, 1], None, True)],
        id="ray_serve_gateway_leaves_the_split_to_ray",
    ),
    pytest.param(
        {
            "policy": _vllm(node_pool="gpu", tensor_parallel_size=8),
            "judge": _vllm(node_pool="aux", port=8001, number_of_instances=8),
        },
        {"gpu": _pool(1), "aux": _pool(1)},
        [("policy", "vllm", None, [0], 8, False), ("judge", "vllm", None, [1], 8, False)],
        id="judge_pinned_to_its_own_node",
    ),
]


@needs_shell
@pytest.mark.parametrize("services, pools, expected", PLACEMENT_CASES)
def test_a_slurm_job_declares_where_each_model_service_runs(tmp_path, services, pools, expected):
    declaration = _run_job_start(tmp_path, services, pools)

    assert declaration.status == "available"
    assert (declaration.run_id, declaration.benchmark, declaration.slurm_job_id) == (RUN_ID, "bench", JOB_ID)
    assert [
        (s.gym_service, s.kind, s.pool_role, s.node_indices, s.gpus_per_node, s.ray_serve)
        for s in declaration.services
    ] == expected
    assert all(s.service == f"gym-vllm/{s.gym_service}" for s in declaration.services)
    used = sorted({i for s in declaration.services for i in s.node_indices})
    assert [(n.index, n.hostname) for n in declaration.nodes] == [(i, f"n{i}") for i in used]


@needs_shell
def test_a_slurm_job_declares_the_parallel_layout_it_knows(tmp_path):
    declaration = _run_job_start(tmp_path, {"policy": _pd()}, _PD_POOLS)
    prefill, decode, router = declaration.services
    assert (prefill.tensor_parallel_size, prefill.pipeline_parallel_size, prefill.data_parallel_size) == (4, 1, 4)
    assert (decode.tensor_parallel_size, decode.data_parallel_size, decode.server_per_node) == (4, 6, False)
    assert (prefill.node_pool, decode.node_pool, prefill.port, decode.port) == ("prefill", "decode", 8001, 8002)
    # The router holds no GPU (a known zero) and has no parallel layout (absent, not zero).
    raw = (tmp_path / RUN_ID / "bench" / "artifacts" / "rollouts_service_declaration.json").read_text()
    assert '"tensor_parallel_size"' not in raw.split('"gym_service":"policy"')[1]
    assert router.tensor_parallel_size is None and router.gpus_per_node == 0
    # The next job of a resume chain rewrites the run folder's file; this job's copy stays in logs/.
    assert (tmp_path / "logs" / f"service_declaration-{JOB_ID}.json").read_text() == raw


_TWO_NODES = {"gpu": _pool(1), "aux": _pool(1)}
GPU_TABLE_CASES = [
    pytest.param(
        {"policy": _vllm(node_pool="gpu"), "judge": _vllm(node_pool="aux", port=8001)},
        _TWO_NODES,
        {"n0": "0, GPU-aaaa-0\n1, GPU-aaaa-1", "n1": "0, GPU-bbbb-0"},
        [
            ("GPU-aaaa-0", 0, 0, ["gym-vllm/policy"]),
            ("GPU-aaaa-1", 0, 1, ["gym-vllm/policy"]),
            ("GPU-bbbb-0", 1, 0, ["gym-vllm/judge"]),
        ],
        {},
        id="one_service_per_node",
    ),
    pytest.param(
        {"policy": _vllm(tensor_parallel_size=2), "judge": _vllm(port=8001, tensor_parallel_size=2)},
        {"gpu": _pool(1)},
        {"n0": "0, GPU-aaaa-0"},
        [("GPU-aaaa-0", 0, 0, ["gym-vllm/policy", "gym-vllm/judge"])],
        {},
        id="services_sharing_a_node_are_both_listed",
    ),
    pytest.param(
        {"policy": _vllm(node_pool="gpu"), "judge": _vllm(node_pool="aux", port=8001)},
        _TWO_NODES,
        {"n0": "0, GPU-aaaa-0"},
        [("GPU-aaaa-0", 0, 0, ["gym-vllm/policy"])],
        {1: f"{NVIDIA_SMI_FAILED_REASON} 9"},
        id="a_node_whose_nvidia_smi_fails_says_how",
    ),
    pytest.param(
        {"policy": _vllm(node_pool="gpu"), "judge": _vllm(node_pool="aux", port=8001)},
        _TWO_NODES,
        {"n0": "0, GPU-aaaa-0", "n1": ""},
        [("GPU-aaaa-0", 0, 0, ["gym-vllm/policy"])],
        {1: GPUS_UNLISTED_REASON},
        id="a_node_whose_nvidia_smi_lists_nothing_says_why",
    ),
    pytest.param(
        {"policy": _vllm(node_pool="gpu")},
        {"gpu": _pool(1)},
        {"n0": "Failed to initialize NVML: Driver/library version mismatch\n0, GPU-aaaa-0\n1, not-a-uuid"},
        [("GPU-aaaa-0", 0, 0, ["gym-vllm/policy"])],
        {},
        id="stray_output_never_becomes_a_gpu",
    ),
]


@needs_shell
@pytest.mark.parametrize(
    "srun_exit, reason",
    [
        pytest.param(124, GPU_LISTING_TIMED_OUT_REASON, id="the_listing_step_timed_out"),
        pytest.param(1, f"{GPU_LISTING_FAILED_REASON} 1", id="the_listing_step_failed"),
    ],
)
def test_a_failed_listing_step_says_why_and_never_fails_the_job(tmp_path, srun_exit, reason):
    """Without a node's own report, the step's exit status says why; a node's nvidia-smi status would come first."""
    declaration = _run_job_start(
        tmp_path, {"policy": _vllm(node_pool="gpu")}, {"gpu": _pool(1)}, listings={"n0": ""}, srun_exit=srun_exit
    )

    assert declaration.status == "available" and declaration.gpus == []
    assert [(node.gpus_observed, node.gpus_unavailable_reason) for node in declaration.nodes] == [(False, reason)]


@needs_shell
@pytest.mark.parametrize("services, pools, listings, expected_gpus, unobserved", GPU_TABLE_CASES)
def test_a_slurm_job_maps_each_gpu_uuid_to_the_services_on_its_node(
    tmp_path, services, pools, listings, expected_gpus, unobserved
):
    declaration = _run_job_start(tmp_path, services, pools, listings=listings)

    assert [(g.gpu_uuid, g.node_index, g.gpu_index, g.services) for g in declaration.gpus] == expected_gpus
    assert {n.index: n.gpus_unavailable_reason for n in declaration.nodes if not n.gpus_observed} == unobserved


@needs_shell
def test_the_router_node_lists_its_gpus_without_the_router(tmp_path):
    declaration = _run_job_start(tmp_path, {"policy": _pd()}, _PD_POOLS, listings={"n0": "0, GPU-aaaa-0"})
    assert declaration.nodes[0].services == ["gym-vllm/policy-prefill", "gym-vllm/policy"]
    assert declaration.gpus[0].services == ["gym-vllm/policy-prefill"]


def test_a_job_without_a_model_service_declares_nothing_at_start(tmp_path):
    script = _script(tmp_path, {"head": {"type": "ray", "container": "ray:img"}}, {"gpu": _pool(1)})
    assert "service declaration" not in script and "nvidia-smi" not in script


def test_a_single_node_job_lists_its_gpus_on_the_batch_host_without_placing_anything(tmp_path):
    script = _script(tmp_path, {"policy": _vllm()}, {"gpu": _pool(1)})
    block = _declaration_block(script)
    assert "--nodelist" not in block and "gym_nodes" not in block
    assert '_gym_decl_node 0 "$SLURMD_NODENAME"' in block


# ---------------------------------------------------------------------------
# what a run records in its run folder
# ---------------------------------------------------------------------------


def _available(job_id):
    """A populated declaration, so a case that drops the service, node or GPU mappings fails."""
    return json.dumps(
        {
            "schema_version": 1,
            "status": "available",
            "slurm_job_id": job_id,
            "services": [
                {"service": "model/policy", "gym_service": "policy", "kind": "vllm", "node_indices": [0], "port": 8000}
            ],
            "nodes": [{"index": 0, "hostname": "node-a", "services": ["model/policy"], "gpus_observed": True}],
            "gpus": [{"gpu_uuid": "GPU-0abc", "node_index": 0, "gpu_index": 0, "services": ["model/policy"]}],
        }
    )


RECORD_CASES = [
    pytest.param(None, None, "unavailable", NOT_PLACED_REASON, id="local_run_without_placement"),
    pytest.param(None, JOB_ID, "unavailable", NOT_PLACED_REASON, id="slurm_run_not_placed_by_submit"),
    pytest.param(_available(JOB_ID), JOB_ID, "available", None, id="this_jobs_declaration_is_kept"),
    pytest.param(
        _available("1111"),
        JOB_ID,
        "unavailable",
        f"The service declaration in the run folder belongs to Slurm job 1111, not to this run (Slurm job {JOB_ID}).",
        id="another_jobs_declaration_is_replaced",
    ),
    pytest.param(
        _available(JOB_ID),
        None,
        "unavailable",
        f"The service declaration in the run folder belongs to Slurm job {JOB_ID}, not to this run (Slurm job none).",
        id="a_reused_run_folder_outside_slurm_is_replaced",
    ),
    pytest.param("{not json", JOB_ID, "unavailable", NOT_PLACED_REASON, id="an_unreadable_declaration_is_replaced"),
]


@pytest.mark.parametrize("existing, job_id, status, reason", RECORD_CASES)
def test_a_run_records_its_service_declaration(tmp_path, monkeypatch, existing, job_id, status, reason):
    output_fpath = tmp_path / "rollouts.jsonl"
    path = tmp_path / "rollouts_service_declaration.json"
    if existing is not None:
        path.write_text(existing)
    if job_id is None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_ID", job_id)

    returned = record_service_declaration(output_fpath)

    on_disk = ServiceDeclaration.model_validate_json(path.read_bytes())
    assert on_disk == returned
    assert (on_disk.status, on_disk.reason) == (status, reason)
    if status == "available":
        assert path.read_text() == existing, "this job's declaration is kept byte for byte"
    else:
        assert on_disk.services == [] and on_disk.gpus == []


def test_an_unwritable_run_folder_does_not_fail_the_run(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    declaration = record_service_declaration(tmp_path / "missing-dir" / "rollouts.jsonl")
    assert declaration.status == "unavailable"
    assert "Could not write the service declaration" in caplog.text


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _script(tmp_path, services, pools):
    config = SubmitConfig.model_validate(
        {
            "services": services,
            "compute": {"hsg": {"type": "slurm", "account": "acct", "node_pools": pools}},
            "driver": {"container": "gym:img", "benchmarks": {"bench": {"run": {}}}},
            "job": {"output_path": str(tmp_path)},
        }
    )
    return build_sbatch_script(
        config, "bench", config.driver.benchmarks["bench"], config.compute["hsg"], tmp_path / RUN_ID / "bench"
    )


def _declaration_block(script):
    return "# service declaration" + script.split("# service declaration")[1].split("\n\n")[0]


def _run_job_start(tmp_path, services, pools, listings=None, srun_exit=None):
    """Run the script's declaration step against a fake srun and nvidia-smi, and read what it wrote.

    The fake srun runs the listing command once per `--nodelist` host (or once on the batch host),
    with SLURMD_NODENAME set as Slurm would; the fake nvidia-smi prints `listings[host]`, or fails.
    """
    total_nodes = sum(pool["nodes"] for pool in pools.values())
    listings = listings if listings is not None else {f"n{i}": f"0, GPU-{i:04d}" for i in range(total_nodes)}
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    listing_dir = tmp_path / "listings"
    listing_dir.mkdir()
    for host, text in listings.items():
        (listing_dir / host).write_text(text + "\n")
    _executable(
        bin_dir / "srun",
        'nodes="$SLURMD_NODENAME"\n'
        'while [ $# -gt 0 ]; do case "$1" in --nodelist=*) nodes="${1#--nodelist=}";; bash) cmd="$3"; break;; esac; '
        "shift; done\n"
        'IFS=, read -ra hosts <<< "$nodes"\n'
        'for host in "${hosts[@]}"; do SLURMD_NODENAME="$host" bash -c "$cmd"; done\n'
        + (f"exit {srun_exit}\n" if srun_exit is not None else ""),
    )
    _executable(bin_dir / "nvidia-smi", f'cat "{listing_dir}/$SLURMD_NODENAME" 2>/dev/null || exit 9\n')
    (tmp_path / "logs").mkdir()

    hosts = " ".join(f"n{i}" for i in range(total_nodes))
    block = f"gym_nodes=({hosts})\n{_declaration_block(_script(tmp_path, services, pools))}"
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "SLURM_JOB_ID": JOB_ID}
    subprocess.run(["bash", "-c", block], env={**env, "SLURMD_NODENAME": "n0"}, cwd=tmp_path, check=True)
    path = tmp_path / RUN_ID / "bench" / "artifacts" / "rollouts_service_declaration.json"
    return ServiceDeclaration.model_validate_json(path.read_bytes())


def _executable(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)
