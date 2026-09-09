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

import socket

import pytest

from nemo_gym.orchestration.ray_serve_gateway import (
    build_instance_command,
    free_local_port,
    max_replicas_per_node,
    parse_args,
)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_required_fields():
    args = parse_args(["--model", "org/model", "--port", "8000"])
    assert args.model == "org/model"
    assert args.port == 8000
    assert args.tensor_parallel_size == 1
    assert args.pipeline_parallel_size == 1
    assert args.number_of_instances == 1
    assert args.trust_remote_code is False


def test_parse_args_all_fields():
    args = parse_args(
        [
            "--model",
            "org/model",
            "--port",
            "9000",
            "--tensor-parallel-size",
            "8",
            "--pipeline-parallel-size",
            "2",
            "--number-of-instances",
            "4",
            "--trust-remote-code",
        ]
    )
    assert args.tensor_parallel_size == 8
    assert args.pipeline_parallel_size == 2
    assert args.number_of_instances == 4
    assert args.trust_remote_code is True


def test_parse_args_missing_required_raises():
    with pytest.raises(SystemExit):
        parse_args(["--port", "8000"])


def test_parse_args_accepts_gpus_per_node_for_caller_compatibility():
    # Unused by this module's own scheduling logic (Ray Serve's max_replicas_per_node plus vLLM's
    # own Ray executor handle node placement without needing it) - just needs to not error, since
    # _build_vllm_ray_serve_command passes it unconditionally.
    args = parse_args(["--model", "org/model", "--port", "8000", "--gpus-per-node", "8"])
    assert args.gpus_per_node == 8


# ---------------------------------------------------------------------------
# free_local_port
# ---------------------------------------------------------------------------


def test_free_local_port_returns_a_usable_port():
    port = free_local_port()
    assert 0 < port < 65536
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", port))


def test_free_local_port_returns_distinct_ports_across_calls():
    # Not a strict guarantee (the OS could theoretically reuse one right away), but with both
    # sockets closed before the next bind, collisions are practically never observed - this is
    # what protects colocated replicas (see max_replicas_per_node) from binding the same port.
    ports = {free_local_port() for _ in range(20)}
    assert len(ports) == 20


# ---------------------------------------------------------------------------
# max_replicas_per_node
# ---------------------------------------------------------------------------


def test_max_replicas_per_node_none_without_gpus_per_node_info():
    assert max_replicas_per_node(tensor_parallel_size=1, pipeline_parallel_size=1, gpus_per_node=None) is None


def test_max_replicas_per_node_allows_multiple_instances_to_share_a_node():
    # TP2 instances comfortably share one 8-GPU node - up to 4 of them.
    assert max_replicas_per_node(tensor_parallel_size=2, pipeline_parallel_size=1, gpus_per_node=8) == 4


def test_max_replicas_per_node_one_when_footprint_exactly_fills_a_node():
    # TP8 fills the whole 8-GPU node - no room for a second instance's driver there.
    assert max_replicas_per_node(tensor_parallel_size=8, pipeline_parallel_size=1, gpus_per_node=8) == 1


def test_max_replicas_per_node_one_when_footprint_exceeds_a_node():
    # TP8 x PP2 = 16 GPUs/instance, spans 2 nodes - no other instance's driver may share either node.
    assert max_replicas_per_node(tensor_parallel_size=8, pipeline_parallel_size=2, gpus_per_node=8) == 1


# ---------------------------------------------------------------------------
# build_instance_command
# ---------------------------------------------------------------------------


def test_build_instance_command_basic():
    cmd = build_instance_command(
        model="org/model", tensor_parallel_size=1, pipeline_parallel_size=1, trust_remote_code=False, port=8001
    )
    assert cmd[:3] == ["vllm", "serve", "org/model"]
    assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "8001"
    assert "--tensor-parallel-size" in cmd
    assert "--distributed-executor-backend" in cmd
    assert cmd[cmd.index("--distributed-executor-backend") + 1] == "ray"


def test_build_instance_command_uses_given_port():
    cmd = build_instance_command(
        model="org/model", tensor_parallel_size=1, pipeline_parallel_size=1, trust_remote_code=False, port=9001
    )
    assert cmd[cmd.index("--port") + 1] == "9001"


def test_build_instance_command_pipeline_parallel_flag_only_when_gt_1():
    cmd = build_instance_command(
        model="org/model", tensor_parallel_size=1, pipeline_parallel_size=1, trust_remote_code=False, port=8001
    )
    assert "--pipeline-parallel-size" not in cmd

    cmd2 = build_instance_command(
        model="org/model", tensor_parallel_size=1, pipeline_parallel_size=2, trust_remote_code=False, port=8001
    )
    assert "--pipeline-parallel-size" in cmd2
    assert cmd2[cmd2.index("--pipeline-parallel-size") + 1] == "2"


def test_build_instance_command_trust_remote_code():
    cmd = build_instance_command(
        model="org/model", tensor_parallel_size=1, pipeline_parallel_size=1, trust_remote_code=True, port=8001
    )
    assert "--trust-remote-code" in cmd


def test_build_instance_command_no_trust_remote_code_by_default():
    cmd = build_instance_command(
        model="org/model", tensor_parallel_size=1, pipeline_parallel_size=1, trust_remote_code=False, port=8001
    )
    assert "--trust-remote-code" not in cmd
