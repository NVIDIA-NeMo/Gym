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

import pytest

from nemo_gym.orchestration.ray_serve_gateway import (
    RoundRobinRouter,
    build_instance_command,
    instance_port,
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


# ---------------------------------------------------------------------------
# instance_port
# ---------------------------------------------------------------------------


def test_instance_port_offsets_above_gateway_port():
    assert instance_port(8000, 0) == 8001
    assert instance_port(8000, 3) == 8004


def test_instance_port_never_collides_with_gateway_port():
    for i in range(8):
        assert instance_port(8000, i) != 8000


# ---------------------------------------------------------------------------
# build_instance_command
# ---------------------------------------------------------------------------


def test_build_instance_command_basic():
    args = parse_args(["--model", "org/model", "--port", "8000"])
    cmd = build_instance_command(args, 0)
    assert cmd[:3] == ["vllm", "serve", "org/model"]
    assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "8001"
    assert "--tensor-parallel-size" in cmd
    assert "--distributed-executor-backend" in cmd
    assert cmd[cmd.index("--distributed-executor-backend") + 1] == "ray"


def test_build_instance_command_per_instance_port():
    args = parse_args(["--model", "org/model", "--port", "8000"])
    cmd0 = build_instance_command(args, 0)
    cmd1 = build_instance_command(args, 1)
    assert cmd0[cmd0.index("--port") + 1] == "8001"
    assert cmd1[cmd1.index("--port") + 1] == "8002"


def test_build_instance_command_pipeline_parallel_flag_only_when_gt_1():
    args = parse_args(["--model", "org/model", "--port", "8000"])
    cmd = build_instance_command(args, 0)
    assert "--pipeline-parallel-size" not in cmd

    args2 = parse_args(["--model", "org/model", "--port", "8000", "--pipeline-parallel-size", "2"])
    cmd2 = build_instance_command(args2, 0)
    assert "--pipeline-parallel-size" in cmd2
    assert cmd2[cmd2.index("--pipeline-parallel-size") + 1] == "2"


def test_build_instance_command_trust_remote_code():
    args = parse_args(["--model", "org/model", "--port", "8000", "--trust-remote-code"])
    cmd = build_instance_command(args, 0)
    assert "--trust-remote-code" in cmd


def test_build_instance_command_no_trust_remote_code_by_default():
    args = parse_args(["--model", "org/model", "--port", "8000"])
    cmd = build_instance_command(args, 0)
    assert "--trust-remote-code" not in cmd


# ---------------------------------------------------------------------------
# RoundRobinRouter
# ---------------------------------------------------------------------------


def test_round_robin_router_cycles_in_order():
    router = RoundRobinRouter(["a", "b", "c"])
    assert [router.next_url() for _ in range(6)] == ["a", "b", "c", "a", "b", "c"]


def test_round_robin_router_single_url():
    router = RoundRobinRouter(["only"])
    assert [router.next_url() for _ in range(3)] == ["only", "only", "only"]


def test_round_robin_router_empty_raises():
    with pytest.raises(ValueError):
        RoundRobinRouter([])
