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
from unittest.mock import call, patch

import pytest

from nemo_gym.orchestration.ray_serve_gateway import (
    MAX_ONGOING_REQUESTS_PER_INSTANCE,
    PROXY_TIMEOUT,
    build_instance_command,
    deployment_options,
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


def test_parse_args_served_model_name_and_extra_args():
    args = parse_args(
        ["--model", "org/model", "--port", "8000", "--served-model-name", "my-model", "--extra-args", "--foo bar"]
    )
    assert args.served_model_name == "my-model"
    assert args.extra_args == "--foo bar"


def test_parse_args_served_model_name_and_extra_args_default_to_none_and_empty():
    args = parse_args(["--model", "org/model", "--port", "8000"])
    assert args.served_model_name is None
    assert args.extra_args == ""


def test_parse_args_missing_required_raises():
    with pytest.raises(SystemExit):
        parse_args(["--port", "8000"])


def test_parse_args_accepts_gpus_per_node_for_caller_compatibility():
    args = parse_args(["--model", "org/model", "--port", "8000", "--gpus-per-node", "8"])
    assert args.gpus_per_node == 8


# ---------------------------------------------------------------------------
# free_local_port
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("port", [12345, 54321])
def test_free_local_port_binds_ephemeral_tcp_port_and_closes_socket(port):
    # The socket is closed on return, so neither uniqueness nor later availability is guaranteed.
    with patch("nemo_gym.orchestration.ray_serve_gateway.socket.socket") as socket_factory:
        sock = socket_factory.return_value.__enter__.return_value
        sock.getsockname.return_value = ("0.0.0.0", port)

        assert free_local_port() == port

    assert socket_factory.mock_calls == [
        call(socket.AF_INET, socket.SOCK_STREAM),
        call().__enter__(),
        call().__enter__().bind(("", 0)),
        call().__enter__().getsockname(),
        call().__exit__(None, None, None),
    ]


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
# deployment_options
# ---------------------------------------------------------------------------


def test_deployment_options_lifts_the_per_instance_request_cap():
    # Ray Serve defaults max_ongoing_requests to 5, which would cap each vLLM instance at 5
    # concurrent requests no matter how large its own batch is.
    args = parse_args(["--model", "/m", "--port", "8000", "--number-of-instances", "6", "--tensor-parallel-size", "8"])
    options = deployment_options(args)
    assert options["max_ongoing_requests"] == MAX_ONGOING_REQUESTS_PER_INSTANCE
    assert MAX_ONGOING_REQUESTS_PER_INSTANCE >= 16384


def test_deployment_options_carries_replica_count_and_placement():
    args = parse_args(
        ["--model", "/m", "--port", "8000", "--number-of-instances", "4", "--tensor-parallel-size", "2"]
        + ["--gpus-per-node", "8"]
    )
    options = deployment_options(args)
    assert options["num_replicas"] == 4
    assert options["max_replicas_per_node"] == 4


def test_proxy_timeout_does_not_cut_off_long_requests():
    # aiohttp's default total timeout is 300 s; a long generation must not be turned into a 500.
    assert PROXY_TIMEOUT.total is None
    assert PROXY_TIMEOUT.sock_read is None


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


def test_build_instance_command_served_model_name():
    cmd = build_instance_command(
        model="org/model",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        trust_remote_code=False,
        port=8001,
        served_model_name="my-model",
    )
    assert cmd[cmd.index("--served-model-name") + 1] == "my-model"


def test_build_instance_command_no_served_model_name_by_default():
    cmd = build_instance_command(
        model="org/model", tensor_parallel_size=1, pipeline_parallel_size=1, trust_remote_code=False, port=8001
    )
    assert "--served-model-name" not in cmd


def test_build_instance_command_extra_args_split_into_separate_tokens():
    cmd = build_instance_command(
        model="org/model",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        trust_remote_code=False,
        port=8001,
        extra_args="--max-model-len 8192",
    )
    assert "--max-model-len" in cmd
    assert cmd[cmd.index("--max-model-len") + 1] == "8192"
