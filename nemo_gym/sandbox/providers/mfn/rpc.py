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

"""Small async gRPC stub for the vendored MFN protocol."""

from typing import Any

from nemo_gym.sandbox.providers.mfn.protos import mfn_sandbox_pb2 as pb


class MFNSandboxStub:
    """Client methods used by the Gym provider."""

    def __init__(self, channel: Any) -> None:
        # The RPC path prefix is the deployed service's wire-level identity and must
        # match it. Keep it here at the transport boundary; all local and
        # user-facing names use MFN.
        prefix = "/poolside.sandbox.SandboxService/"
        self.Create = channel.unary_unary(
            prefix + "Create",
            request_serializer=pb.CreateOptions.SerializeToString,
            response_deserializer=pb.CreateResult.FromString,
        )
        self.Get = channel.unary_unary(
            prefix + "Get",
            request_serializer=pb.GetRequest.SerializeToString,
            response_deserializer=pb.GetResponse.FromString,
        )
        self.Shutdown = channel.unary_unary(
            prefix + "Shutdown",
            request_serializer=pb.ShutdownOptions.SerializeToString,
            response_deserializer=pb.ShutdownResult.FromString,
        )
        self.AddFile = channel.stream_unary(
            prefix + "AddFile",
            request_serializer=pb.AddFileRequest.SerializeToString,
            response_deserializer=pb.AddFileResult.FromString,
        )
        self.Exec = channel.unary_stream(
            prefix + "Exec",
            request_serializer=pb.ExecRequest.SerializeToString,
            response_deserializer=pb.ExecResponse.FromString,
        )
        self.ExecStream = channel.stream_stream(
            prefix + "ExecStream",
            request_serializer=pb.ExecStreamRequest.SerializeToString,
            response_deserializer=pb.ExecStreamResponse.FromString,
        )
        self.GetHost = channel.unary_unary(
            prefix + "GetHost",
            request_serializer=pb.GetHostRequest.SerializeToString,
            response_deserializer=pb.GetHostResponse.FromString,
        )
        self.ReadFile = channel.unary_stream(
            prefix + "ReadFile",
            request_serializer=pb.ReadFileRequest.SerializeToString,
            response_deserializer=pb.ReadFileResponse.FromString,
        )
