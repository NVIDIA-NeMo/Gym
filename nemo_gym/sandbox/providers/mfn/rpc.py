"""Small async gRPC stub for the vendored MFN protocol."""

from typing import Any

from nemo_gym.sandbox.providers.mfn.protos import mfn_sandbox_pb2 as pb


class MFNSandboxStub:
    """Client methods used by the Gym provider."""

    def __init__(self, channel: Any) -> None:
        # MFN's currently deployed service identity is a legacy wire-level name.
        # Keep it here at the transport boundary; all local and user-facing names
        # use MFN.
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
