import datetime

from google.protobuf import duration_pb2 as _duration_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ResourceRequest(_message.Message):
    __slots__ = ("cpu_request", "memory_request", "memory_limit", "storage_request", "storage_limit", "gpu")
    CPU_REQUEST_FIELD_NUMBER: _ClassVar[int]
    MEMORY_REQUEST_FIELD_NUMBER: _ClassVar[int]
    MEMORY_LIMIT_FIELD_NUMBER: _ClassVar[int]
    STORAGE_REQUEST_FIELD_NUMBER: _ClassVar[int]
    STORAGE_LIMIT_FIELD_NUMBER: _ClassVar[int]
    GPU_FIELD_NUMBER: _ClassVar[int]
    cpu_request: str
    memory_request: str
    memory_limit: str
    storage_request: str
    storage_limit: str
    gpu: GpuRequest
    def __init__(self, cpu_request: _Optional[str] = ..., memory_request: _Optional[str] = ..., memory_limit: _Optional[str] = ..., storage_request: _Optional[str] = ..., storage_limit: _Optional[str] = ..., gpu: _Optional[_Union[GpuRequest, _Mapping]] = ...) -> None: ...

class GpuRequest(_message.Message):
    __slots__ = ("count", "type_preferences")
    COUNT_FIELD_NUMBER: _ClassVar[int]
    TYPE_PREFERENCES_FIELD_NUMBER: _ClassVar[int]
    count: int
    type_preferences: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, count: _Optional[int] = ..., type_preferences: _Optional[_Iterable[str]] = ...) -> None: ...

class SandboxSpecs(_message.Message):
    __slots__ = ("resource_request", "sandbox_ttl", "env_vars", "image", "snapshot_id", "idle_timeout", "auto_pause_after", "require_vm", "beta_no_ttl_cap_use_only_when_really_needed")
    class EnvVarsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    RESOURCE_REQUEST_FIELD_NUMBER: _ClassVar[int]
    SANDBOX_TTL_FIELD_NUMBER: _ClassVar[int]
    ENV_VARS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
    IDLE_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    AUTO_PAUSE_AFTER_FIELD_NUMBER: _ClassVar[int]
    REQUIRE_VM_FIELD_NUMBER: _ClassVar[int]
    BETA_NO_TTL_CAP_USE_ONLY_WHEN_REALLY_NEEDED_FIELD_NUMBER: _ClassVar[int]
    resource_request: ResourceRequest
    sandbox_ttl: _duration_pb2.Duration
    env_vars: _containers.ScalarMap[str, str]
    image: str
    snapshot_id: str
    idle_timeout: _duration_pb2.Duration
    auto_pause_after: _duration_pb2.Duration
    require_vm: bool
    beta_no_ttl_cap_use_only_when_really_needed: bool
    def __init__(self, resource_request: _Optional[_Union[ResourceRequest, _Mapping]] = ..., sandbox_ttl: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., env_vars: _Optional[_Mapping[str, str]] = ..., image: _Optional[str] = ..., snapshot_id: _Optional[str] = ..., idle_timeout: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., auto_pause_after: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ..., require_vm: bool = ..., beta_no_ttl_cap_use_only_when_really_needed: bool = ...) -> None: ...

class SandboxAttributes(_message.Message):
    __slots__ = ("attributes",)
    class AttributesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    attributes: _containers.ScalarMap[str, str]
    def __init__(self, attributes: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ContainerPort(_message.Message):
    __slots__ = ("name", "port")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    name: str
    port: int
    def __init__(self, name: _Optional[str] = ..., port: _Optional[int] = ...) -> None: ...

class NetworkConfig(_message.Message):
    __slots__ = ("allowed_cidrs", "allowed_domains", "mode", "blocked_cidrs", "blocked_domains", "name")
    class Mode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        ALLOW: _ClassVar[NetworkConfig.Mode]
        BLOCK: _ClassVar[NetworkConfig.Mode]
    ALLOW: NetworkConfig.Mode
    BLOCK: NetworkConfig.Mode
    ALLOWED_CIDRS_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_DOMAINS_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    BLOCKED_CIDRS_FIELD_NUMBER: _ClassVar[int]
    BLOCKED_DOMAINS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    allowed_cidrs: _containers.RepeatedScalarFieldContainer[str]
    allowed_domains: _containers.RepeatedScalarFieldContainer[str]
    mode: NetworkConfig.Mode
    blocked_cidrs: _containers.RepeatedScalarFieldContainer[str]
    blocked_domains: _containers.RepeatedScalarFieldContainer[str]
    name: str
    def __init__(self, allowed_cidrs: _Optional[_Iterable[str]] = ..., allowed_domains: _Optional[_Iterable[str]] = ..., mode: _Optional[_Union[NetworkConfig.Mode, str]] = ..., blocked_cidrs: _Optional[_Iterable[str]] = ..., blocked_domains: _Optional[_Iterable[str]] = ..., name: _Optional[str] = ...) -> None: ...

class ClientInfo(_message.Message):
    __slots__ = ("caller", "username")
    CALLER_FIELD_NUMBER: _ClassVar[int]
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    caller: str
    username: str
    def __init__(self, caller: _Optional[str] = ..., username: _Optional[str] = ...) -> None: ...

class CreateOptions(_message.Message):
    __slots__ = ("specs", "attributes", "ports", "idempotency_key", "network_config", "client_info", "shard_pin")
    SPECS_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    PORTS_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_KEY_FIELD_NUMBER: _ClassVar[int]
    NETWORK_CONFIG_FIELD_NUMBER: _ClassVar[int]
    CLIENT_INFO_FIELD_NUMBER: _ClassVar[int]
    SHARD_PIN_FIELD_NUMBER: _ClassVar[int]
    specs: SandboxSpecs
    attributes: SandboxAttributes
    ports: _containers.RepeatedCompositeFieldContainer[ContainerPort]
    idempotency_key: str
    network_config: NetworkConfig
    client_info: ClientInfo
    shard_pin: str
    def __init__(self, specs: _Optional[_Union[SandboxSpecs, _Mapping]] = ..., attributes: _Optional[_Union[SandboxAttributes, _Mapping]] = ..., ports: _Optional[_Iterable[_Union[ContainerPort, _Mapping]]] = ..., idempotency_key: _Optional[str] = ..., network_config: _Optional[_Union[NetworkConfig, _Mapping]] = ..., client_info: _Optional[_Union[ClientInfo, _Mapping]] = ..., shard_pin: _Optional[str] = ...) -> None: ...

class CreateResult(_message.Message):
    __slots__ = ("sandbox_id", "already_exists", "is_ready")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    ALREADY_EXISTS_FIELD_NUMBER: _ClassVar[int]
    IS_READY_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    already_exists: bool
    is_ready: bool
    def __init__(self, sandbox_id: _Optional[str] = ..., already_exists: bool = ..., is_ready: bool = ...) -> None: ...

class GetRequest(_message.Message):
    __slots__ = ("sandbox_id",)
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    def __init__(self, sandbox_id: _Optional[str] = ...) -> None: ...

class Status(_message.Message):
    __slots__ = ("is_ready", "phase")
    IS_READY_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    is_ready: bool
    phase: str
    def __init__(self, is_ready: bool = ..., phase: _Optional[str] = ...) -> None: ...

class GetResponse(_message.Message):
    __slots__ = ("sandbox_id", "status")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    status: Status
    def __init__(self, sandbox_id: _Optional[str] = ..., status: _Optional[_Union[Status, _Mapping]] = ...) -> None: ...

class ShutdownOptions(_message.Message):
    __slots__ = ("sandbox_id",)
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    def __init__(self, sandbox_id: _Optional[str] = ...) -> None: ...

class ShutdownResult(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class AddFileRequest(_message.Message):
    __slots__ = ("content",)
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    content: bytes
    def __init__(self, content: _Optional[bytes] = ...) -> None: ...

class AddFileResult(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Pty(_message.Message):
    __slots__ = ("rows", "cols")
    ROWS_FIELD_NUMBER: _ClassVar[int]
    COLS_FIELD_NUMBER: _ClassVar[int]
    rows: int
    cols: int
    def __init__(self, rows: _Optional[int] = ..., cols: _Optional[int] = ...) -> None: ...

class ExecRequest(_message.Message):
    __slots__ = ("sandbox_id", "command", "output_buffer_size", "env", "cwd", "unbuffered", "container", "pty")
    class EnvEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_BUFFER_SIZE_FIELD_NUMBER: _ClassVar[int]
    ENV_FIELD_NUMBER: _ClassVar[int]
    CWD_FIELD_NUMBER: _ClassVar[int]
    UNBUFFERED_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_FIELD_NUMBER: _ClassVar[int]
    PTY_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    command: _containers.RepeatedScalarFieldContainer[str]
    output_buffer_size: int
    env: _containers.ScalarMap[str, str]
    cwd: str
    unbuffered: bool
    container: str
    pty: Pty
    def __init__(self, sandbox_id: _Optional[str] = ..., command: _Optional[_Iterable[str]] = ..., output_buffer_size: _Optional[int] = ..., env: _Optional[_Mapping[str, str]] = ..., cwd: _Optional[str] = ..., unbuffered: bool = ..., container: _Optional[str] = ..., pty: _Optional[_Union[Pty, _Mapping]] = ...) -> None: ...

class ExecDetails(_message.Message):
    __slots__ = ("execution_id",)
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    execution_id: str
    def __init__(self, execution_id: _Optional[str] = ...) -> None: ...

class ExecOutput(_message.Message):
    __slots__ = ("stream", "data")
    class StreamType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STDOUT: _ClassVar[ExecOutput.StreamType]
        STDERR: _ClassVar[ExecOutput.StreamType]
    STDOUT: ExecOutput.StreamType
    STDERR: ExecOutput.StreamType
    STREAM_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    stream: ExecOutput.StreamType
    data: bytes
    def __init__(self, stream: _Optional[_Union[ExecOutput.StreamType, str]] = ..., data: _Optional[bytes] = ...) -> None: ...

class ExecComplete(_message.Message):
    __slots__ = ("exit_code", "error", "termination_detail")
    EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    TERMINATION_DETAIL_FIELD_NUMBER: _ClassVar[int]
    exit_code: int
    error: str
    termination_detail: str
    def __init__(self, exit_code: _Optional[int] = ..., error: _Optional[str] = ..., termination_detail: _Optional[str] = ...) -> None: ...

class ExecResponse(_message.Message):
    __slots__ = ("output", "complete", "details")
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    output: ExecOutput
    complete: ExecComplete
    details: ExecDetails
    def __init__(self, output: _Optional[_Union[ExecOutput, _Mapping]] = ..., complete: _Optional[_Union[ExecComplete, _Mapping]] = ..., details: _Optional[_Union[ExecDetails, _Mapping]] = ...) -> None: ...

class SendStdInRequest(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...

class ExecStreamRequest(_message.Message):
    __slots__ = ("request", "stdin", "resize")
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    STDIN_FIELD_NUMBER: _ClassVar[int]
    RESIZE_FIELD_NUMBER: _ClassVar[int]
    request: ExecRequest
    stdin: SendStdInRequest
    resize: Pty
    def __init__(self, request: _Optional[_Union[ExecRequest, _Mapping]] = ..., stdin: _Optional[_Union[SendStdInRequest, _Mapping]] = ..., resize: _Optional[_Union[Pty, _Mapping]] = ...) -> None: ...

class ExecStreamResponse(_message.Message):
    __slots__ = ("output", "complete")
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    output: ExecOutput
    complete: ExecComplete
    def __init__(self, output: _Optional[_Union[ExecOutput, _Mapping]] = ..., complete: _Optional[_Union[ExecComplete, _Mapping]] = ...) -> None: ...

class GetHostRequest(_message.Message):
    __slots__ = ("sandbox_id", "port")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    port: int
    def __init__(self, sandbox_id: _Optional[str] = ..., port: _Optional[int] = ...) -> None: ...

class GetHostResponse(_message.Message):
    __slots__ = ("uri",)
    URI_FIELD_NUMBER: _ClassVar[int]
    uri: str
    def __init__(self, uri: _Optional[str] = ...) -> None: ...

class ReadFileRequest(_message.Message):
    __slots__ = ("sandbox_id", "path", "container")
    SANDBOX_ID_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_FIELD_NUMBER: _ClassVar[int]
    sandbox_id: str
    path: str
    container: str
    def __init__(self, sandbox_id: _Optional[str] = ..., path: _Optional[str] = ..., container: _Optional[str] = ...) -> None: ...

class ReadFileResponse(_message.Message):
    __slots__ = ("content",)
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    content: bytes
    def __init__(self, content: _Optional[bytes] = ...) -> None: ...
