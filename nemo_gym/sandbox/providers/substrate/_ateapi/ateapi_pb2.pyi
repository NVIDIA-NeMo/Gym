import datetime

from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SnapshotContentScope(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SNAPSHOT_CONTENT_SCOPE_UNSPECIFIED: _ClassVar[SnapshotContentScope]
    SNAPSHOT_CONTENT_SCOPE_FULL: _ClassVar[SnapshotContentScope]
    SNAPSHOT_CONTENT_SCOPE_DATA: _ClassVar[SnapshotContentScope]

class TagScope(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TAG_SCOPE_UNSPECIFIED: _ClassVar[TagScope]
    TAG_SCOPE_ATESPACE: _ClassVar[TagScope]
    TAG_SCOPE_PUBLISHED: _ClassVar[TagScope]

class ActorState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTOR_STATE_UNSPECIFIED: _ClassVar[ActorState]
    ACTOR_STATE_RESUMING: _ClassVar[ActorState]
    ACTOR_STATE_RUNNING: _ClassVar[ActorState]
    ACTOR_STATE_SUSPENDING: _ClassVar[ActorState]
    ACTOR_STATE_SUSPENDED: _ClassVar[ActorState]
    ACTOR_STATE_PAUSING: _ClassVar[ActorState]
    ACTOR_STATE_PAUSED: _ClassVar[ActorState]
    ACTOR_STATE_CRASHED: _ClassVar[ActorState]
    ACTOR_STATE_DELETING: _ClassVar[ActorState]

class SandboxClass(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SANDBOX_CLASS_UNSPECIFIED: _ClassVar[SandboxClass]
    SANDBOX_CLASS_GVISOR: _ClassVar[SandboxClass]
    SANDBOX_CLASS_MICROVM: _ClassVar[SandboxClass]

class ResumeSource(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESUME_SOURCE_UNSPECIFIED: _ClassVar[ResumeSource]
    RESUME_SOURCE_COLD_BOOT: _ClassVar[ResumeSource]
    RESUME_SOURCE_GOLDEN: _ClassVar[ResumeSource]

class ActorMetadataField(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTOR_METADATA_FIELD_UNSPECIFIED: _ClassVar[ActorMetadataField]
    ACTOR_METADATA_FIELD_NAME: _ClassVar[ActorMetadataField]
    ACTOR_METADATA_FIELD_ATESPACE: _ClassVar[ActorMetadataField]
    ACTOR_METADATA_FIELD_UID: _ClassVar[ActorMetadataField]

class WorkerState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WORKER_STATE_UNSPECIFIED: _ClassVar[WorkerState]
    WORKER_STATE_ACTIVE: _ClassVar[WorkerState]
    WORKER_STATE_DRAINING: _ClassVar[WorkerState]

class ActorCertificatePurpose(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTOR_CERTIFICATE_PURPOSE_UNSPECIFIED: _ClassVar[ActorCertificatePurpose]
    ACTOR_CERTIFICATE_PURPOSE_ATUNNEL: _ClassVar[ActorCertificatePurpose]
SNAPSHOT_CONTENT_SCOPE_UNSPECIFIED: SnapshotContentScope
SNAPSHOT_CONTENT_SCOPE_FULL: SnapshotContentScope
SNAPSHOT_CONTENT_SCOPE_DATA: SnapshotContentScope
TAG_SCOPE_UNSPECIFIED: TagScope
TAG_SCOPE_ATESPACE: TagScope
TAG_SCOPE_PUBLISHED: TagScope
ACTOR_STATE_UNSPECIFIED: ActorState
ACTOR_STATE_RESUMING: ActorState
ACTOR_STATE_RUNNING: ActorState
ACTOR_STATE_SUSPENDING: ActorState
ACTOR_STATE_SUSPENDED: ActorState
ACTOR_STATE_PAUSING: ActorState
ACTOR_STATE_PAUSED: ActorState
ACTOR_STATE_CRASHED: ActorState
ACTOR_STATE_DELETING: ActorState
SANDBOX_CLASS_UNSPECIFIED: SandboxClass
SANDBOX_CLASS_GVISOR: SandboxClass
SANDBOX_CLASS_MICROVM: SandboxClass
RESUME_SOURCE_UNSPECIFIED: ResumeSource
RESUME_SOURCE_COLD_BOOT: ResumeSource
RESUME_SOURCE_GOLDEN: ResumeSource
ACTOR_METADATA_FIELD_UNSPECIFIED: ActorMetadataField
ACTOR_METADATA_FIELD_NAME: ActorMetadataField
ACTOR_METADATA_FIELD_ATESPACE: ActorMetadataField
ACTOR_METADATA_FIELD_UID: ActorMetadataField
WORKER_STATE_UNSPECIFIED: WorkerState
WORKER_STATE_ACTIVE: WorkerState
WORKER_STATE_DRAINING: WorkerState
ACTOR_CERTIFICATE_PURPOSE_UNSPECIFIED: ActorCertificatePurpose
ACTOR_CERTIFICATE_PURPOSE_ATUNNEL: ActorCertificatePurpose

class ExternalSnapshot(_message.Message):
    __slots__ = ("snapshot_uri", "content_scope")
    SNAPSHOT_URI_FIELD_NUMBER: _ClassVar[int]
    CONTENT_SCOPE_FIELD_NUMBER: _ClassVar[int]
    snapshot_uri: str
    content_scope: SnapshotContentScope
    def __init__(self, snapshot_uri: _Optional[str] = ..., content_scope: _Optional[_Union[SnapshotContentScope, str]] = ...) -> None: ...

class LocalSnapshotInfo(_message.Message):
    __slots__ = ("snapshot_name", "node_vms_with_local_snapshots", "content_scope")
    SNAPSHOT_NAME_FIELD_NUMBER: _ClassVar[int]
    NODE_VMS_WITH_LOCAL_SNAPSHOTS_FIELD_NUMBER: _ClassVar[int]
    CONTENT_SCOPE_FIELD_NUMBER: _ClassVar[int]
    snapshot_name: str
    node_vms_with_local_snapshots: _containers.RepeatedScalarFieldContainer[str]
    content_scope: SnapshotContentScope
    def __init__(self, snapshot_name: _Optional[str] = ..., node_vms_with_local_snapshots: _Optional[_Iterable[str]] = ..., content_scope: _Optional[_Union[SnapshotContentScope, str]] = ...) -> None: ...

class Selector(_message.Message):
    __slots__ = ("match_labels",)
    class MatchLabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MATCH_LABELS_FIELD_NUMBER: _ClassVar[int]
    match_labels: _containers.ScalarMap[str, str]
    def __init__(self, match_labels: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ResourceMetadata(_message.Message):
    __slots__ = ("atespace", "name", "uid", "version", "create_time", "update_time")
    ATESPACE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    UID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    CREATE_TIME_FIELD_NUMBER: _ClassVar[int]
    UPDATE_TIME_FIELD_NUMBER: _ClassVar[int]
    atespace: str
    name: str
    uid: str
    version: int
    create_time: _timestamp_pb2.Timestamp
    update_time: _timestamp_pb2.Timestamp
    def __init__(self, atespace: _Optional[str] = ..., name: _Optional[str] = ..., uid: _Optional[str] = ..., version: _Optional[int] = ..., create_time: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., update_time: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ExternalVolume(_message.Message):
    __slots__ = ("volume_name", "storage_volume_id", "volume_type", "status", "volume_context")
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATUS_UNSPECIFIED: _ClassVar[ExternalVolume.Status]
        STATUS_PENDING: _ClassVar[ExternalVolume.Status]
        STATUS_CREATED: _ClassVar[ExternalVolume.Status]
        STATUS_DELETING: _ClassVar[ExternalVolume.Status]
    STATUS_UNSPECIFIED: ExternalVolume.Status
    STATUS_PENDING: ExternalVolume.Status
    STATUS_CREATED: ExternalVolume.Status
    STATUS_DELETING: ExternalVolume.Status
    class VolumeContextEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    VOLUME_NAME_FIELD_NUMBER: _ClassVar[int]
    STORAGE_VOLUME_ID_FIELD_NUMBER: _ClassVar[int]
    VOLUME_TYPE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    VOLUME_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    volume_name: str
    storage_volume_id: str
    volume_type: str
    status: ExternalVolume.Status
    volume_context: _containers.ScalarMap[str, str]
    def __init__(self, volume_name: _Optional[str] = ..., storage_volume_id: _Optional[str] = ..., volume_type: _Optional[str] = ..., status: _Optional[_Union[ExternalVolume.Status, str]] = ..., volume_context: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Actor(_message.Message):
    __slots__ = ("metadata", "actor_template", "worker_selector", "source_tag", "status")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ACTOR_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    WORKER_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TAG_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    metadata: ResourceMetadata
    actor_template: ObjectRef
    worker_selector: Selector
    source_tag: ObjectRef
    status: ActorStatus
    def __init__(self, metadata: _Optional[_Union[ResourceMetadata, _Mapping]] = ..., actor_template: _Optional[_Union[ObjectRef, _Mapping]] = ..., worker_selector: _Optional[_Union[Selector, _Mapping]] = ..., source_tag: _Optional[_Union[ObjectRef, _Mapping]] = ..., status: _Optional[_Union[ActorStatus, _Mapping]] = ...) -> None: ...

class EgressPolicy(_message.Message):
    __slots__ = ("metadata", "rules")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    RULES_FIELD_NUMBER: _ClassVar[int]
    metadata: ResourceMetadata
    rules: _containers.RepeatedCompositeFieldContainer[EgressRule]
    def __init__(self, metadata: _Optional[_Union[ResourceMetadata, _Mapping]] = ..., rules: _Optional[_Iterable[_Union[EgressRule, _Mapping]]] = ...) -> None: ...

class EgressRule(_message.Message):
    __slots__ = ("hostnames", "ip_blocks", "all")
    HOSTNAMES_FIELD_NUMBER: _ClassVar[int]
    IP_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    ALL_FIELD_NUMBER: _ClassVar[int]
    hostnames: HostnameRule
    ip_blocks: IPBlockRule
    all: _empty_pb2.Empty
    def __init__(self, hostnames: _Optional[_Union[HostnameRule, _Mapping]] = ..., ip_blocks: _Optional[_Union[IPBlockRule, _Mapping]] = ..., all: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ...) -> None: ...

class HostnameRule(_message.Message):
    __slots__ = ("patterns", "effects")
    PATTERNS_FIELD_NUMBER: _ClassVar[int]
    EFFECTS_FIELD_NUMBER: _ClassVar[int]
    patterns: _containers.RepeatedScalarFieldContainer[str]
    effects: EgressRuleEffects
    def __init__(self, patterns: _Optional[_Iterable[str]] = ..., effects: _Optional[_Union[EgressRuleEffects, _Mapping]] = ...) -> None: ...

class IPBlockRule(_message.Message):
    __slots__ = ("cidrs",)
    CIDRS_FIELD_NUMBER: _ClassVar[int]
    cidrs: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, cidrs: _Optional[_Iterable[str]] = ...) -> None: ...

class EgressRuleEffects(_message.Message):
    __slots__ = ("inject_static_headers",)
    INJECT_STATIC_HEADERS_FIELD_NUMBER: _ClassVar[int]
    inject_static_headers: _containers.RepeatedCompositeFieldContainer[CredentialHeaderInjection]
    def __init__(self, inject_static_headers: _Optional[_Iterable[_Union[CredentialHeaderInjection, _Mapping]]] = ...) -> None: ...

class CredentialHeaderInjection(_message.Message):
    __slots__ = ("header", "prefix", "credential_uri")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    PREFIX_FIELD_NUMBER: _ClassVar[int]
    CREDENTIAL_URI_FIELD_NUMBER: _ClassVar[int]
    header: str
    prefix: str
    credential_uri: str
    def __init__(self, header: _Optional[str] = ..., prefix: _Optional[str] = ..., credential_uri: _Optional[str] = ...) -> None: ...

class ActorStatus(_message.Message):
    __slots__ = ("state", "worker_assignment", "in_progress_snapshot_name", "external_snapshot", "local_snapshot_info", "actor_volumes", "in_progress_local_snapshot_name", "current_actor_template_uid")
    STATE_FIELD_NUMBER: _ClassVar[int]
    WORKER_ASSIGNMENT_FIELD_NUMBER: _ClassVar[int]
    IN_PROGRESS_SNAPSHOT_NAME_FIELD_NUMBER: _ClassVar[int]
    EXTERNAL_SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    LOCAL_SNAPSHOT_INFO_FIELD_NUMBER: _ClassVar[int]
    ACTOR_VOLUMES_FIELD_NUMBER: _ClassVar[int]
    IN_PROGRESS_LOCAL_SNAPSHOT_NAME_FIELD_NUMBER: _ClassVar[int]
    CURRENT_ACTOR_TEMPLATE_UID_FIELD_NUMBER: _ClassVar[int]
    state: ActorState
    worker_assignment: WorkerAssignment
    in_progress_snapshot_name: str
    external_snapshot: ExternalSnapshot
    local_snapshot_info: LocalSnapshotInfo
    actor_volumes: _containers.RepeatedCompositeFieldContainer[ExternalVolume]
    in_progress_local_snapshot_name: str
    current_actor_template_uid: str
    def __init__(self, state: _Optional[_Union[ActorState, str]] = ..., worker_assignment: _Optional[_Union[WorkerAssignment, _Mapping]] = ..., in_progress_snapshot_name: _Optional[str] = ..., external_snapshot: _Optional[_Union[ExternalSnapshot, _Mapping]] = ..., local_snapshot_info: _Optional[_Union[LocalSnapshotInfo, _Mapping]] = ..., actor_volumes: _Optional[_Iterable[_Union[ExternalVolume, _Mapping]]] = ..., in_progress_local_snapshot_name: _Optional[str] = ..., current_actor_template_uid: _Optional[str] = ...) -> None: ...

class WorkerAssignment(_message.Message):
    __slots__ = ("worker", "worker_namespace", "worker_pool", "worker_pod", "worker_pod_uid", "worker_pod_ip")
    WORKER_FIELD_NUMBER: _ClassVar[int]
    WORKER_NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    WORKER_POOL_FIELD_NUMBER: _ClassVar[int]
    WORKER_POD_FIELD_NUMBER: _ClassVar[int]
    WORKER_POD_UID_FIELD_NUMBER: _ClassVar[int]
    WORKER_POD_IP_FIELD_NUMBER: _ClassVar[int]
    worker: ObjectRef
    worker_namespace: str
    worker_pool: str
    worker_pod: str
    worker_pod_uid: str
    worker_pod_ip: str
    def __init__(self, worker: _Optional[_Union[ObjectRef, _Mapping]] = ..., worker_namespace: _Optional[str] = ..., worker_pool: _Optional[str] = ..., worker_pod: _Optional[str] = ..., worker_pod_uid: _Optional[str] = ..., worker_pod_ip: _Optional[str] = ...) -> None: ...

class TagStatus(_message.Message):
    __slots__ = ("snapshot", "actor_template_uid", "in_progress_snapshot_uri", "source_actor_uid")
    SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    ACTOR_TEMPLATE_UID_FIELD_NUMBER: _ClassVar[int]
    IN_PROGRESS_SNAPSHOT_URI_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ACTOR_UID_FIELD_NUMBER: _ClassVar[int]
    snapshot: ExternalSnapshot
    actor_template_uid: str
    in_progress_snapshot_uri: str
    source_actor_uid: str
    def __init__(self, snapshot: _Optional[_Union[ExternalSnapshot, _Mapping]] = ..., actor_template_uid: _Optional[str] = ..., in_progress_snapshot_uri: _Optional[str] = ..., source_actor_uid: _Optional[str] = ...) -> None: ...

class Tag(_message.Message):
    __slots__ = ("metadata", "status", "scope", "source_actor")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ACTOR_FIELD_NUMBER: _ClassVar[int]
    metadata: ResourceMetadata
    status: TagStatus
    scope: TagScope
    source_actor: ObjectRef
    def __init__(self, metadata: _Optional[_Union[ResourceMetadata, _Mapping]] = ..., status: _Optional[_Union[TagStatus, _Mapping]] = ..., scope: _Optional[_Union[TagScope, str]] = ..., source_actor: _Optional[_Union[ObjectRef, _Mapping]] = ...) -> None: ...

class Atespace(_message.Message):
    __slots__ = ("metadata",)
    METADATA_FIELD_NUMBER: _ClassVar[int]
    metadata: ResourceMetadata
    def __init__(self, metadata: _Optional[_Union[ResourceMetadata, _Mapping]] = ...) -> None: ...

class ObjectRef(_message.Message):
    __slots__ = ("atespace", "name")
    ATESPACE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    atespace: str
    name: str
    def __init__(self, atespace: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class ActorTemplate(_message.Message):
    __slots__ = ("metadata", "worker_selector", "containers", "volumes", "snapshots_config", "sandbox_config", "resources", "status")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    WORKER_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    CONTAINERS_FIELD_NUMBER: _ClassVar[int]
    VOLUMES_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOTS_CONFIG_FIELD_NUMBER: _ClassVar[int]
    SANDBOX_CONFIG_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    metadata: ResourceMetadata
    worker_selector: Selector
    containers: _containers.RepeatedCompositeFieldContainer[Container]
    volumes: _containers.RepeatedCompositeFieldContainer[Volume]
    snapshots_config: SnapshotsConfig
    sandbox_config: SandboxConfig
    resources: Resources
    status: ActorTemplateStatus
    def __init__(self, metadata: _Optional[_Union[ResourceMetadata, _Mapping]] = ..., worker_selector: _Optional[_Union[Selector, _Mapping]] = ..., containers: _Optional[_Iterable[_Union[Container, _Mapping]]] = ..., volumes: _Optional[_Iterable[_Union[Volume, _Mapping]]] = ..., snapshots_config: _Optional[_Union[SnapshotsConfig, _Mapping]] = ..., sandbox_config: _Optional[_Union[SandboxConfig, _Mapping]] = ..., resources: _Optional[_Union[Resources, _Mapping]] = ..., status: _Optional[_Union[ActorTemplateStatus, _Mapping]] = ...) -> None: ...

class Resources(_message.Message):
    __slots__ = ("limits",)
    LIMITS_FIELD_NUMBER: _ClassVar[int]
    limits: _containers.RepeatedCompositeFieldContainer[Limits]
    def __init__(self, limits: _Optional[_Iterable[_Union[Limits, _Mapping]]] = ...) -> None: ...

class Limits(_message.Message):
    __slots__ = ("name", "quantity")
    NAME_FIELD_NUMBER: _ClassVar[int]
    QUANTITY_FIELD_NUMBER: _ClassVar[int]
    name: str
    quantity: str
    def __init__(self, name: _Optional[str] = ..., quantity: _Optional[str] = ...) -> None: ...

class GoldenSnapshotStatus(_message.Message):
    __slots__ = ("golden_snapshot", "take_golden_snapshot_at", "error_message")
    GOLDEN_SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    TAKE_GOLDEN_SNAPSHOT_AT_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    golden_snapshot: ExternalSnapshot
    take_golden_snapshot_at: _timestamp_pb2.Timestamp
    error_message: str
    def __init__(self, golden_snapshot: _Optional[_Union[ExternalSnapshot, _Mapping]] = ..., take_golden_snapshot_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., error_message: _Optional[str] = ...) -> None: ...

class ActorTemplateStatus(_message.Message):
    __slots__ = ("golden_snapshot_status",)
    GOLDEN_SNAPSHOT_STATUS_FIELD_NUMBER: _ClassVar[int]
    golden_snapshot_status: GoldenSnapshotStatus
    def __init__(self, golden_snapshot_status: _Optional[_Union[GoldenSnapshotStatus, _Mapping]] = ...) -> None: ...

class SandboxConfig(_message.Message):
    __slots__ = ("sandbox_class", "config_name")
    SANDBOX_CLASS_FIELD_NUMBER: _ClassVar[int]
    CONFIG_NAME_FIELD_NUMBER: _ClassVar[int]
    sandbox_class: SandboxClass
    config_name: str
    def __init__(self, sandbox_class: _Optional[_Union[SandboxClass, str]] = ..., config_name: _Optional[str] = ...) -> None: ...

class SnapshotsConfig(_message.Message):
    __slots__ = ("on_pause", "on_commit", "on_resume", "storage_location")
    ON_PAUSE_FIELD_NUMBER: _ClassVar[int]
    ON_COMMIT_FIELD_NUMBER: _ClassVar[int]
    ON_RESUME_FIELD_NUMBER: _ClassVar[int]
    STORAGE_LOCATION_FIELD_NUMBER: _ClassVar[int]
    on_pause: SnapshotContentScope
    on_commit: SnapshotContentScope
    on_resume: OnResumeConfig
    storage_location: str
    def __init__(self, on_pause: _Optional[_Union[SnapshotContentScope, str]] = ..., on_commit: _Optional[_Union[SnapshotContentScope, str]] = ..., on_resume: _Optional[_Union[OnResumeConfig, _Mapping]] = ..., storage_location: _Optional[str] = ...) -> None: ...

class OnResumeConfig(_message.Message):
    __slots__ = ("from_data",)
    FROM_DATA_FIELD_NUMBER: _ClassVar[int]
    from_data: ResumeSource
    def __init__(self, from_data: _Optional[_Union[ResumeSource, str]] = ...) -> None: ...

class Container(_message.Message):
    __slots__ = ("name", "image", "command", "args", "env", "readyz", "volume_mounts", "security_context", "resources")
    NAME_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    ENV_FIELD_NUMBER: _ClassVar[int]
    READYZ_FIELD_NUMBER: _ClassVar[int]
    VOLUME_MOUNTS_FIELD_NUMBER: _ClassVar[int]
    SECURITY_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    name: str
    image: str
    command: _containers.RepeatedScalarFieldContainer[str]
    args: _containers.RepeatedScalarFieldContainer[str]
    env: _containers.RepeatedCompositeFieldContainer[EnvVar]
    readyz: ContainerReadyz
    volume_mounts: _containers.RepeatedCompositeFieldContainer[VolumeMount]
    security_context: SecurityContext
    resources: Resources
    def __init__(self, name: _Optional[str] = ..., image: _Optional[str] = ..., command: _Optional[_Iterable[str]] = ..., args: _Optional[_Iterable[str]] = ..., env: _Optional[_Iterable[_Union[EnvVar, _Mapping]]] = ..., readyz: _Optional[_Union[ContainerReadyz, _Mapping]] = ..., volume_mounts: _Optional[_Iterable[_Union[VolumeMount, _Mapping]]] = ..., security_context: _Optional[_Union[SecurityContext, _Mapping]] = ..., resources: _Optional[_Union[Resources, _Mapping]] = ...) -> None: ...

class SecurityContext(_message.Message):
    __slots__ = ("capabilities",)
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    capabilities: Capabilities
    def __init__(self, capabilities: _Optional[_Union[Capabilities, _Mapping]] = ...) -> None: ...

class Capabilities(_message.Message):
    __slots__ = ("add", "drop")
    ADD_FIELD_NUMBER: _ClassVar[int]
    DROP_FIELD_NUMBER: _ClassVar[int]
    add: _containers.RepeatedScalarFieldContainer[str]
    drop: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, add: _Optional[_Iterable[str]] = ..., drop: _Optional[_Iterable[str]] = ...) -> None: ...

class EnvVar(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: str
    def __init__(self, name: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class ContainerReadyz(_message.Message):
    __slots__ = ("http_get", "timeout_seconds")
    HTTP_GET_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    http_get: HTTPGetAction
    timeout_seconds: int
    def __init__(self, http_get: _Optional[_Union[HTTPGetAction, _Mapping]] = ..., timeout_seconds: _Optional[int] = ...) -> None: ...

class HTTPGetAction(_message.Message):
    __slots__ = ("path", "port")
    PATH_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    path: str
    port: int
    def __init__(self, path: _Optional[str] = ..., port: _Optional[int] = ...) -> None: ...

class Volume(_message.Message):
    __slots__ = ("name", "durable_dir", "external_volume_template", "system_info", "image")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DURABLE_DIR_FIELD_NUMBER: _ClassVar[int]
    EXTERNAL_VOLUME_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_INFO_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    name: str
    durable_dir: DurableDirVolumeSource
    external_volume_template: ExternalVolumeTemplate
    system_info: SystemInfoVolumeSource
    image: ImageVolumeSource
    def __init__(self, name: _Optional[str] = ..., durable_dir: _Optional[_Union[DurableDirVolumeSource, _Mapping]] = ..., external_volume_template: _Optional[_Union[ExternalVolumeTemplate, _Mapping]] = ..., system_info: _Optional[_Union[SystemInfoVolumeSource, _Mapping]] = ..., image: _Optional[_Union[ImageVolumeSource, _Mapping]] = ...) -> None: ...

class ImageVolumeSource(_message.Message):
    __slots__ = ("reference",)
    REFERENCE_FIELD_NUMBER: _ClassVar[int]
    reference: str
    def __init__(self, reference: _Optional[str] = ...) -> None: ...

class DurableDirVolumeSource(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ExternalVolumeTemplate(_message.Message):
    __slots__ = ("capacity", "storage_class_name")
    CAPACITY_FIELD_NUMBER: _ClassVar[int]
    STORAGE_CLASS_NAME_FIELD_NUMBER: _ClassVar[int]
    capacity: str
    storage_class_name: str
    def __init__(self, capacity: _Optional[str] = ..., storage_class_name: _Optional[str] = ...) -> None: ...

class SystemInfoVolumeSource(_message.Message):
    __slots__ = ("data_sources",)
    DATA_SOURCES_FIELD_NUMBER: _ClassVar[int]
    data_sources: _containers.RepeatedCompositeFieldContainer[SystemInfoDataSource]
    def __init__(self, data_sources: _Optional[_Iterable[_Union[SystemInfoDataSource, _Mapping]]] = ...) -> None: ...

class SystemInfoDataSource(_message.Message):
    __slots__ = ("actor_metadata", "trust_bundle")
    ACTOR_METADATA_FIELD_NUMBER: _ClassVar[int]
    TRUST_BUNDLE_FIELD_NUMBER: _ClassVar[int]
    actor_metadata: ActorMetadataDataSource
    trust_bundle: TrustBundleDataSource
    def __init__(self, actor_metadata: _Optional[_Union[ActorMetadataDataSource, _Mapping]] = ..., trust_bundle: _Optional[_Union[TrustBundleDataSource, _Mapping]] = ...) -> None: ...

class ActorMetadataDataSource(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[ActorMetadataItem]
    def __init__(self, items: _Optional[_Iterable[_Union[ActorMetadataItem, _Mapping]]] = ...) -> None: ...

class ActorMetadataItem(_message.Message):
    __slots__ = ("field", "path")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    field: ActorMetadataField
    path: str
    def __init__(self, field: _Optional[_Union[ActorMetadataField, str]] = ..., path: _Optional[str] = ...) -> None: ...

class TrustBundleDataSource(_message.Message):
    __slots__ = ("name", "path")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    name: str
    path: str
    def __init__(self, name: _Optional[str] = ..., path: _Optional[str] = ...) -> None: ...

class VolumeMount(_message.Message):
    __slots__ = ("name", "mount_path")
    NAME_FIELD_NUMBER: _ClassVar[int]
    MOUNT_PATH_FIELD_NUMBER: _ClassVar[int]
    name: str
    mount_path: str
    def __init__(self, name: _Optional[str] = ..., mount_path: _Optional[str] = ...) -> None: ...

class CreateAtespaceRequest(_message.Message):
    __slots__ = ("atespace",)
    ATESPACE_FIELD_NUMBER: _ClassVar[int]
    atespace: Atespace
    def __init__(self, atespace: _Optional[_Union[Atespace, _Mapping]] = ...) -> None: ...

class GetAtespaceRequest(_message.Message):
    __slots__ = ("atespace",)
    ATESPACE_FIELD_NUMBER: _ClassVar[int]
    atespace: ObjectRef
    def __init__(self, atespace: _Optional[_Union[ObjectRef, _Mapping]] = ...) -> None: ...

class ListAtespacesRequest(_message.Message):
    __slots__ = ("page_size", "page_token")
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    page_size: int
    page_token: str
    def __init__(self, page_size: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class ListAtespacesResponse(_message.Message):
    __slots__ = ("atespaces", "next_page_token")
    ATESPACES_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    atespaces: _containers.RepeatedCompositeFieldContainer[Atespace]
    next_page_token: str
    def __init__(self, atespaces: _Optional[_Iterable[_Union[Atespace, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class DeleteAtespaceRequest(_message.Message):
    __slots__ = ("atespace",)
    ATESPACE_FIELD_NUMBER: _ClassVar[int]
    atespace: ObjectRef
    def __init__(self, atespace: _Optional[_Union[ObjectRef, _Mapping]] = ...) -> None: ...

class CreateActorTemplateRequest(_message.Message):
    __slots__ = ("actor_template",)
    ACTOR_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    actor_template: ActorTemplate
    def __init__(self, actor_template: _Optional[_Union[ActorTemplate, _Mapping]] = ...) -> None: ...

class GetActorTemplateRequest(_message.Message):
    __slots__ = ("actor_template",)
    ACTOR_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    actor_template: ObjectRef
    def __init__(self, actor_template: _Optional[_Union[ObjectRef, _Mapping]] = ...) -> None: ...

class ListActorTemplatesRequest(_message.Message):
    __slots__ = ("atespace", "page_size", "page_token")
    ATESPACE_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    atespace: str
    page_size: int
    page_token: str
    def __init__(self, atespace: _Optional[str] = ..., page_size: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class ListActorTemplatesResponse(_message.Message):
    __slots__ = ("actor_templates", "next_page_token")
    ACTOR_TEMPLATES_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    actor_templates: _containers.RepeatedCompositeFieldContainer[ActorTemplate]
    next_page_token: str
    def __init__(self, actor_templates: _Optional[_Iterable[_Union[ActorTemplate, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class DeleteActorTemplateRequest(_message.Message):
    __slots__ = ("actor_template",)
    ACTOR_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    actor_template: ObjectRef
    def __init__(self, actor_template: _Optional[_Union[ObjectRef, _Mapping]] = ...) -> None: ...

class GetActorRequest(_message.Message):
    __slots__ = ("actor",)
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    actor: ObjectRef
    def __init__(self, actor: _Optional[_Union[ObjectRef, _Mapping]] = ...) -> None: ...

class CreateActorRequest(_message.Message):
    __slots__ = ("actor",)
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    actor: Actor
    def __init__(self, actor: _Optional[_Union[Actor, _Mapping]] = ...) -> None: ...

class UpdateActorRequest(_message.Message):
    __slots__ = ("actor",)
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    actor: Actor
    def __init__(self, actor: _Optional[_Union[Actor, _Mapping]] = ...) -> None: ...

class SuspendActorRequest(_message.Message):
    __slots__ = ("actor",)
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    actor: ObjectRef
    def __init__(self, actor: _Optional[_Union[ObjectRef, _Mapping]] = ...) -> None: ...

class SuspendActorResponse(_message.Message):
    __slots__ = ("actor",)
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    actor: Actor
    def __init__(self, actor: _Optional[_Union[Actor, _Mapping]] = ...) -> None: ...

class PauseActorRequest(_message.Message):
    __slots__ = ("actor",)
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    actor: ObjectRef
    def __init__(self, actor: _Optional[_Union[ObjectRef, _Mapping]] = ...) -> None: ...

class PauseActorResponse(_message.Message):
    __slots__ = ("actor",)
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    actor: Actor
    def __init__(self, actor: _Optional[_Union[Actor, _Mapping]] = ...) -> None: ...

class ResumeActorRequest(_message.Message):
    __slots__ = ("actor", "boot")
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    BOOT_FIELD_NUMBER: _ClassVar[int]
    actor: ObjectRef
    boot: bool
    def __init__(self, actor: _Optional[_Union[ObjectRef, _Mapping]] = ..., boot: bool = ...) -> None: ...

class ResumeActorResponse(_message.Message):
    __slots__ = ("actor", "resumed")
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    RESUMED_FIELD_NUMBER: _ClassVar[int]
    actor: Actor
    resumed: bool
    def __init__(self, actor: _Optional[_Union[Actor, _Mapping]] = ..., resumed: bool = ...) -> None: ...

class DeleteActorRequest(_message.Message):
    __slots__ = ("actor", "any_state")
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    ANY_STATE_FIELD_NUMBER: _ClassVar[int]
    actor: ObjectRef
    any_state: bool
    def __init__(self, actor: _Optional[_Union[ObjectRef, _Mapping]] = ..., any_state: bool = ...) -> None: ...

class GetActorEgressPolicyRequest(_message.Message):
    __slots__ = ("actor",)
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    actor: ObjectRef
    def __init__(self, actor: _Optional[_Union[ObjectRef, _Mapping]] = ...) -> None: ...

class CreateActorEgressPolicyRequest(_message.Message):
    __slots__ = ("actor", "egress_policy")
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    EGRESS_POLICY_FIELD_NUMBER: _ClassVar[int]
    actor: ObjectRef
    egress_policy: EgressPolicy
    def __init__(self, actor: _Optional[_Union[ObjectRef, _Mapping]] = ..., egress_policy: _Optional[_Union[EgressPolicy, _Mapping]] = ...) -> None: ...

class UpdateActorEgressPolicyRequest(_message.Message):
    __slots__ = ("actor", "egress_policy")
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    EGRESS_POLICY_FIELD_NUMBER: _ClassVar[int]
    actor: ObjectRef
    egress_policy: EgressPolicy
    def __init__(self, actor: _Optional[_Union[ObjectRef, _Mapping]] = ..., egress_policy: _Optional[_Union[EgressPolicy, _Mapping]] = ...) -> None: ...

class DeleteActorEgressPolicyRequest(_message.Message):
    __slots__ = ("actor",)
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    actor: ObjectRef
    def __init__(self, actor: _Optional[_Union[ObjectRef, _Mapping]] = ...) -> None: ...

class GetTagRequest(_message.Message):
    __slots__ = ("tag",)
    TAG_FIELD_NUMBER: _ClassVar[int]
    tag: ObjectRef
    def __init__(self, tag: _Optional[_Union[ObjectRef, _Mapping]] = ...) -> None: ...

class ListTagsRequest(_message.Message):
    __slots__ = ("atespace", "page_size", "page_token")
    ATESPACE_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    atespace: str
    page_size: int
    page_token: str
    def __init__(self, atespace: _Optional[str] = ..., page_size: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class ListTagsResponse(_message.Message):
    __slots__ = ("tags", "next_page_token")
    TAGS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    tags: _containers.RepeatedCompositeFieldContainer[Tag]
    next_page_token: str
    def __init__(self, tags: _Optional[_Iterable[_Union[Tag, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class CreateTagRequest(_message.Message):
    __slots__ = ("tag",)
    TAG_FIELD_NUMBER: _ClassVar[int]
    tag: Tag
    def __init__(self, tag: _Optional[_Union[Tag, _Mapping]] = ...) -> None: ...

class UpdateTagRequest(_message.Message):
    __slots__ = ("tag",)
    TAG_FIELD_NUMBER: _ClassVar[int]
    tag: Tag
    def __init__(self, tag: _Optional[_Union[Tag, _Mapping]] = ...) -> None: ...

class DeleteTagRequest(_message.Message):
    __slots__ = ("tag",)
    TAG_FIELD_NUMBER: _ClassVar[int]
    tag: ObjectRef
    def __init__(self, tag: _Optional[_Union[ObjectRef, _Mapping]] = ...) -> None: ...

class DeleteOptions(_message.Message):
    __slots__ = ("version", "uid")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    UID_FIELD_NUMBER: _ClassVar[int]
    version: int
    uid: str
    def __init__(self, version: _Optional[int] = ..., uid: _Optional[str] = ...) -> None: ...

class ListWorkerActorAssignmentsRequest(_message.Message):
    __slots__ = ("worker", "page_size", "page_token")
    WORKER_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    worker: ObjectRef
    page_size: int
    page_token: str
    def __init__(self, worker: _Optional[_Union[ObjectRef, _Mapping]] = ..., page_size: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class ListWorkerActorAssignmentsResponse(_message.Message):
    __slots__ = ("actor_assignments", "next_page_token")
    ACTOR_ASSIGNMENTS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    actor_assignments: _containers.RepeatedCompositeFieldContainer[ActorAssignment]
    next_page_token: str
    def __init__(self, actor_assignments: _Optional[_Iterable[_Union[ActorAssignment, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class ListWorkersRequest(_message.Message):
    __slots__ = ("page_size", "page_token")
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    page_size: int
    page_token: str
    def __init__(self, page_size: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class ListWorkersResponse(_message.Message):
    __slots__ = ("workers", "next_page_token")
    WORKERS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    workers: _containers.RepeatedCompositeFieldContainer[Worker]
    next_page_token: str
    def __init__(self, workers: _Optional[_Iterable[_Union[Worker, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class GetWorkerRequest(_message.Message):
    __slots__ = ("worker",)
    WORKER_FIELD_NUMBER: _ClassVar[int]
    worker: ObjectRef
    def __init__(self, worker: _Optional[_Union[ObjectRef, _Mapping]] = ...) -> None: ...

class CreateWorkerRequest(_message.Message):
    __slots__ = ("worker",)
    WORKER_FIELD_NUMBER: _ClassVar[int]
    worker: Worker
    def __init__(self, worker: _Optional[_Union[Worker, _Mapping]] = ...) -> None: ...

class UpdateWorkerRequest(_message.Message):
    __slots__ = ("worker",)
    WORKER_FIELD_NUMBER: _ClassVar[int]
    worker: Worker
    def __init__(self, worker: _Optional[_Union[Worker, _Mapping]] = ...) -> None: ...

class DeleteWorkerRequest(_message.Message):
    __slots__ = ("worker", "options")
    WORKER_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    worker: ObjectRef
    options: DeleteOptions
    def __init__(self, worker: _Optional[_Union[ObjectRef, _Mapping]] = ..., options: _Optional[_Union[DeleteOptions, _Mapping]] = ...) -> None: ...

class DrainWorkerRequest(_message.Message):
    __slots__ = ("worker",)
    WORKER_FIELD_NUMBER: _ClassVar[int]
    worker: ObjectRef
    def __init__(self, worker: _Optional[_Union[ObjectRef, _Mapping]] = ...) -> None: ...

class ListActorsRequest(_message.Message):
    __slots__ = ("atespace", "page_size", "page_token")
    ATESPACE_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    atespace: str
    page_size: int
    page_token: str
    def __init__(self, atespace: _Optional[str] = ..., page_size: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class ListActorsResponse(_message.Message):
    __slots__ = ("actors", "next_page_token")
    ACTORS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    actors: _containers.RepeatedCompositeFieldContainer[Actor]
    next_page_token: str
    def __init__(self, actors: _Optional[_Iterable[_Union[Actor, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class Worker(_message.Message):
    __slots__ = ("metadata", "worker_namespace", "worker_pool", "worker_pod", "worker_pod_uid", "node_name", "ip", "sandbox_class", "labels", "status")
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    METADATA_FIELD_NUMBER: _ClassVar[int]
    WORKER_NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    WORKER_POOL_FIELD_NUMBER: _ClassVar[int]
    WORKER_POD_FIELD_NUMBER: _ClassVar[int]
    WORKER_POD_UID_FIELD_NUMBER: _ClassVar[int]
    NODE_NAME_FIELD_NUMBER: _ClassVar[int]
    IP_FIELD_NUMBER: _ClassVar[int]
    SANDBOX_CLASS_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    metadata: ResourceMetadata
    worker_namespace: str
    worker_pool: str
    worker_pod: str
    worker_pod_uid: str
    node_name: str
    ip: str
    sandbox_class: str
    labels: _containers.ScalarMap[str, str]
    status: WorkerStatus
    def __init__(self, metadata: _Optional[_Union[ResourceMetadata, _Mapping]] = ..., worker_namespace: _Optional[str] = ..., worker_pool: _Optional[str] = ..., worker_pod: _Optional[str] = ..., worker_pod_uid: _Optional[str] = ..., node_name: _Optional[str] = ..., ip: _Optional[str] = ..., sandbox_class: _Optional[str] = ..., labels: _Optional[_Mapping[str, str]] = ..., status: _Optional[_Union[WorkerStatus, _Mapping]] = ...) -> None: ...

class WorkerStatus(_message.Message):
    __slots__ = ("state", "capacity", "allocated")
    STATE_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_FIELD_NUMBER: _ClassVar[int]
    ALLOCATED_FIELD_NUMBER: _ClassVar[int]
    state: WorkerState
    capacity: WorkerResources
    allocated: WorkerResources
    def __init__(self, state: _Optional[_Union[WorkerState, str]] = ..., capacity: _Optional[_Union[WorkerResources, _Mapping]] = ..., allocated: _Optional[_Union[WorkerResources, _Mapping]] = ...) -> None: ...

class WorkerResources(_message.Message):
    __slots__ = ("resources", "actors")
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    ACTORS_FIELD_NUMBER: _ClassVar[int]
    resources: Resources
    actors: int
    def __init__(self, resources: _Optional[_Union[Resources, _Mapping]] = ..., actors: _Optional[int] = ...) -> None: ...

class ActorAssignment(_message.Message):
    __slots__ = ("metadata", "actor", "actor_uid", "actor_template_ref", "resources")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    ACTOR_UID_FIELD_NUMBER: _ClassVar[int]
    ACTOR_TEMPLATE_REF_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    metadata: ResourceMetadata
    actor: ObjectRef
    actor_uid: str
    actor_template_ref: ObjectRef
    resources: Resources
    def __init__(self, metadata: _Optional[_Union[ResourceMetadata, _Mapping]] = ..., actor: _Optional[_Union[ObjectRef, _Mapping]] = ..., actor_uid: _Optional[str] = ..., actor_template_ref: _Optional[_Union[ObjectRef, _Mapping]] = ..., resources: _Optional[_Union[Resources, _Mapping]] = ...) -> None: ...

class SetWorkerCapacityRequest(_message.Message):
    __slots__ = ("worker", "capacity")
    WORKER_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_FIELD_NUMBER: _ClassVar[int]
    worker: ObjectRef
    capacity: WorkerResources
    def __init__(self, worker: _Optional[_Union[ObjectRef, _Mapping]] = ..., capacity: _Optional[_Union[WorkerResources, _Mapping]] = ...) -> None: ...

class SetWorkerCapacityResponse(_message.Message):
    __slots__ = ("worker",)
    WORKER_FIELD_NUMBER: _ClassVar[int]
    worker: Worker
    def __init__(self, worker: _Optional[_Union[Worker, _Mapping]] = ...) -> None: ...

class MintJWTRequest(_message.Message):
    __slots__ = ("audience", "atespace", "actor_name", "actor_uid")
    AUDIENCE_FIELD_NUMBER: _ClassVar[int]
    ATESPACE_FIELD_NUMBER: _ClassVar[int]
    ACTOR_NAME_FIELD_NUMBER: _ClassVar[int]
    ACTOR_UID_FIELD_NUMBER: _ClassVar[int]
    audience: _containers.RepeatedScalarFieldContainer[str]
    atespace: str
    actor_name: str
    actor_uid: str
    def __init__(self, audience: _Optional[_Iterable[str]] = ..., atespace: _Optional[str] = ..., actor_name: _Optional[str] = ..., actor_uid: _Optional[str] = ...) -> None: ...

class MintJWTResponse(_message.Message):
    __slots__ = ("actor_jwt",)
    ACTOR_JWT_FIELD_NUMBER: _ClassVar[int]
    actor_jwt: str
    def __init__(self, actor_jwt: _Optional[str] = ...) -> None: ...

class MintCertRequest(_message.Message):
    __slots__ = ("worker", "certificate_signing_request", "expected_actor_uid", "purpose")
    WORKER_FIELD_NUMBER: _ClassVar[int]
    CERTIFICATE_SIGNING_REQUEST_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_ACTOR_UID_FIELD_NUMBER: _ClassVar[int]
    PURPOSE_FIELD_NUMBER: _ClassVar[int]
    worker: ObjectRef
    certificate_signing_request: bytes
    expected_actor_uid: str
    purpose: ActorCertificatePurpose
    def __init__(self, worker: _Optional[_Union[ObjectRef, _Mapping]] = ..., certificate_signing_request: _Optional[bytes] = ..., expected_actor_uid: _Optional[str] = ..., purpose: _Optional[_Union[ActorCertificatePurpose, str]] = ...) -> None: ...

class MintCertResponse(_message.Message):
    __slots__ = ("actor_certificates",)
    ACTOR_CERTIFICATES_FIELD_NUMBER: _ClassVar[int]
    actor_certificates: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, actor_certificates: _Optional[_Iterable[bytes]] = ...) -> None: ...
