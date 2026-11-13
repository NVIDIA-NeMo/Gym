# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"Public sandbox API for NeMo Gym."

from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
    from nemo_gym.sandbox.adapters.docker_compose import PLACEMENT_GROUP_KEY, AsyncSandboxCompose
    from nemo_gym.sandbox.api import AsyncSandbox, AsyncSandboxSidecar, Sandbox, SandboxPty
    from nemo_gym.sandbox.compose_config import (
        adapt_non_root_sidecars,
        deployment_platform,
        document_volume_sources,
        opensandbox_shm_labels,
        resolve_compose,
        sandbox_platform,
    )
    from nemo_gym.sandbox.compose_fields import service_gpus, service_resources
    from nemo_gym.sandbox.config import (
        SandboxConfig,
        resolve_provider_config,
        resolve_provider_metadata,
        resolve_provider_volume_sources,
    )
    from nemo_gym.sandbox.failures import sandbox_failure
    from nemo_gym.sandbox.providers.apptainer.provider import ApptainerProvider
    from nemo_gym.sandbox.providers.base import (
        ORPHAN_REAP_TIMEOUT_S,
        ConnectableProvider,
        ExecResult,
        SandboxCreateError,
        SandboxCreateVerificationError,
        SandboxEndpoint,
        SandboxExecResult,
        SandboxHandle,
        SandboxProvider,
        SandboxPtyError,
        SandboxPtySession,
        SandboxPtySpec,
        SandboxResources,
        SandboxSidecarSpec,
        SandboxSpec,
        SandboxStatus,
        SupportsSandboxEndpoint,
        SupportsSandboxNetwork,
        SupportsSandboxPauseResume,
        SupportsSandboxPortForwarding,
        SupportsSandboxPty,
        SupportsSandboxPtyAttach,
        SupportsSandboxRuntimeRequirements,
        SupportsSandboxSharedStorage,
        SupportsSandboxSidecars,
        clamp_resource_requests,
        sandbox_credentials_rejected,
    )
    from nemo_gym.sandbox.providers.docker.provider import DockerCreateConfig, DockerProvider
    from nemo_gym.sandbox.providers.local.provider import LocalProvider
    from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider
    from nemo_gym.sandbox.providers.registry import (
        create_provider,
        get_provider_class,
        list_providers,
        register_provider,
    )
    from nemo_gym.sandbox.transfer import (
        TRANSFER_USER,
        SandboxProbeError,
        SandboxTransferError,
        SandboxUnavailableError,
        download_dir,
        download_path,
        exec_as_root,
        remote_kind,
        upload_archive,
        upload_dir,
        upload_path,
    )
    from nemo_gym.sandbox.utils import (
        CPU_CAP_ENV_VARS,
        MEDIA_DIRECTORY,
        cpu_cap_env,
        download_bytes,
        media_path,
        read_text,
        rewrite_image,
        stage_media,
        upload_bytes,
        upload_text,
    )

_PUBLIC_API = {
    "ApptainerProvider": "providers.apptainer.provider",
    "AsyncSandbox": "api",
    "AsyncSandboxCompose": "adapters.docker_compose",
    "AsyncSandboxSidecar": "api",
    "CPU_CAP_ENV_VARS": "utils",
    "ConnectableProvider": "providers.base",
    "DirectSandboxConnection": "access",
    "DockerCreateConfig": "providers.docker.provider",
    "DockerProvider": "providers.docker.provider",
    "ExecResult": "providers.base",
    "LocalProvider": "providers.local.provider",
    "MEDIA_DIRECTORY": "utils",
    "ORPHAN_REAP_TIMEOUT_S": "providers.base",
    "OpenSandboxProvider": "providers.opensandbox.provider",
    "PLACEMENT_GROUP_KEY": "adapters.docker_compose",
    "Sandbox": "api",
    "SandboxAccess": "access",
    "SandboxConfig": "config",
    "SandboxCreateError": "providers.base",
    "SandboxCreateVerificationError": "providers.base",
    "SandboxEndpoint": "providers.base",
    "SandboxExecResult": "providers.base",
    "SandboxHandle": "providers.base",
    "SandboxProbeError": "transfer",
    "SandboxProvider": "providers.base",
    "SandboxPty": "api",
    "SandboxPtyError": "providers.base",
    "SandboxPtySession": "providers.base",
    "SandboxPtySpec": "providers.base",
    "SandboxResources": "providers.base",
    "SandboxSidecarSpec": "providers.base",
    "SandboxSpec": "providers.base",
    "SandboxStatus": "providers.base",
    "SandboxTransferError": "transfer",
    "SandboxUnavailableError": "transfer",
    "SupportsSandboxEndpoint": "providers.base",
    "SupportsSandboxNetwork": "providers.base",
    "SupportsSandboxPauseResume": "providers.base",
    "SupportsSandboxPortForwarding": "providers.base",
    "SupportsSandboxPty": "providers.base",
    "SupportsSandboxPtyAttach": "providers.base",
    "SupportsSandboxRuntimeRequirements": "providers.base",
    "SupportsSandboxSharedStorage": "providers.base",
    "SupportsSandboxSidecars": "providers.base",
    "TRANSFER_USER": "transfer",
    "adapt_non_root_sidecars": "compose_config",
    "clamp_resource_requests": "providers.base",
    "cpu_cap_env": "utils",
    "create_provider": "providers.registry",
    "deployment_platform": "compose_config",
    "document_volume_sources": "compose_config",
    "download_bytes": "utils",
    "download_dir": "transfer",
    "download_path": "transfer",
    "exec_as_root": "transfer",
    "get_provider_class": "providers.registry",
    "list_providers": "providers.registry",
    "media_path": "utils",
    "opensandbox_shm_labels": "compose_config",
    "read_text": "utils",
    "register_provider": "providers.registry",
    "remote_kind": "transfer",
    "resolve_compose": "compose_config",
    "resolve_provider_config": "config",
    "resolve_provider_metadata": "config",
    "resolve_provider_volume_sources": "config",
    "rewrite_image": "utils",
    "sandbox_credentials_rejected": "providers.base",
    "sandbox_failure": "failures",
    "sandbox_platform": "compose_config",
    "service_gpus": "compose_fields",
    "service_resources": "compose_fields",
    "stage_media": "utils",
    "upload_archive": "transfer",
    "upload_bytes": "utils",
    "upload_dir": "transfer",
    "upload_path": "transfer",
    "upload_text": "utils",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
