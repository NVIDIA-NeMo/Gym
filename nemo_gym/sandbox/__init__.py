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

"""Public sandbox API for NeMo Gym."""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
    from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose
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
    )
    from nemo_gym.sandbox.providers.docker.provider import DockerCreateConfig, DockerProvider
    from nemo_gym.sandbox.providers.opensandbox.provider import SandboxBackendUnreachableError
    from nemo_gym.sandbox.providers.registry import (
        create_provider,
        get_provider_class,
        list_providers,
        register_provider,
    )
    from nemo_gym.sandbox.transfer import (
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
    from nemo_gym.sandbox.utils import cpu_cap_env, read_text, rewrite_image, upload_text


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "ApptainerProvider": "providers.apptainer.provider",
    "AsyncSandbox": "api",
    "AsyncSandboxCompose": "adapters.docker_compose",
    "AsyncSandboxSidecar": "api",
    "ConnectableProvider": "providers.base",
    "DirectSandboxConnection": "access",
    "DockerCreateConfig": "providers.docker.provider",
    "DockerProvider": "providers.docker.provider",
    "ExecResult": "providers.base",
    "ORPHAN_REAP_TIMEOUT_S": "providers.base",
    "Sandbox": "api",
    "SandboxAccess": "access",
    "SandboxBackendUnreachableError": "providers.opensandbox.provider",
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
    "adapt_non_root_sidecars": "compose_config",
    "clamp_resource_requests": "providers.base",
    "cpu_cap_env": "utils",
    "create_provider": "providers.registry",
    "deployment_platform": "compose_config",
    "document_volume_sources": "compose_config",
    "download_dir": "transfer",
    "download_path": "transfer",
    "exec_as_root": "transfer",
    "get_provider_class": "providers.registry",
    "list_providers": "providers.registry",
    "opensandbox_shm_labels": "compose_config",
    "read_text": "utils",
    "register_provider": "providers.registry",
    "remote_kind": "transfer",
    "resolve_compose": "compose_config",
    "resolve_provider_config": "config",
    "resolve_provider_metadata": "config",
    "resolve_provider_volume_sources": "config",
    "rewrite_image": "utils",
    "sandbox_platform": "compose_config",
    "service_gpus": "compose_fields",
    "service_resources": "compose_fields",
    "upload_archive": "transfer",
    "upload_dir": "transfer",
    "upload_path": "transfer",
    "upload_text": "utils",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
