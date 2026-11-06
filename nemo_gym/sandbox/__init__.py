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

'Public sandbox API for NeMo Gym.'

from typing import TYPE_CHECKING
from nemo_gym._public_api import lazy_exports

if TYPE_CHECKING:
    from nemo_gym.sandbox.providers.apptainer.provider import ApptainerProvider
    from nemo_gym.sandbox.api import AsyncSandbox
    from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose
    from nemo_gym.sandbox.api import AsyncSandboxSidecar
    from nemo_gym.sandbox.utils import CPU_CAP_ENV_VARS
    from nemo_gym.sandbox.providers.base import ConnectableProvider
    from nemo_gym.sandbox.access import DirectSandboxConnection
    from nemo_gym.sandbox.providers.docker.provider import DockerCreateConfig
    from nemo_gym.sandbox.providers.docker.provider import DockerProvider
    from nemo_gym.sandbox.providers.base import ExecResult
    from nemo_gym.sandbox.providers.local.provider import LocalProvider
    from nemo_gym.sandbox.utils import MEDIA_DIRECTORY
    from nemo_gym.sandbox.providers.base import ORPHAN_REAP_TIMEOUT_S
    from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider
    from nemo_gym.sandbox.adapters.docker_compose import PLACEMENT_GROUP_KEY
    from nemo_gym.sandbox.api import Sandbox
    from nemo_gym.sandbox.access import SandboxAccess
    from nemo_gym.sandbox.config import SandboxConfig
    from nemo_gym.sandbox.providers.base import SandboxCreateError
    from nemo_gym.sandbox.providers.base import SandboxCreateVerificationError
    from nemo_gym.sandbox.providers.base import SandboxEndpoint
    from nemo_gym.sandbox.providers.base import SandboxExecResult
    from nemo_gym.sandbox.providers.base import SandboxHandle
    from nemo_gym.sandbox.transfer import SandboxProbeError
    from nemo_gym.sandbox.providers.base import SandboxProvider
    from nemo_gym.sandbox.api import SandboxPty
    from nemo_gym.sandbox.providers.base import SandboxPtyError
    from nemo_gym.sandbox.providers.base import SandboxPtySession
    from nemo_gym.sandbox.providers.base import SandboxPtySpec
    from nemo_gym.sandbox.providers.base import SandboxResources
    from nemo_gym.sandbox.providers.base import SandboxSidecarSpec
    from nemo_gym.sandbox.providers.base import SandboxSpec
    from nemo_gym.sandbox.providers.base import SandboxStatus
    from nemo_gym.sandbox.transfer import SandboxTransferError
    from nemo_gym.sandbox.transfer import SandboxUnavailableError
    from nemo_gym.sandbox.providers.base import SupportsSandboxEndpoint
    from nemo_gym.sandbox.providers.base import SupportsSandboxNetwork
    from nemo_gym.sandbox.providers.base import SupportsSandboxPauseResume
    from nemo_gym.sandbox.providers.base import SupportsSandboxPortForwarding
    from nemo_gym.sandbox.providers.base import SupportsSandboxPty
    from nemo_gym.sandbox.providers.base import SupportsSandboxPtyAttach
    from nemo_gym.sandbox.providers.base import SupportsSandboxRuntimeRequirements
    from nemo_gym.sandbox.providers.base import SupportsSandboxSharedStorage
    from nemo_gym.sandbox.providers.base import SupportsSandboxSidecars
    from nemo_gym.sandbox.transfer import TRANSFER_USER
    from nemo_gym.sandbox.compose_config import adapt_non_root_sidecars
    from nemo_gym.sandbox.providers.base import clamp_resource_requests
    from nemo_gym.sandbox.utils import cpu_cap_env
    from nemo_gym.sandbox.providers.registry import create_provider
    from nemo_gym.sandbox.compose_config import deployment_platform
    from nemo_gym.sandbox.compose_config import document_volume_sources
    from nemo_gym.sandbox.utils import download_bytes
    from nemo_gym.sandbox.transfer import download_dir
    from nemo_gym.sandbox.transfer import download_path
    from nemo_gym.sandbox.transfer import exec_as_root
    from nemo_gym.sandbox.providers.registry import get_provider_class
    from nemo_gym.sandbox.providers.registry import list_providers
    from nemo_gym.sandbox.utils import media_path
    from nemo_gym.sandbox.compose_config import opensandbox_shm_labels
    from nemo_gym.sandbox.utils import read_text
    from nemo_gym.sandbox.providers.registry import register_provider
    from nemo_gym.sandbox.transfer import remote_kind
    from nemo_gym.sandbox.compose_config import resolve_compose
    from nemo_gym.sandbox.config import resolve_provider_config
    from nemo_gym.sandbox.config import resolve_provider_metadata
    from nemo_gym.sandbox.config import resolve_provider_volume_sources
    from nemo_gym.sandbox.utils import rewrite_image
    from nemo_gym.sandbox.providers.base import sandbox_credentials_rejected
    from nemo_gym.sandbox.failures import sandbox_failure
    from nemo_gym.sandbox.compose_config import sandbox_platform
    from nemo_gym.sandbox.compose_fields import service_gpus
    from nemo_gym.sandbox.compose_fields import service_resources
    from nemo_gym.sandbox.utils import stage_media
    from nemo_gym.sandbox.transfer import upload_archive
    from nemo_gym.sandbox.utils import upload_bytes
    from nemo_gym.sandbox.transfer import upload_dir
    from nemo_gym.sandbox.transfer import upload_path
    from nemo_gym.sandbox.utils import upload_text

_PUBLIC_API = {'ApptainerProvider': 'providers.apptainer.provider', 'AsyncSandbox': 'api', 'AsyncSandboxCompose': 'adapters.docker_compose', 'AsyncSandboxSidecar': 'api', 'CPU_CAP_ENV_VARS': 'utils', 'ConnectableProvider': 'providers.base', 'DirectSandboxConnection': 'access', 'DockerCreateConfig': 'providers.docker.provider', 'DockerProvider': 'providers.docker.provider', 'ExecResult': 'providers.base', 'LocalProvider': 'providers.local.provider', 'MEDIA_DIRECTORY': 'utils', 'ORPHAN_REAP_TIMEOUT_S': 'providers.base', 'OpenSandboxProvider': 'providers.opensandbox.provider', 'PLACEMENT_GROUP_KEY': 'adapters.docker_compose', 'Sandbox': 'api', 'SandboxAccess': 'access', 'SandboxConfig': 'config', 'SandboxCreateError': 'providers.base', 'SandboxCreateVerificationError': 'providers.base', 'SandboxEndpoint': 'providers.base', 'SandboxExecResult': 'providers.base', 'SandboxHandle': 'providers.base', 'SandboxProbeError': 'transfer', 'SandboxProvider': 'providers.base', 'SandboxPty': 'api', 'SandboxPtyError': 'providers.base', 'SandboxPtySession': 'providers.base', 'SandboxPtySpec': 'providers.base', 'SandboxResources': 'providers.base', 'SandboxSidecarSpec': 'providers.base', 'SandboxSpec': 'providers.base', 'SandboxStatus': 'providers.base', 'SandboxTransferError': 'transfer', 'SandboxUnavailableError': 'transfer', 'SupportsSandboxEndpoint': 'providers.base', 'SupportsSandboxNetwork': 'providers.base', 'SupportsSandboxPauseResume': 'providers.base', 'SupportsSandboxPortForwarding': 'providers.base', 'SupportsSandboxPty': 'providers.base', 'SupportsSandboxPtyAttach': 'providers.base', 'SupportsSandboxRuntimeRequirements': 'providers.base', 'SupportsSandboxSharedStorage': 'providers.base', 'SupportsSandboxSidecars': 'providers.base', 'TRANSFER_USER': 'transfer', 'adapt_non_root_sidecars': 'compose_config', 'clamp_resource_requests': 'providers.base', 'cpu_cap_env': 'utils', 'create_provider': 'providers.registry', 'deployment_platform': 'compose_config', 'document_volume_sources': 'compose_config', 'download_bytes': 'utils', 'download_dir': 'transfer', 'download_path': 'transfer', 'exec_as_root': 'transfer', 'get_provider_class': 'providers.registry', 'list_providers': 'providers.registry', 'media_path': 'utils', 'opensandbox_shm_labels': 'compose_config', 'read_text': 'utils', 'register_provider': 'providers.registry', 'remote_kind': 'transfer', 'resolve_compose': 'compose_config', 'resolve_provider_config': 'config', 'resolve_provider_metadata': 'config', 'resolve_provider_volume_sources': 'config', 'rewrite_image': 'utils', 'sandbox_credentials_rejected': 'providers.base', 'sandbox_failure': 'failures', 'sandbox_platform': 'compose_config', 'service_gpus': 'compose_fields', 'service_resources': 'compose_fields', 'stage_media': 'utils', 'upload_archive': 'transfer', 'upload_bytes': 'utils', 'upload_dir': 'transfer', 'upload_path': 'transfer', 'upload_text': 'utils'}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
