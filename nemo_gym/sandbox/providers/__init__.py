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

'Sandbox provider registry.'

from typing import TYPE_CHECKING
from nemo_gym._public_api import lazy_exports

if TYPE_CHECKING:
    from nemo_gym.sandbox.providers.base import ConnectableProvider
    from nemo_gym.sandbox.providers.base import ExecResult
    from nemo_gym.sandbox.providers.base import SandboxCreateError
    from nemo_gym.sandbox.providers.base import SandboxCreateVerificationError
    from nemo_gym.sandbox.providers.base import SandboxEndpoint
    from nemo_gym.sandbox.providers.base import SandboxExecResult
    from nemo_gym.sandbox.providers.base import SandboxHandle
    from nemo_gym.sandbox.providers.base import SandboxProvider
    from nemo_gym.sandbox.providers.base import SandboxPtyError
    from nemo_gym.sandbox.providers.base import SandboxPtySession
    from nemo_gym.sandbox.providers.base import SandboxPtySpec
    from nemo_gym.sandbox.providers.base import SandboxResources
    from nemo_gym.sandbox.providers.base import SandboxSidecarSpec
    from nemo_gym.sandbox.providers.base import SandboxSpec
    from nemo_gym.sandbox.providers.base import SandboxStatus
    from nemo_gym.sandbox.providers.base import SupportsSandboxBackgroundServices
    from nemo_gym.sandbox.providers.base import SupportsSandboxEndpoint
    from nemo_gym.sandbox.providers.base import SupportsSandboxIndependentUploads
    from nemo_gym.sandbox.providers.base import SupportsSandboxNetwork
    from nemo_gym.sandbox.providers.base import SupportsSandboxPauseResume
    from nemo_gym.sandbox.providers.base import SupportsSandboxPortForwarding
    from nemo_gym.sandbox.providers.base import SupportsSandboxPty
    from nemo_gym.sandbox.providers.base import SupportsSandboxPtyAttach
    from nemo_gym.sandbox.providers.base import SupportsSandboxPullPolicy
    from nemo_gym.sandbox.providers.base import SupportsSandboxRuntimeRequirements
    from nemo_gym.sandbox.providers.base import SupportsSandboxServiceOptions
    from nemo_gym.sandbox.providers.base import SupportsSandboxSharedStorage
    from nemo_gym.sandbox.providers.base import SupportsSandboxSidecars
    from nemo_gym.sandbox.providers.base import clamp_resource_requests
    from nemo_gym.sandbox.providers.registry import create_provider
    from nemo_gym.sandbox.providers.registry import get_provider_class
    from nemo_gym.sandbox.providers.registry import list_providers
    from nemo_gym.sandbox.providers.registry import provider_service_option_fields
    from nemo_gym.sandbox.providers.registry import register_provider

_PUBLIC_API = {'ConnectableProvider': 'base', 'ExecResult': 'base', 'SandboxCreateError': 'base', 'SandboxCreateVerificationError': 'base', 'SandboxEndpoint': 'base', 'SandboxExecResult': 'base', 'SandboxHandle': 'base', 'SandboxProvider': 'base', 'SandboxPtyError': 'base', 'SandboxPtySession': 'base', 'SandboxPtySpec': 'base', 'SandboxResources': 'base', 'SandboxSidecarSpec': 'base', 'SandboxSpec': 'base', 'SandboxStatus': 'base', 'SupportsSandboxBackgroundServices': 'base', 'SupportsSandboxEndpoint': 'base', 'SupportsSandboxIndependentUploads': 'base', 'SupportsSandboxNetwork': 'base', 'SupportsSandboxPauseResume': 'base', 'SupportsSandboxPortForwarding': 'base', 'SupportsSandboxPty': 'base', 'SupportsSandboxPtyAttach': 'base', 'SupportsSandboxPullPolicy': 'base', 'SupportsSandboxRuntimeRequirements': 'base', 'SupportsSandboxServiceOptions': 'base', 'SupportsSandboxSharedStorage': 'base', 'SupportsSandboxSidecars': 'base', 'clamp_resource_requests': 'base', 'create_provider': 'registry', 'get_provider_class': 'registry', 'list_providers': 'registry', 'provider_service_option_fields': 'registry', 'register_provider': 'registry'}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
