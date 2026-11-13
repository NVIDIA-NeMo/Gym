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

"Sandbox provider registry."

from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.sandbox.providers.base import (
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
        SupportsSandboxBackgroundServices,
        SupportsSandboxEndpoint,
        SupportsSandboxIndependentUploads,
        SupportsSandboxNetwork,
        SupportsSandboxPauseResume,
        SupportsSandboxPortForwarding,
        SupportsSandboxPty,
        SupportsSandboxPtyAttach,
        SupportsSandboxPullPolicy,
        SupportsSandboxRuntimeRequirements,
        SupportsSandboxServiceOptions,
        SupportsSandboxSharedStorage,
        SupportsSandboxSidecars,
        clamp_resource_requests,
    )
    from nemo_gym.sandbox.providers.registry import (
        create_provider,
        get_provider_class,
        list_providers,
        provider_service_option_fields,
        register_provider,
    )

_PUBLIC_API = {
    "ConnectableProvider": "base",
    "ExecResult": "base",
    "SandboxCreateError": "base",
    "SandboxCreateVerificationError": "base",
    "SandboxEndpoint": "base",
    "SandboxExecResult": "base",
    "SandboxHandle": "base",
    "SandboxProvider": "base",
    "SandboxPtyError": "base",
    "SandboxPtySession": "base",
    "SandboxPtySpec": "base",
    "SandboxResources": "base",
    "SandboxSidecarSpec": "base",
    "SandboxSpec": "base",
    "SandboxStatus": "base",
    "SupportsSandboxBackgroundServices": "base",
    "SupportsSandboxEndpoint": "base",
    "SupportsSandboxIndependentUploads": "base",
    "SupportsSandboxNetwork": "base",
    "SupportsSandboxPauseResume": "base",
    "SupportsSandboxPortForwarding": "base",
    "SupportsSandboxPty": "base",
    "SupportsSandboxPtyAttach": "base",
    "SupportsSandboxPullPolicy": "base",
    "SupportsSandboxRuntimeRequirements": "base",
    "SupportsSandboxServiceOptions": "base",
    "SupportsSandboxSharedStorage": "base",
    "SupportsSandboxSidecars": "base",
    "clamp_resource_requests": "base",
    "create_provider": "registry",
    "get_provider_class": "registry",
    "list_providers": "registry",
    "provider_service_option_fields": "registry",
    "register_provider": "registry",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
