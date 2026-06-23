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

"""SWE-env sandbox providers.

Importing this package registers the providers with ``nemo_gym.sandbox`` so a
config like ``provider: {docker: {...}}`` resolves. ``docker`` runs sandboxes
as local Docker containers; ``apptainer`` runs them as ``apptainer instance``
processes from ``.sif`` images for on-prem clusters.
"""

from nemo_gym.sandbox import list_providers, register_provider
from responses_api_agents.swe_env.providers.apptainer_provider import ApptainerSandboxProvider
from responses_api_agents.swe_env.providers.docker_provider import DockerSandboxProvider


def register_swe_env_providers() -> None:
    """Register the swe_env sandbox providers, skipping any already registered.

    Registers the ``docker`` and ``apptainer`` providers with
    ``nemo_gym.sandbox``. Safe to call multiple times; providers already present
    in the registry are left untouched.
    """
    existing = set(list_providers())
    if "docker" not in existing:
        register_provider("docker", DockerSandboxProvider)
    if "apptainer" not in existing:
        register_provider("apptainer", ApptainerSandboxProvider)


register_swe_env_providers()


__all__ = ["ApptainerSandboxProvider", "DockerSandboxProvider", "register_swe_env_providers"]
