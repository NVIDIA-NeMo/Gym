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

"""Typed configuration for sandbox providers.

Defaults for these fields belong in YAML exemplars. Code should require keys
from enabled configs instead of silently supplying behavior here.
"""

from typing import Any, NotRequired, TypedDict


class SandboxProviderConfig(TypedDict):
    """Underlying runtime and infrastructure provider.

    Keys:
        name: Provider registry name, for example ``opensandbox``.
        kwargs: Provider-specific constructor settings such as OpenSandbox
            domain, API key, or proxy mode.
    """

    name: str
    kwargs: NotRequired[dict[str, Any]]
