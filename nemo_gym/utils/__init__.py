# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""General helpers shared across nemo_gym: paths, decorators, text, process information, and import compatibility."""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.utils.path_utils import (
        aggregation_sources_path_for,
        failures_path_for,
        materialized_path_for,
        resubmit_offset_path_for,
    )
    from nemo_gym.utils.server_process import is_nemo_gym_fastapi_entrypoint


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "aggregation_sources_path_for": "path_utils",
    "failures_path_for": "path_utils",
    "is_nemo_gym_fastapi_entrypoint": "server_process",
    "materialized_path_for": "path_utils",
    "resubmit_offset_path_for": "path_utils",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
