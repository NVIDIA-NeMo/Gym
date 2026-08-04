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

"""Lazy E2B SDK loading and NeMo Gym traffic attribution.

The public E2B high-level API owns its pooled HTTP and ConnectRPC transports;
this integration deliberately avoids private transport monkeypatches.
"""

import threading
from typing import Any

from nemo_gym.package_info import __version__


E2B_SDK_CONSTRAINT = "e2b>=2.36.0,<3.0.0"
_INTEGRATION = f"nemo-gym/{__version__}"
_CONFIGURED_SDK_MODULES: dict[int, Any] = {}
_CONFIGURE_LOCK = threading.Lock()


def require_e2b_sdk(feature: str) -> Any:
    """Import E2B lazily and attribute this process's NeMo Gym SDK traffic."""
    try:
        import e2b
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        if exc.name != "e2b":
            # Preserve the actual missing transitive module from a broken SDK
            # installation instead of incorrectly claiming E2B is absent.
            raise
        raise ImportError(
            f"{feature} requires the 'e2b' package. Install it with `pip install '{E2B_SDK_CONSTRAINT}'`."
        ) from exc

    module_id = id(e2b)
    with _CONFIGURE_LOCK:
        if _CONFIGURED_SDK_MODULES.get(module_id) is not e2b:
            e2b.ConnectionConfig.set_integration(_INTEGRATION)
            _CONFIGURED_SDK_MODULES[module_id] = e2b
    return e2b
