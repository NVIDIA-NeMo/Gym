# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Lazy Modal SDK loading.

Kept as a one-function seam (mirroring ``nemo_gym.sandbox.providers.e2b._sdk``)
so provider unit tests can substitute a fake SDK module without importing or
authenticating against Modal.
"""

from typing import Any


# 1.5.5 supplies the streaming filesystem and termination contracts exercised
# by the provider. Bound the major version because Modal 2.x may change them.
MODAL_SDK_CONSTRAINT = "modal>=1.5.5,<2.0.0"


def require_modal_sdk(feature: str) -> Any:
    """Import ``modal`` lazily with a clear install hint when it is absent."""
    try:
        import modal
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        if exc.name != "modal":
            # A broken transitive dependency is not "modal is missing"; keep the real error.
            raise
        raise ImportError(
            f"{feature} requires the 'modal' package. Install it with `pip install '{MODAL_SDK_CONSTRAINT}'` "
            "and authenticate with `modal token new` (or set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET)."
        ) from exc
    return modal
