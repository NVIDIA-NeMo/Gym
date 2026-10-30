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
"""The middleware that strips a rollout prefix from a request path and sets the current episode for the request.

Tracking the current episode lives in `nemo_gym.core.episode`.
"""

import re
from typing import Any

from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.config.config_types import ROLLOUT_PATH_PREFIX
from nemo_gym.core.episode import (
    ROLLOUT_ID_PATTERN,
    rollout_context,
)


ROUTED_PATH_STATE_KEY = "nemo_gym.routed_path"


def scope_with_routed_path(scope: dict[str, Any], path: str) -> dict[str, Any]:
    """Copy the scope for routing and share the rewritten path with outer middleware."""
    scope.setdefault("state", {})[ROUTED_PATH_STATE_KEY] = path
    return {**scope, "path": path, "raw_path": path.encode("utf-8")}


class RolloutContextMiddleware:
    """Strip a rollout prefix and expose it to downstream Gym calls for this request."""

    # Correlation prefixes are handled before resource routes; direct MCP dispatch has no prefix.
    skipped_on_direct_mcp_dispatch = True

    # Match the same id characters as ``ROLLOUT_ID_PATTERN``.
    # Anchor the id between the prefix and the remaining path.
    _PREFIX = re.compile(
        rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<rollout_id>{ROLLOUT_ID_PATTERN.pattern.strip('^$')})(?P<rest>/.*)$"
    )

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        match = self._PREFIX.match(scope.get("path", "")) if scope.get("type") == "http" else None
        if match is None:
            await self._app(scope, receive, send)
            return

        path = match.group("rest")
        scope = scope_with_routed_path(scope, path)
        with rollout_context(match.group("rollout_id")):
            await self._app(scope, receive, send)


# Names that moved to other modules stay importable from here until their removal release; see nemo_gym/_moved_paths.py.
__getattr__ = moved_names_getattr(__name__)
