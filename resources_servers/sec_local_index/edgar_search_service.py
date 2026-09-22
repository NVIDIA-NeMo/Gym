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
"""Everything an ``edgar_search`` call does apart from fetching the results.

A search makes the same decisions whichever server hosts it and whichever
backend answers it: coerce the arguments a tool-call parser produced, normalize
and clamp them, run them, and serialize the outcome. Owning those here leaves
each server an adapter that chooses a backend and a cutoff date, so the two
cannot answer the same request differently.
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any, Awaitable, Callable, Literal, Optional, Protocol

from resources_servers.sec_local_index.local_edgar_search import LocalEdgarRequest, normalize_request


logger = logging.getLogger(__name__)

COLLECTION_ARGUMENTS = ("form_types", "ciks")

SecMode = Literal["live", "local"]


def resolve_sec_mode(configured: Optional[str], local_index_path: Optional[str]) -> SecMode:
    """Choose a backend, falling back to whatever the configured artifacts imply.

    Inferring keeps configurations written before sec_mode existed working
    unchanged: those turned the local index on by setting its path alone.
    """
    if configured is not None:
        if configured not in ("live", "local"):
            raise ValueError(f"sec_mode must be 'live' or 'local', got {configured!r}")
        return configured
    return "local" if local_index_path else "live"


class EdgarSearchBackend(Protocol):
    """Answers a normalized request with sec-api-shaped filing records."""

    async def execute_async(self, request: LocalEdgarRequest) -> list[dict[str, Any]]: ...


def coerce_stringified_collection(value: Any) -> Any:
    """Deserialize a stringified list/dict into its native Python type.

    Tool-call parsers may serialize nested arguments as strings rather than
    native types, in either JSON (``'["a", "b"]'``) or Python repr
    (``"['a', 'b']"``) form. Values that parse as neither are returned
    unchanged so the caller reports its own validation error.
    """
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
        if isinstance(parsed, (list, dict)):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, dict)):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return value


class EdgarSearchService:
    """Runs ``edgar_search`` against a backend and renders the tool output."""

    def __init__(
        self,
        backend: EdgarSearchBackend,
        *,
        max_end_date: str,
        on_results: Optional[Callable[[list[dict[str, Any]]], Awaitable[None]]] = None,
    ) -> None:
        self._backend = backend
        self._max_end_date = max_end_date
        self._on_results = on_results

    @property
    def max_end_date(self) -> str:
        return self._max_end_date

    def normalize(self, arguments: dict[str, Any]) -> LocalEdgarRequest:
        coerced = dict(arguments)
        for name in COLLECTION_ARGUMENTS:
            if name in coerced:
                coerced[name] = coerce_stringified_collection(coerced[name])
        # An omitted bound means "as far as the corpus goes", which on the upper
        # side is the cutoff rather than today.
        if not coerced.get("end_date"):
            coerced["end_date"] = self._max_end_date
        if not coerced.get("start_date"):
            coerced.pop("start_date", None)
        return normalize_request(**coerced, max_end_date=self._max_end_date)

    async def run(self, arguments: dict[str, Any]) -> str:
        """Return the serialized tool output for one set of raw arguments."""
        try:
            results = await self._backend.execute_async(self.normalize(arguments))
        except Exception as error:
            logger.warning("edgar_search failed: %s", error)
            return json.dumps({"error": str(error)})

        if self._on_results is not None:
            await self._on_results(results)
        return json.dumps(results, default=str)
