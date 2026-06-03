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
"""Smoke test for the Gym adapter framework.

Confirms the framework imports, an empty pipeline raises the expected
no-response error, and the registry pre-lists every builtin name.
"""

from __future__ import annotations

import pytest

from nemo_gym.adapters import (
    AdapterPipeline,
    AdapterRequest,
    InterceptorContext,
    InterceptorRegistry,
    install_middleware,
)


# Names this PR ships in ``nemo_gym.adapters.registry._BUILTIN``. Follow-on
# PRs (observability / caching / request-rewriting) extend the set —
# assertion below is a subset check so adding new builtins doesn't require
# editing this file.
_FRAMEWORK_BUILTINS = {
    "endpoint",
    "logging",
}


def test_framework_imports() -> None:
    """All public symbols import cleanly from the package."""
    assert callable(install_middleware)
    assert AdapterPipeline is not None
    assert InterceptorRegistry is not None


@pytest.mark.asyncio
async def test_empty_pipeline_raises_no_response_error() -> None:
    """A pipeline with no interceptors cannot produce a response."""
    pipeline = AdapterPipeline([])
    req = AdapterRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={},
        body={"model": "test"},
        ctx=InterceptorContext(),
    )

    with pytest.raises(RuntimeError, match="No interceptor produced a response"):
        await pipeline.process(req)


def test_registry_pre_lists_framework_builtins() -> None:
    """``available()`` includes every framework-level builtin name.

    Subset check (not equality) so this assertion is stable as follow-on
    interceptor families extend the builtin set without touching this file.
    """
    names = set(InterceptorRegistry.available())
    missing = _FRAMEWORK_BUILTINS - names
    assert not missing, f"Framework builtins missing from registry: {sorted(missing)}"
