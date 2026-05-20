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
"""Phase-1 smoke test for the Gym adapter framework.

Confirms the framework imports, an empty pipeline raises the expected
no-response error, and the registry pre-lists all 14 builtin names even
though Phase-2 interceptor modules haven't landed yet.
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


# Names registered in nemo_gym.adapters.registry._BUILTIN. Kept in this test
# as a hard-coded expectation so the smoke test catches accidental drift.
_EXPECTED_BUILTINS = {
    "caching",
    "consolidate_system",
    "drop_params",
    "endpoint",
    "log_tokens",
    "logging",
    "modify_tools",
    "payload_modifier",
    "progress_tracking",
    "raise_client_errors",
    "reasoning",
    "response_stats",
    "system_message",
    "turn_counter",
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


def test_registry_pre_lists_all_14_builtins() -> None:
    """``available()`` returns the 14 builtin names without importing them.

    Phase-2 interceptor modules don't exist yet, so ``resolve_class()`` would
    raise. ``available()`` is intentionally cheap and string-only.
    """
    names = set(InterceptorRegistry.available())
    assert names == _EXPECTED_BUILTINS, (
        f"Expected exactly the 14 builtin names; got {sorted(names)}"
    )
    assert len(names) == 14
