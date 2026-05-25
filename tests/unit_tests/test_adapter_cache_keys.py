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
"""Cache-key derivation golden tests — ported from NEL.

The golden SHA-256 strings are byte-identical to NEL's: the cache_key
algorithm is the same code re-rooted at ``nemo_gym.adapters.cache``, so
both implementations must produce the same digests for the same inputs.
This file protects against accidental drift.
"""

from nemo_gym.adapters.cache.disk_cache import DiskCache


def _compute_key(body):
    return DiskCache.cache_key(body)


def test_golden_key_simple():
    body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.7}
    key = _compute_key(body)
    assert key == _compute_key(body)
    assert key == "92232807c9b5c0bab68fd1abb38bc83c884fb04b0eb9b8b453c19daead7d682f"  # pragma: allowlist secret


def test_golden_key_with_tools():
    body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    }
    key = _compute_key(body)
    assert key == _compute_key(body)
    assert key == "ce43d524c627c96eaadbb9abdac38ad5eb60a93d15b1d26880fa1fe0f86d4c68"  # pragma: allowlist secret


def test_key_ignores_irrelevant_fields():
    base = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
    with_stream = {**base, "stream": True}
    assert _compute_key(base) == _compute_key(with_stream)


def test_key_changes_with_temperature():
    body_a = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.7}
    body_b = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.9}
    assert _compute_key(body_a) != _compute_key(body_b)


def test_key_differs_with_session_prefix():
    """Session prefix ensures repeats of the same problem never share cache entries."""
    body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
    k_none = DiskCache.cache_key(body)
    k_a = DiskCache.cache_key(body, session_prefix="repeat-0")
    k_b = DiskCache.cache_key(body, session_prefix="repeat-1")
    assert k_none != k_a
    assert k_a != k_b
    assert k_none == DiskCache.cache_key(body, session_prefix="")
