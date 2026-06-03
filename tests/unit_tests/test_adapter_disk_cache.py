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
"""DiskCache behavior tests — ported from NEL."""

import asyncio

import pytest

from nemo_gym.adapters.cache.disk_cache import DiskCache


@pytest.fixture
def cache(tmp_path):
    return DiskCache(str(tmp_path / "cache"))


def test_cache_key_deterministic():
    body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
    assert DiskCache.cache_key(body) == DiskCache.cache_key(body)


def test_cache_key_varies_with_model():
    base = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
    other = {**base, "model": "gpt-3.5-turbo"}
    assert DiskCache.cache_key(base) != DiskCache.cache_key(other)


async def test_get_set_roundtrip(cache):
    key = DiskCache.cache_key({"model": "x", "messages": []})
    value = {"choices": [{"message": {"content": "hi"}}]}
    await cache.set(key, value)
    result = await cache.get(key)
    assert result == value


async def test_get_missing_returns_none(cache):
    result = await cache.get("nonexistent_key")
    assert result is None


async def test_concurrent_writes(cache):
    pairs = [(f"key_{i}", {"i": i}) for i in range(50)]

    await asyncio.gather(*(cache.set(k, v) for k, v in pairs))

    for k, v in pairs:
        assert await cache.get(k) == v


async def test_degrade_on_bad_path():
    bad = DiskCache.__new__(DiskCache)
    bad._db_path = "/dev/null/bad/adapter_cache.db"
    bad._lock = __import__("threading").Lock()

    assert await bad.get("anything") is None
    await bad.set("anything", {"x": 1})
