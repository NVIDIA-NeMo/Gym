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
"""InterceptorRegistry tests — ported from NEL ``tests/test_adapters/test_registry.py``."""

import pytest

from nemo_gym.adapters.registry import InterceptorRegistry


def test_resolve_known_interceptor():
    cls = InterceptorRegistry.resolve_class("drop_params")
    assert cls.__name__ == "Interceptor"
    assert cls.__module__ == "nemo_gym.adapters.interceptors.drop_params"


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown interceptor 'nonexistent'"):
        InterceptorRegistry.resolve_class("nonexistent")


def test_create_with_config():
    instance = InterceptorRegistry.create("drop_params", {"params": ["foo"]})
    from nemo_gym.adapters.interceptors.drop_params import Interceptor

    assert isinstance(instance, Interceptor)


def test_available_list():
    names = InterceptorRegistry.available()
    assert isinstance(names, list)
    for expected in ("drop_params", "endpoint", "system_message", "turn_counter"):
        assert expected in names
