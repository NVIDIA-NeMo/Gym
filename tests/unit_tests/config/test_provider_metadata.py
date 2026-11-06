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

from types import SimpleNamespace

import pytest

import nemo_gym.config.provider_metadata as provider_metadata


def test_plugin_distributions_names_each_distribution_once(monkeypatch: pytest.MonkeyPatch) -> None:
    eps = [
        SimpleNamespace(name="a", dist=SimpleNamespace(name="plugin-b")),
        SimpleNamespace(name="b", dist=SimpleNamespace(name="plugin-b")),
        SimpleNamespace(name="c", dist=SimpleNamespace(name="plugin-a")),
        SimpleNamespace(name="d", dist=None),
    ]
    monkeypatch.setattr(provider_metadata, "entry_points", lambda group: eps)
    assert provider_metadata.plugin_distributions() == ["plugin-a", "plugin-b"]
