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

import pytest

from nemo_gym.orchestration.ray_serve_vllm_app import _to_bool


def test_to_bool_passthrough():
    assert _to_bool(True) is True
    assert _to_bool(False) is False


@pytest.mark.parametrize("value", ["true", "True", "1", "yes"])
def test_to_bool_truthy_strings(value):
    assert _to_bool(value) is True


@pytest.mark.parametrize("value", ["false", "False", "0", "no", ""])
def test_to_bool_falsy_strings(value):
    assert _to_bool(value) is False


def test_build_app_requires_vllm_and_ray():
    pytest.importorskip("vllm")
    pytest.importorskip("ray.serve")
    from nemo_gym.orchestration.ray_serve_vllm_app import build_app

    # Mirrors serve run's builder-function convention: args always arrive as a single dict of
    # strings, never as individual keyword parameters.
    app = build_app({"model": "org/model", "tensor_parallel_size": "2", "number_of_instances": "4"})
    assert app is not None


def test_build_app_requires_model_key():
    pytest.importorskip("vllm")
    pytest.importorskip("ray.serve")
    from nemo_gym.orchestration.ray_serve_vllm_app import build_app

    with pytest.raises(KeyError):
        build_app({})
