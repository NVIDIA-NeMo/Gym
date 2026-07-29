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

from nemo_gym.orchestration.executors.utils import flatten_run_args as _flatten_run_args


def test_scalar_values():
    assert _flatten_run_args({"temperature": 0.05, "top_p": 0.9}) == [
        "+temperature=0.05",
        "+top_p=0.9",
    ]


def test_nested_dict():
    assert _flatten_run_args({"responses_create_params": {"max_concurrent": 92, "temperature": 0.05}}) == [
        "+responses_create_params.max_concurrent=92",
        "+responses_create_params.temperature=0.05",
    ]


def test_list_value():
    assert _flatten_run_args({"config_paths": ["benchmarks/gsm8k/config.yaml", "benchmarks/foo/config.yaml"]}) == [
        "'+config_paths=[benchmarks/gsm8k/config.yaml,benchmarks/foo/config.yaml]'",
    ]


def test_empty():
    assert _flatten_run_args({}) == []


def test_value_with_spaces_is_quoted():
    result = _flatten_run_args({"name": "my model"})
    assert result == ["'+name=my model'"]


def test_deeply_nested():
    assert _flatten_run_args({"a": {"b": {"c": 1}}}) == ["+a.b.c=1"]
