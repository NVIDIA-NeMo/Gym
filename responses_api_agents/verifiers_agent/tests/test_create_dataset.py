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
"""Tests for the verifiers_agent dataset creation script."""

import importlib.util
from pathlib import Path

import pytest


# create_dataset.py imports `verifiers` at module scope, which is installed in
# this server's venv rather than the core one. Skip cleanly when absent.
pytest.importorskip("verifiers", reason="verifiers is installed in the verifiers_agent venv")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "create_dataset.py"
_spec = importlib.util.spec_from_file_location("vf_create_dataset", _SCRIPT)
create_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(create_dataset)

_as_dict = create_dataset._as_dict


class TestAsDict:
    """`info` must reach the agent as a dict.

    HuggingFace `datasets` serializes structurally heterogeneous columns to JSON
    strings, so environments with non-trivial `info` yield a `str` here. Passing
    that through verbatim makes the agent's `/run` reject the row with
    `422 body.info: Input should be a valid dictionary`.
    """

    def test_dict_passes_through_unchanged(self):
        info = {"initial_state": {"meta": {"schema_version": 1}}}
        assert _as_dict(info) is info

    def test_empty_dict_is_preserved(self):
        # acereason-math ships `info == {}`, which is why it never hit the bug.
        assert _as_dict({}) == {}

    def test_json_string_is_decoded(self):
        assert _as_dict('{"zapier_tools": ["gmail_send_email"], "n": 1}') == {
            "zapier_tools": ["gmail_send_email"],
            "n": 1,
        }

    def test_nested_json_string_is_decoded(self):
        raw = '{"initial_state": {"records": [{"id": "a"}]}}'
        assert _as_dict(raw) == {"initial_state": {"records": [{"id": "a"}]}}

    def test_malformed_json_string_is_preserved_as_raw(self):
        assert _as_dict("not json at all") == {"raw": "not json at all"}

    def test_json_string_decoding_to_non_dict_is_wrapped(self):
        # Valid JSON, but a list/scalar still isn't acceptable to the agent.
        assert _as_dict("[1, 2, 3]") == {"raw": "[1, 2, 3]"}
        assert _as_dict("42") == {"raw": "42"}

    def test_none_becomes_empty_dict(self):
        assert _as_dict(None) == {}

    def test_other_types_are_wrapped(self):
        assert _as_dict(7) == {"raw": 7}

    def test_result_is_always_a_dict(self):
        for value in [{}, {"a": 1}, '{"a": 1}', "bad", "[1]", None, 7]:
            assert isinstance(_as_dict(value), dict)
