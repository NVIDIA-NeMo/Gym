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
"""Unit tests for the speed_bench multi-turn replay agent.

The full `responses()` flow exercises several Gym layers (server_client,
HTTP, OpenAI types). Rather than re-mocking the full chain, these tests
assert the *turn-splitting* logic that is unique to this agent — specifically
that trailing user messages are correctly identified.
"""

from app import _content_text


def test_content_text_string_passthrough():
    assert _content_text("hello") == "hello"


def test_content_text_list_with_text_attr():
    class _Item:
        text = "world"

    assert _content_text([_Item(), _Item()]) == "worldworld"


def test_content_text_list_with_dict_text_key():
    assert _content_text([{"text": "a"}, {"text": "b"}]) == "ab"


def test_content_text_list_falls_back_to_content_key():
    assert _content_text([{"content": "x"}]) == "x"


def test_content_text_list_skips_unknown_items():
    assert _content_text([{"foo": "bar"}, {"text": "ok"}]) == "ok"


def test_content_text_other_types_returns_empty():
    assert _content_text(None) == ""
    assert _content_text(42) == ""
