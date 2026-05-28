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
"""Tests for the consolidate_system interceptor — ported from NEL."""

from __future__ import annotations

from nemo_gym.adapters.interceptors.consolidate_system import Interceptor, _content_to_str
from nemo_gym.adapters.types import AdapterRequest, InterceptorContext


def _req(messages: list[dict], session_id: str = "test-session") -> AdapterRequest:
    ctx = InterceptorContext()
    ctx.extra["session_id"] = session_id
    return AdapterRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={"content-type": "application/json"},
        body={"model": "test", "messages": messages},
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# _content_to_str helper
# ---------------------------------------------------------------------------


class TestContentToStr:
    def test_string_passthrough(self):
        assert _content_to_str("hello") == "hello"

    def test_list_of_dicts(self):
        content = [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]
        assert _content_to_str(content) == "part1\npart2"

    def test_list_of_strings(self):
        assert _content_to_str(["a", "b"]) == "a\nb"

    def test_mixed_list(self):
        content = [{"type": "text", "text": "dict-part"}, "str-part"]
        assert _content_to_str(content) == "dict-part\nstr-part"

    def test_none(self):
        assert _content_to_str(None) == ""

    def test_empty_string(self):
        assert _content_to_str("") == ""

    def test_empty_list(self):
        assert _content_to_str([]) == ""

    def test_dict_without_text_key(self):
        assert _content_to_str([{"type": "image_url", "image_url": "x"}]) == ""


# ---------------------------------------------------------------------------
# No-op cases: interceptor should return the request unchanged
# ---------------------------------------------------------------------------


class TestNoOp:
    async def test_empty_messages(self):
        ic = Interceptor()
        req = _req([])
        result = await ic.intercept_request(req)
        assert result.body["messages"] == []

    async def test_no_messages_key(self):
        ic = Interceptor()
        ctx = InterceptorContext()
        req = AdapterRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "test"},
            ctx=ctx,
        )
        result = await ic.intercept_request(req)
        assert "messages" not in result.body

    async def test_single_system_at_pos_0(self):
        ic = Interceptor()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        req = _req(messages)
        result = await ic.intercept_request(req)
        assert result.body["messages"] == messages
        assert ic._fix_count == 0

    async def test_no_system_messages(self):
        ic = Interceptor()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        req = _req(messages)
        result = await ic.intercept_request(req)
        assert result.body["messages"] == messages
        assert ic._fix_count == 0


# ---------------------------------------------------------------------------
# Fix cases: interceptor should consolidate system messages
# ---------------------------------------------------------------------------


class TestConsolidate:
    async def test_system_not_at_pos_0(self):
        """Single system message at position > 0 gets moved to front."""
        ic = Interceptor()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "Be helpful."},
            {"role": "assistant", "content": "hello"},
        ]
        result = await ic.intercept_request(_req(messages))
        msgs = result.body["messages"]
        assert msgs[0] == {"role": "system", "content": "Be helpful."}
        assert msgs[1] == {"role": "user", "content": "hi"}
        assert msgs[2] == {"role": "assistant", "content": "hello"}
        assert ic._fix_count == 1

    async def test_duplicate_system_messages(self):
        """Two system messages get merged into one at position 0."""
        ic = Interceptor()
        messages = [
            {"role": "system", "content": "First."},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "Second."},
            {"role": "assistant", "content": "hello"},
        ]
        result = await ic.intercept_request(_req(messages))
        msgs = result.body["messages"]
        assert len(msgs) == 3
        assert msgs[0] == {"role": "system", "content": "First.\n\nSecond."}
        assert msgs[1] == {"role": "user", "content": "hi"}
        assert msgs[2] == {"role": "assistant", "content": "hello"}

    async def test_empty_system_at_0_real_system_later(self):
        """Empty system at pos 0 + real system later triggers fix."""
        ic = Interceptor()
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "Real prompt."},
        ]
        result = await ic.intercept_request(_req(messages))
        msgs = result.body["messages"]
        assert msgs[0] == {"role": "system", "content": "Real prompt."}
        assert msgs[1] == {"role": "user", "content": "hi"}
        assert len(msgs) == 2

    async def test_list_format_content(self):
        """System message with OpenAI list-format content is handled."""
        ic = Interceptor()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": [{"type": "text", "text": "Be helpful."}]},
        ]
        result = await ic.intercept_request(_req(messages))
        msgs = result.body["messages"]
        assert msgs[0] == {"role": "system", "content": "Be helpful."}
        assert msgs[1] == {"role": "user", "content": "hi"}

    async def test_three_system_messages(self):
        """Three system messages scattered across conversation."""
        ic = Interceptor()
        messages = [
            {"role": "system", "content": "A"},
            {"role": "user", "content": "q1"},
            {"role": "system", "content": "B"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "C"},
        ]
        result = await ic.intercept_request(_req(messages))
        msgs = result.body["messages"]
        assert msgs[0] == {"role": "system", "content": "A\n\nB\n\nC"}
        assert [m["role"] for m in msgs[1:]] == ["user", "assistant"]

    async def test_system_without_content_key(self):
        """System message with no content key at all doesn't crash."""
        ic = Interceptor()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "system"},
        ]
        result = await ic.intercept_request(_req(messages))
        msgs = result.body["messages"]
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# Non-system message ordering is preserved
# ---------------------------------------------------------------------------


class TestOrderPreservation:
    async def test_non_system_order_preserved(self):
        ic = Interceptor()
        messages = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        result = await ic.intercept_request(_req(messages))
        msgs = result.body["messages"]
        assert msgs[0]["role"] == "system"
        non_system = msgs[1:]
        assert [m["content"] for m in non_system] == ["u1", "a1", "u2", "a2"]

    async def test_extra_message_fields_preserved(self):
        """Fields like tool_calls, name, etc. are not lost."""
        ic = Interceptor()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "ok", "tool_calls": [{"id": "1"}]},
        ]
        result = await ic.intercept_request(_req(messages))
        msgs = result.body["messages"]
        assert msgs[2]["tool_calls"] == [{"id": "1"}]


# ---------------------------------------------------------------------------
# Custom separator
# ---------------------------------------------------------------------------


class TestCustomSeparator:
    async def test_custom_separator(self):
        ic = Interceptor(separator=" | ")
        messages = [
            {"role": "system", "content": "A"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "B"},
        ]
        result = await ic.intercept_request(_req(messages))
        assert result.body["messages"][0]["content"] == "A | B"


# ---------------------------------------------------------------------------
# Idempotency and fix counter
# ---------------------------------------------------------------------------


class TestIdempotency:
    async def test_idempotent_after_fix(self):
        """Running the interceptor twice on the fixed output is a no-op."""
        ic = Interceptor()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "sys"},
        ]
        result1 = await ic.intercept_request(_req(messages))
        assert ic._fix_count == 1

        result2 = await ic.intercept_request(_req(result1.body["messages"]))
        assert ic._fix_count == 1
        assert result2.body["messages"] == result1.body["messages"]

    async def test_fix_counter_increments(self):
        ic = Interceptor()
        for i in range(3):
            messages = [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": f"sys-{i}"},
            ]
            await ic.intercept_request(_req(messages))
        assert ic._fix_count == 3


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistry:
    async def test_create_via_registry(self):
        from nemo_gym.adapters.registry import InterceptorRegistry

        ic = InterceptorRegistry.create("consolidate_system", {})
        assert isinstance(ic, Interceptor)

    async def test_create_with_separator(self):
        from nemo_gym.adapters.registry import InterceptorRegistry

        ic = InterceptorRegistry.create("consolidate_system", {"separator": "---"})
        assert ic._sep == "---"
