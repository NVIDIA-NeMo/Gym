# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tool-call identity in the lineage hashes: a harness that rewrites a served call but keeps its id."""

from __future__ import annotations

import pytest

from nemo_gym.token_id_capture import fingerprint as fp
from nemo_gym.token_id_capture.lineage import FileLineageStore
from nemo_gym.token_id_capture.records import ParentResolutionStatus
from tests.unit_tests.test_token_capture_ledger import _call_record, _commit


USER = {"role": "user", "content": "fix the bug"}
# The served turn: the model called a tool that does not exist.
SERVED = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "chatcmpl-tool-1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
        }
    ],
}
# What OpenCode echoes back for that turn: same id, its own placeholder name and arguments.
ECHOED = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "chatcmpl-tool-1",
            "type": "function",
            "function": {"name": "invalid", "arguments": '{"tool": "read_file", "error": "Tool read_file not found"}'},
        }
    ],
}
TOOL_RESULT = {"role": "tool", "tool_call_id": "chatcmpl-tool-1", "content": "Tool read_file not found"}
FOLLOW_UP = {"role": "user", "content": "try again"}


@pytest.fixture
def identity(monkeypatch):
    def set_mode(mode: str) -> None:
        monkeypatch.setattr(fp, "_TOOL_CALL_IDENTITY", mode)

    return set_mode


async def _resolve_echo(tmp_path):
    store = FileLineageStore(tmp_path)
    await store.record(_commit(_call_record("c1"), [USER], [SERVED]))
    return await store.resolve("r1", [USER, ECHOED, TOOL_RESULT, FOLLOW_UP])


@pytest.mark.asyncio
async def test_full_identity_rejects_a_rewritten_tool_call(tmp_path, identity):
    identity("full")
    resolution = await _resolve_echo(tmp_path)
    assert resolution.status == ParentResolutionStatus.UNRESOLVED
    assert resolution.reason == "no_match"


@pytest.mark.asyncio
async def test_id_identity_resolves_a_rewritten_tool_call(tmp_path, identity):
    identity("id")
    resolution = await _resolve_echo(tmp_path)
    assert resolution.status == ParentResolutionStatus.RESOLVED
    assert resolution.match is not None and resolution.match.model_call_id == "c1"


@pytest.mark.asyncio
async def test_id_identity_still_requires_the_same_call_id(tmp_path, identity):
    identity("id")
    store = FileLineageStore(tmp_path)
    await store.record(_commit(_call_record("c1"), [USER], [SERVED]))
    renumbered = {**ECHOED, "tool_calls": [{**ECHOED["tool_calls"][0], "id": "call_other"}]}
    resolution = await store.resolve("r1", [USER, renumbered, TOOL_RESULT, FOLLOW_UP])
    assert resolution.status == ParentResolutionStatus.UNRESOLVED


def test_modes_hash_into_different_domains(identity):
    identity("full")
    full = fp.assistant_fingerprint([USER, SERVED])
    identity("id")
    by_id = fp.assistant_fingerprint([USER, SERVED])
    assert full != by_id
    # Under "id" the served and echoed turns agree; under "full" they do not.
    assert fp.assistant_fingerprint([USER, ECHOED]) == by_id
    identity("full")
    assert fp.assistant_fingerprint([USER, ECHOED]) != full
