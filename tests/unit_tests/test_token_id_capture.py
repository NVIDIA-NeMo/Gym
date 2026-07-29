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
"""Training-token capture: schema, store, readers, source, and the served path.

The served-path tests build a real ``SimpleResponsesAPIModel`` so the full chain runs:
the capture middleware mints a ``model_call_id`` and sets a per-request token sink, the
model server records a ``TokenEntry`` from its complete response, and the entry is read
back through the store, the HTTP route, and a ``TokenSource``.
"""

import asyncio
import subprocess
import sys
from time import time
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import Body, Request
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    CaptureStore,
    SimpleResponsesAPIModel,
    read_model_call_records,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture import (
    TOKEN_FIELDS,
    TokenCaptureStore,
    TokenEntry,
    TokenIdCaptureConfig,
    compute_digest,
    cumulative_tokens,
    extract_token_fields,
    stamp_lineage,
)
from nemo_gym.token_id_capture import reader as reader_module
from nemo_gym.token_id_capture.lineage import LineageIndex, RolloutLineage, assistant_fingerprint
from nemo_gym.token_id_capture.reader import HttpTokenReader, LocalTokenReader
from nemo_gym.token_id_capture.routes import make_token_store


PTOKS = [1, 2, 3]
GTOKS = [4, 5]
LPS = [-0.1, -0.2]


# --- schema / extractor -------------------------------------------------------


def test_extract_token_fields_responses_shape():
    payload = {
        "output": [
            {"type": "message", "prompt_token_ids": PTOKS, "generation_token_ids": GTOKS, "generation_log_probs": LPS}
        ]
    }
    assert extract_token_fields(payload) == {
        "prompt_token_ids": PTOKS,
        "generation_token_ids": GTOKS,
        "generation_log_probs": LPS,
        "routed_experts": None,
    }


def test_extract_token_fields_chat_shape():
    payload = {
        "choices": [
            {"message": {"prompt_token_ids": [1], "generation_token_ids": [7], "generation_log_probs": [-0.3]}}
        ]
    }
    got = extract_token_fields(payload)
    assert got["generation_token_ids"] == [7] and got["prompt_token_ids"] == [1]


def test_extract_token_fields_absent_returns_none():
    assert extract_token_fields({"output": [{"type": "message"}]}) is None
    assert extract_token_fields({}) is None


# --- store --------------------------------------------------------------------


def test_token_store_round_trip(tmp_path):
    store = TokenCaptureStore(tmp_path)
    entry = TokenEntry(
        rollout_id="t0-r0",
        model_call_id="abc",
        model="m",
        prompt_token_ids=PTOKS,
        generation_token_ids=GTOKS,
        generation_log_probs=LPS,
    )
    store.append(entry)
    store.append(entry.model_copy(update={"model_call_id": "def"}))
    read = store.read_entries("t0-r0")
    assert [e.model_call_id for e in read] == ["abc", "def"]
    assert read[0].prompt_token_ids == PTOKS
    assert store.read_entries("missing") == []


@pytest.mark.parametrize("bad", ["", "a/b", "../x", "a b"])
def test_token_store_rejects_unsafe_rollout_ids(tmp_path, bad):
    with pytest.raises(ValueError):
        TokenCaptureStore(tmp_path).path_for(bad)


# --- config -------------------------------------------------------------------


def test_config_disabled_needs_no_dir():
    cfg = TokenIdCaptureConfig.model_validate({})
    assert cfg.token_id_capture_enabled is False
    assert make_token_store({}) is None


def test_config_enabled_requires_absolute_dir(tmp_path):
    with pytest.raises(ValueError):
        TokenIdCaptureConfig(token_id_capture_enabled=True)
    with pytest.raises(ValueError):
        TokenIdCaptureConfig(token_id_capture_enabled=True, token_id_capture_dir="relative/dir")
    cfg = TokenIdCaptureConfig(token_id_capture_enabled=True, token_id_capture_dir=str(tmp_path))
    assert cfg.resolved_dir() == tmp_path


def test_config_falls_back_to_model_call_capture_dir(tmp_path):
    cfg = TokenIdCaptureConfig(token_id_capture_enabled=True, model_call_capture_dir=str(tmp_path))
    assert cfg.resolved_dir() == tmp_path


# --- source / readers ---------------------------------------------------------


def test_capture_token_source_over_local_reader(tmp_path):
    store = TokenCaptureStore(tmp_path)
    store.append(
        TokenEntry(
            rollout_id="r",
            model_call_id="c",
            prompt_token_ids=PTOKS,
            generation_token_ids=GTOKS,
            generation_log_probs=LPS,
        )
    )
    source = LocalTokenReader(store)
    entries = asyncio.run(source.tokens_for("r"))
    assert len(entries) == 1 and entries[0].generation_token_ids == GTOKS


def test_http_token_reader_parses_ndjson(monkeypatch):
    entry = TokenEntry(
        rollout_id="r", model_call_id="c", prompt_token_ids=PTOKS, generation_token_ids=GTOKS, generation_log_probs=LPS
    )
    body = entry.model_dump_json() + "\n"

    class _FakeResp:
        async def text(self):
            return body

    async def _fake_request(method, url, **kwargs):
        assert method == "GET" and url.endswith("/ng-capture/tokens/r")
        return _FakeResp()

    async def _fake_raise(_resp):
        return None

    monkeypatch.setattr(reader_module, "request", _fake_request)
    monkeypatch.setattr(reader_module, "raise_for_status", _fake_raise)
    entries = asyncio.run(HttpTokenReader("http://model:9000").tokens_for("r"))
    assert len(entries) == 1 and entries[0].model_call_id == "c"


# --- served path (full model server) -----------------------------------------


def _training_response(text: str, model: str = "downstream-model") -> NeMoGymResponse:
    return NeMoGymResponse(
        id=f"resp_{uuid4().hex}",
        created_at=int(time()),
        model=model,
        object="response",
        output=[
            {
                "type": "message",
                "id": f"msg_{uuid4().hex}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
                "prompt_token_ids": PTOKS,
                "generation_token_ids": GTOKS,
                "generation_log_probs": LPS,
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


def _training_chat_completion(model: str = "downstream-model") -> NeMoGymChatCompletion:
    return NeMoGymChatCompletion.model_validate(
        {
            "id": f"chatcmpl_{uuid4().hex}",
            "created": int(time()),
            "model": model,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "prompt_token_ids": PTOKS,
                        "generation_token_ids": GTOKS,
                        "generation_log_probs": LPS,
                    },
                }
            ],
        }
    )


class _CapturingModel(SimpleResponsesAPIModel):
    config: BaseResponsesAPIModelConfig
    model_config = {"arbitrary_types_allowed": True}

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        return _training_response("hi from responses")

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        return _training_chat_completion()


def _server(global_config_dict) -> SimpleResponsesAPIModel:
    return _CapturingModel(
        config=BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="srv"),
        server_client=MagicMock(spec=ServerClient, global_config_dict=global_config_dict),
    )


def _both_enabled(tmp_path) -> dict:
    return {
        "observability_enabled": True,
        "model_call_capture_dir": str(tmp_path),
        "token_id_capture_enabled": True,
        "token_id_capture_dir": str(tmp_path),
    }


def test_responses_call_captures_tokens_joined_to_eval_record(tmp_path):
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    resp = client.post("/ng-rollout/task0-roll0/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200

    tokens = TokenCaptureStore(tmp_path).read_entries("task0-roll0")
    assert len(tokens) == 1
    assert tokens[0].generation_token_ids == GTOKS and tokens[0].prompt_token_ids == PTOKS

    records = read_model_call_records(CaptureStore(tmp_path), "task0-roll0")
    assert len(records) == 1
    # The training entry joins its eval record by the middleware-minted model_call_id.
    assert tokens[0].model_call_id == records[0].model_call_id


def test_captured_entry_carries_content(tmp_path):
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    client.post("/ng-rollout/task0-rollC/v1/responses", json={"input": "hi"})
    tokens = TokenCaptureStore(tmp_path).read_entries("task0-rollC")
    assert len(tokens) == 1
    # Not token-only: the captured record carries the content-bearing output items.
    assert tokens[0].output_items
    text = tokens[0].output_items[-1]["content"][0]["text"]
    assert text == "hi from responses"


def test_token_arrays_are_stored_once(tmp_path):
    """The served response carries the arrays on an output item; the record does not repeat them.

    Storing them again per item roughly doubles a record, and the per-item copy is not the
    value a trainer reads: an item's prompt in a chained trajectory is the running sequence.
    """
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    client.post("/ng-rollout/task0-rollDedup/v1/responses", json={"input": "hi"})
    entry = TokenCaptureStore(tmp_path).read_entries("task0-rollDedup")[0]
    assert entry.generation_token_ids == GTOKS
    for item in entry.output_items:
        assert not any(field in item for field in TOKEN_FIELDS)
    # Content is kept; only the arrays move off.
    assert entry.output_items[-1]["content"][0]["text"] == "hi from responses"
    # Which item they came off, so a consumer can put the chain-correct values back.
    assert entry.token_item_index == len(entry.output_items) - 1


def test_messages_call_captures_tokens(tmp_path):
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    resp = client.post(
        "/ng-rollout/task0-roll1/v1/messages",
        json={"model": "claude-x", "max_tokens": 16, "messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200
    # The Anthropic response on the wire never carries token ids.
    assert "generation_token_ids" not in resp.text
    tokens = TokenCaptureStore(tmp_path).read_entries("task0-roll1")
    assert len(tokens) == 1 and tokens[0].generation_token_ids == GTOKS


def test_chat_completions_call_captures_tokens(tmp_path):
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    resp = client.post(
        "/ng-rollout/task0-roll2/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 200
    tokens = TokenCaptureStore(tmp_path).read_entries("task0-roll2")
    assert len(tokens) == 1 and tokens[0].generation_token_ids == GTOKS


def test_tokens_captured_even_when_eval_capture_disabled(tmp_path):
    config = {"token_id_capture_enabled": True, "token_id_capture_dir": str(tmp_path)}
    client = TestClient(_server(config).setup_webserver())
    resp = client.post("/ng-rollout/task1-roll0/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200
    assert len(TokenCaptureStore(tmp_path).read_entries("task1-roll0")) == 1
    # No eval capture file was written.
    assert read_model_call_records(CaptureStore(tmp_path), "task1-roll0") == []


def test_uncorrelated_call_captures_nothing(tmp_path):
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    resp = client.post("/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200
    # No rollout prefix -> nothing recorded, no file created.
    assert list(tmp_path.glob("*.tokens.jsonl")) == []


def test_http_route_returns_tokens_and_404_when_disabled(tmp_path):
    # Enabled: the route serves the captured entries as ndjson.
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    client.post("/ng-rollout/task2-roll0/v1/responses", json={"input": "hi"})
    got = client.get("/ng-capture/tokens/task2-roll0")
    assert got.status_code == 200
    lines = [line for line in got.text.splitlines() if line.strip()]
    assert len(lines) == 1
    parsed = TokenEntry.model_validate_json(lines[0])
    assert parsed.generation_token_ids == GTOKS

    # Disabled: the route is not registered.
    disabled = TestClient(_server({}).setup_webserver())
    assert disabled.get("/ng-capture/tokens/task2-roll0").status_code == 404


def test_package_is_dependency_free_leaf():
    """``nemo_gym.token_id_capture`` must import without Gym's server stack.

    A training framework's inference worker imports the record, the protocols,
    and the capture core so it can write into its own data plane (see
    ``protocols.py``). If the package drags in ray/fastapi/uvicorn, that is not
    possible. Run in a subprocess so this test is unaffected by whatever the
    rest of the suite has already imported.
    """
    heavy = ("ray", "fastapi", "uvicorn", "aiohttp", "requests", "torch")
    program = (
        f"import sys; import nemo_gym.token_id_capture; print(','.join(m for m in {heavy!r} if m in sys.modules))"
    )
    proc = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", f"leaf package pulled in: {proc.stdout.strip()}"


def test_streamed_messages_capture_tokens_absent_from_the_stream(tmp_path):
    """The Claude Code shape: streamed /v1/messages.

    Token ids exist only on the assembled response, before it is converted to
    Anthropic and split into SSE. This is the case the whole design turns on, so
    it is asserted end to end rather than only through the non-streamed path.
    """
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    with client.stream(
        "POST",
        "/ng-rollout/stream0-roll0/v1/messages",
        json={
            "model": "claude-x",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    # Nothing on the wire carries token ids.
    assert "generation_token_ids" not in body
    assert "prompt_token_ids" not in body
    # ...yet the record is complete.
    entries = TokenCaptureStore(tmp_path).read_entries("stream0-roll0")
    assert len(entries) == 1
    assert entries[0].generation_token_ids == GTOKS
    assert entries[0].prompt_token_ids == PTOKS
    assert entries[0].output_items, "content must be captured alongside the tokens"


def test_capture_failure_marks_the_rollout_incomplete(tmp_path, monkeypatch):
    """A lost call must not leave the rollout looking complete.

    Capture stays best-effort so a bad payload cannot break the harness's run,
    but delivery has to be able to tell "10 of 10 captured" from "9 of 10".
    """
    store = TokenCaptureStore(tmp_path)

    async def boom(self, entry):
        raise RuntimeError("sink is down")

    monkeypatch.setattr(TokenCaptureStore, "put", boom)
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    # The model call still succeeds.
    resp = client.post("/ng-rollout/fail0-roll0/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200
    assert store.read_entries("fail0-roll0") == []
    assert store.is_incomplete("fail0-roll0")


def test_delete_removes_records_and_marker(tmp_path):
    store = TokenCaptureStore(tmp_path)
    store.append(
        TokenEntry(
            rollout_id="gone-0",
            model_call_id="c",
            prompt_token_ids=PTOKS,
            generation_token_ids=GTOKS,
            generation_log_probs=LPS,
        )
    )
    store.mark_incomplete("gone-0", "c")
    assert store.path_for("gone-0").exists() and store.is_incomplete("gone-0")
    store.delete("gone-0")
    assert not store.path_for("gone-0").exists()
    assert not store.is_incomplete("gone-0")
    # Idempotent: consuming a rollout twice must not raise.
    store.delete("gone-0")


def test_concurrent_appends_to_one_rollout_stay_intact(tmp_path):
    """Writes take an exclusive file lock, which is what keeps two writers from interleaving a
    partial line. Under sharding the writers are separate processes, so the lock has to hold there
    too; this covers the same code path with threads."""
    import concurrent.futures

    store = TokenCaptureStore(tmp_path)
    entries = [
        TokenEntry(
            rollout_id="r0",
            model_call_id=f"call-{i}",
            prompt_token_ids=list(range(200)),
            generation_token_ids=[i] * 64,
            generation_log_probs=[-0.1] * 64,
        )
        for i in range(32)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(store.append, entries))

    read_back = store.read_entries("r0")
    assert len(read_back) == 32
    assert sorted(e.model_call_id for e in read_back) == sorted(e.model_call_id for e in entries)
    # Every line parsed, so no write landed inside another.
    assert all(len(e.generation_token_ids) == 64 for e in read_back)


def test_capture_stamps_cum_len_and_digest(tmp_path):
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    client.post("/ng-rollout/lineage0-roll0/v1/responses", json={"input": "hi"})
    (entry,) = TokenCaptureStore(tmp_path).read_entries("lineage0-roll0")
    assert entry.cum_len == len(PTOKS) + len(GTOKS)
    assert entry.digest == compute_digest(PTOKS + GTOKS)
    # No parent index yet, so the link is absent and the builder infers instead.
    assert entry.parent_call_id is None


def test_digest_round_trip_and_stamp_lineage():
    entry = TokenEntry(
        rollout_id="r",
        model_call_id="c",
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.5],
    )
    stamp_lineage(entry, "parent-1")
    assert cumulative_tokens(entry) == [1, 2, 3]
    assert entry.cum_len == 3
    assert entry.parent_call_id == "parent-1"
    assert entry.digest == compute_digest([1, 2, 3])
    # Distinct sequences must not collide, and the empty sequence is well defined.
    assert compute_digest([1, 2, 3]) != compute_digest([1, 2, 4])
    assert compute_digest([]) == compute_digest([])
    with pytest.raises(ValueError):
        compute_digest([-1])


def test_fingerprint_ignores_non_assistant_turns():
    """Only assistant turns identify lineage: they are what we produced. User and
    tool content varies with the environment and is irrelevant."""
    a = assistant_fingerprint([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    b = assistant_fingerprint([{"role": "user", "content": "DIFFERENT"}, {"role": "assistant", "content": "a"}])
    assert a == b != ""
    # No assistant turn at all is a new conversation, not a match.
    assert assistant_fingerprint([{"role": "user", "content": "q"}]) == ""


def test_fingerprint_survives_tool_argument_reserialization():
    """Harnesses re-serialize tool-call arguments between turns -- compact one
    turn, pretty-printed the next. Without canonicalization the same call would
    not compare equal to itself and every tool-using turn would miss."""
    compact = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": '{"b":1,"a":2}'}}]}
    ]
    pretty = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "f", "arguments": '{\n  "a": 2,\n  "b": 1\n}'}}],
        }
    ]
    assert assistant_fingerprint(compact) == assistant_fingerprint(pretty)


def test_lineage_resolves_the_parent_across_a_turn():
    lineage = RolloutLineage()
    first_request = [{"role": "user", "content": "hello"}]
    lineage.record("call-1", first_request + [{"role": "assistant", "content": "hi"}], [1, 2, 3], "d1")

    # The next request echoes the assistant turn, as any harness must to continue.
    second_request = first_request + [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "more"}]
    parent = lineage.resolve(second_request)
    assert parent is not None and parent.call_id == "call-1"
    assert parent.cum_tokens == [1, 2, 3] and parent.cum_len == 3


def test_lineage_misses_on_a_rewritten_history():
    """A compacted or rewritten context is a new root, not a wrong parent."""
    lineage = RolloutLineage()
    lineage.record("call-1", [{"role": "assistant", "content": "hi"}], [1, 2, 3], "d1")
    assert lineage.resolve([{"role": "assistant", "content": "a summary of the above"}]) is None


def test_lineage_refuses_an_ambiguous_parent():
    """Two recorded calls with byte-identical output cannot be told apart. Guessing
    would attribute tokens to the wrong parent, so a unique match is required."""
    lineage = RolloutLineage()
    messages = [{"role": "assistant", "content": "same"}]
    lineage.record("call-a", messages, [1, 2], "da")
    lineage.record("call-b", messages, [3, 4], "db")
    assert lineage.resolve(messages) is None


def test_lineage_is_a_tree_so_forks_get_the_parent_not_the_previous_call():
    """Two sub-agents branching from one parent must BOTH resolve to that parent.

    A running cursor ("the last call") would hand the second branch a prefix
    containing the first branch's generation, and the splice applies a supplied
    prefix unconditionally -- so that would be silently wrong, not just wasteful.
    """
    lineage = RolloutLineage()
    shared = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "plan"}]
    lineage.record("parent", shared, [1, 2, 3], "dp")
    lineage.record(
        "branch-a",
        shared + [{"role": "user", "content": "a"}, {"role": "assistant", "content": "A"}],
        [1, 2, 3, 4],
        "da",
    )

    # The second branch continues the PARENT, not branch-a.
    second = shared + [{"role": "user", "content": "b"}]
    parent = lineage.resolve(second)
    assert parent is not None and parent.call_id == "parent"
    assert parent.cum_tokens == [1, 2, 3]


def test_lineage_index_is_bounded():
    """An abandoned rollout is never read again, so eviction cannot wait for
    consumption. Losing an entry costs a fallback, never a wrong answer."""
    index = LineageIndex(max_rollouts=3)
    for i in range(10):
        index.for_rollout(f"r{i}")
    assert len(index) == 3


def test_served_calls_link_to_their_parent(tmp_path):
    """End to end: a second call whose request echoes the first call's assistant
    turn is recorded with parent_call_id pointing at it."""
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    first = [{"role": "user", "content": "hello"}]
    client.post("/ng-rollout/lin0-roll0/v1/chat/completions", json={"messages": first})
    entries = TokenCaptureStore(tmp_path).read_entries("lin0-roll0")
    assert len(entries) == 1 and entries[0].parent_call_id is None

    content = entries[0].output_items[0]["content"]
    served_text = content if isinstance(content, str) else content[0]["text"]
    second = first + [{"role": "assistant", "content": served_text}, {"role": "user", "content": "more"}]
    client.post("/ng-rollout/lin0-roll0/v1/chat/completions", json={"messages": second})

    entries = TokenCaptureStore(tmp_path).read_entries("lin0-roll0")
    assert len(entries) == 2
    assert entries[1].parent_call_id == entries[0].model_call_id


def test_fingerprint_matches_across_openai_and_anthropic_tool_shapes():
    """A tool-using turn must match itself across dialects.

    We record the turn we produced in OpenAI shape (``tool_calls``), but Claude
    Code echoes it back in Anthropic shape (``content`` blocks of type
    ``tool_use``). If those hash differently, the parent is never resolved for
    exactly the turns that create multi-turn rollouts -- which is what happened
    in the first live multi-turn run: every record came back with
    ``parent_call_id: None`` even though the calls chained perfectly.
    """
    recorded = [
        {
            "role": "assistant",
            "content": "Let me compute that.",
            "tool_calls": [{"function": {"name": "Bash", "arguments": '{"command":"echo 6"}'}}],
        }
    ]
    echoed = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me compute that."},
                {"type": "tool_use", "name": "Bash", "input": {"command": "echo 6"}},
            ],
        }
    ]
    assert assistant_fingerprint(recorded) == assistant_fingerprint(echoed) != ""


def test_fingerprint_agrees_across_all_three_dialects():
    """The same turn must hash identically however the harness represents it.

    Chat puts tool calls on the message, Anthropic nests them in content blocks, and Responses
    emits a standalone function_call item with no role. A role-only check misses the Responses
    case entirely, which would silently disable parent resolution and prefix supply for every
    harness that speaks it.
    """

    anthropic = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}}]},
    ]
    chat = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "Bash", "arguments": '{"cmd":"ls"}'}}],
        },
    ]
    responses = [
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "function_call", "name": "Bash", "arguments": '{"cmd":"ls"}', "call_id": "c1"},
    ]

    assert assistant_fingerprint(anthropic) == assistant_fingerprint(chat) == assistant_fingerprint(responses)
    assert assistant_fingerprint(responses) != ""


def test_responses_tool_calls_are_distinguished():
    """Two Responses turns differing only in tool arguments must not collide, or the index
    treats them as the same call and refuses to resolve either."""

    def turn(cmd):
        return [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "checking"}]},
            {"type": "function_call", "name": "Bash", "arguments": '{"cmd":"%s"}' % cmd, "call_id": "c1"},
        ]

    assert assistant_fingerprint(turn("ls")) != assistant_fingerprint(turn("rm -rf /"))


def test_lineage_resolves_a_tool_using_turn_echoed_in_anthropic_shape():
    lineage = RolloutLineage()
    produced = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "Bash", "arguments": '{"command":"factor 420"}'}}],
    }
    lineage.record("call-1", [{"role": "user", "content": "factor 420"}, produced], [1, 2, 3], "d1")

    # The harness continues the conversation, echoing the turn as Anthropic blocks
    # and appending the tool result.
    next_request = [
        {"role": "user", "content": "factor 420"},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "factor 420"}}]},
        {"role": "user", "content": [{"type": "tool_result", "content": "420: 2 2 3 5 7"}]},
    ]
    parent = lineage.resolve(next_request)
    assert parent is not None and parent.call_id == "call-1"
    assert parent.cum_tokens == [1, 2, 3]


def test_read_route_requires_a_token_when_one_is_configured(tmp_path):
    """The route serves raw training tokens on the same app the harness calls to
    generate. Inside a trusted cluster that is fine; once the harness runs in a
    sandbox whose only egress is this server it could read its own training data,
    or another rollout's."""
    config = dict(_both_enabled(tmp_path))
    config["token_id_capture_read_token"] = "s3cret"  # pragma: allowlist secret
    client = TestClient(_server(config).setup_webserver())
    client.post("/ng-rollout/auth0-roll0/v1/responses", json={"input": "hi"})

    assert client.get("/ng-capture/tokens/auth0-roll0").status_code == 401
    assert client.get("/ng-capture/tokens/auth0-roll0", headers={"Authorization": "Bearer wrong"}).status_code == 401

    ok = client.get("/ng-capture/tokens/auth0-roll0", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert TokenEntry.model_validate_json(ok.text.splitlines()[0]).generation_token_ids == GTOKS


def test_read_route_stays_open_when_no_token_is_configured(tmp_path):
    """Existing deployments keep working; the gap is logged rather than enforced."""
    client = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())
    client.post("/ng-rollout/auth1-roll0/v1/responses", json={"input": "hi"})
    assert client.get("/ng-capture/tokens/auth1-roll0").status_code == 200
