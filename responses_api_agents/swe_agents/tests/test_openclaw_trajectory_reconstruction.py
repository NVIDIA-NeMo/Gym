# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from responses_api_agents.swe_agents.openclaw.trajectory_reconstruction import (
    _dedupe_superseded_turns,
    classify_openclaw_agent_error,
    reconstruct_responses_items,
)


def _assistant_item(text, *, tokens=None):
    item = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
    if tokens is not None:
        item["prompt_token_ids"] = tokens["p"]
        item["generation_token_ids"] = tokens["g"]
    return item


def _call_item(name, call_id, *, tokens=None):
    item = {"type": "function_call", "name": name, "call_id": call_id, "arguments": "{}"}
    if tokens is not None:
        item["prompt_token_ids"] = tokens["p"]
        item["generation_token_ids"] = tokens["g"]
    return item


def _output_item(call_id, output_text):
    return {"type": "function_call_output", "call_id": call_id, "output": output_text}


def test_single_turn():
    log = [
        {
            "turn": 0,
            "endpoint": "/v1/responses",
            "request": {
                "input": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "go"},
                ],
                "tools": [{"type": "function", "name": "read"}],
            },
            "response": {"output": [_assistant_item("done", tokens={"p": [1, 2], "g": [3]})]},
            "upstream_status": 200,
        }
    ]
    inputs, outputs, tools = reconstruct_responses_items(log)
    assert inputs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
    ]
    assert len(outputs) == 1
    assert outputs[-1]["prompt_token_ids"] == [1, 2]
    assert tools == [{"type": "function", "name": "read"}]


def test_multi_turn_with_tool_calls_and_outputs():
    log = [
        {
            "turn": 0,
            "endpoint": "/v1/responses",
            "request": {"input": [{"role": "user", "content": "fix"}], "tools": []},
            "response": {"output": [_call_item("read", "c1", tokens={"p": [1], "g": [2]})]},
            "upstream_status": 200,
        },
        {
            "turn": 1,
            "endpoint": "/v1/responses",
            "request": {
                "input": [
                    {"role": "user", "content": "fix"},
                    _call_item("read", "c1"),
                    _output_item("c1", "<file contents>"),
                ],
                "tools": [],
            },
            "response": {"output": [_assistant_item("done", tokens={"p": [1, 2, 3], "g": [4]})]},
            "upstream_status": 200,
        },
    ]
    inputs, outputs, _tools = reconstruct_responses_items(log)
    assert [o["type"] for o in outputs] == ["function_call", "function_call_output", "message"]
    assert outputs[0].get("prompt_token_ids") == [1]
    assert outputs[2].get("prompt_token_ids") == [1, 2, 3]


def test_empty_log_returns_empty_tuples():
    assert reconstruct_responses_items([]) == ([], [], [])


def test_string_response_turn_is_skipped_not_crashed():
    log = [
        {
            "turn": 0,
            "endpoint": "/v1/responses",
            "request": {"input": [{"role": "user", "content": "go"}], "tools": []},
            "response": {"output": [_assistant_item("ok", tokens={"p": [1], "g": [2]})]},
            "upstream_status": 200,
        },
        {
            "turn": 1,
            "endpoint": "/v1/responses",
            "request": {"input": [{"role": "user", "content": "go"}], "tools": []},
            "response": "Internal Server Error",  # non-dict body from up.json()
            "upstream_status": 500,
        },
    ]
    inputs, outputs, _tools = reconstruct_responses_items(log)
    assert inputs == [{"role": "user", "content": "go"}]
    assert len(outputs) == 1  # only turn 0's output; the string-response turn is skipped


def test_string_request_and_malformed_input_item_are_tolerated():
    """Defensive: a non-dict `request`, or a non-dict item inside `input`, must not crash."""
    log = [
        {
            "turn": 0,
            "endpoint": "/v1/responses",
            "request": {"input": [{"role": "user", "content": "go"}], "tools": []},
            "response": {"output": [_assistant_item("ok", tokens={"p": [1], "g": [2]})]},
            "upstream_status": 200,
        },
        {
            "turn": 1,
            "endpoint": "/v1/responses",
            "request": {
                "input": [{"role": "user", "content": "go"}, "stray-string-item"],
                "tools": [],
            },
            "response": {"output": []},
            "upstream_status": 200,
        },
        {"turn": 2, "endpoint": "/v1/responses", "request": "bad", "response": "bad"},
    ]
    inputs, outputs, _tools = reconstruct_responses_items(log)
    assert inputs == [{"role": "user", "content": "go"}]
    assert len(outputs) == 1  # turn 0 only; stray string item skipped, turn 2 skipped


def test_idle_timeout_retry_dedupes_superseded_turn():
    log = [
        {  # turn 0 — ABANDONED (idle-timeout); same [user] prompt p=[1,2]
            "turn": 0,
            "endpoint": "/v1/responses",
            "request": {"input": [{"role": "user", "content": "fix"}], "tools": []},
            "response": {"output": [_call_item("exec", "abandoned", tokens={"p": [1, 2], "g": [3]})]},
            "upstream_status": 200,
        },
        {  # turn 0 — RETRY (used); identical prompt p=[1,2], different gen
            "turn": 0,
            "endpoint": "/v1/responses",
            "request": {"input": [{"role": "user", "content": "fix"}], "tools": []},
            "response": {"output": [_call_item("exec", "used", tokens={"p": [1, 2], "g": [4, 5]})]},
            "upstream_status": 200,
        },
        {  # turn 2 — history advances from the RETRY's exec call (used)
            "turn": 2,
            "endpoint": "/v1/responses",
            "request": {
                "input": [
                    {"role": "user", "content": "fix"},
                    _call_item("exec", "used"),
                    _output_item("used", "<exec output>"),
                ],
                "tools": [],
            },
            "response": {"output": [_assistant_item("done", tokens={"p": [1, 2, 4, 5, 6], "g": [7]})]},
            "upstream_status": 200,
        },
    ]
    inputs, outputs, _tools = reconstruct_responses_items(log)
    call_ids = [o.get("call_id") for o in outputs if o.get("type") == "function_call"]
    assert call_ids == ["used"]  # abandoned turn-0 generation dropped
    assert all(o.get("call_id") != "abandoned" for o in outputs)
    # The kept turn-0 generation token IDs are the retry's, and turn-2's prompt extends them.
    first_call = next(o for o in outputs if o.get("type") == "function_call")
    assert first_call["generation_token_ids"] == [4, 5]
    last_msg = outputs[-1]
    assert last_msg["prompt_token_ids"] == [1, 2, 4, 5, 6]


def test_dedupe_keeps_retry_after_errored_same_turn():
    """An errored generation (non-2xx, _turns_used not advanced) followed by a same-turn
    retry: keep the retry, drop the error entry — even though the error entry would have
    been skipped downstream anyway, dedup makes the intent explicit and order-independent."""
    log = [
        {
            "turn": 0,
            "endpoint": "/v1/responses",
            "request": {"input": [{"role": "user", "content": "go"}], "tools": []},
            "response": "Internal Server Error",  # errored, non-dict body
            "upstream_status": 500,
        },
        {
            "turn": 0,
            "endpoint": "/v1/responses",
            "request": {"input": [{"role": "user", "content": "go"}], "tools": []},
            "response": {"output": [_assistant_item("ok", tokens={"p": [1], "g": [2]})]},
            "upstream_status": 200,
        },
    ]
    inputs, outputs, _tools = reconstruct_responses_items(log)
    assert inputs == [{"role": "user", "content": "go"}]
    assert len(outputs) == 1
    assert outputs[0].get("content")[0]["text"] == "ok"


def test_dedupe_superseded_turns_drops_abandoned_same_turn_entry():
    """Direct unit test of `_dedupe_superseded_turns`: two entries share turn 0 (the first
    abandoned, the second the retry); the function itself must drop the earlier (superseded)
    entry and keep the last-logged one. Fails if the dedupe logic were a no-op."""
    abandoned = {"turn": 0, "tag": "abandoned", "request": {"input": []}}
    retry = {"turn": 0, "tag": "retry", "request": {"input": []}}
    later = {"turn": 1, "tag": "later", "request": {"input": []}}
    deduped = _dedupe_superseded_turns([abandoned, retry, later])
    assert deduped == [retry, later]
    assert [e["tag"] for e in deduped] == ["retry", "later"]


def test_classify_context_window():
    log = [
        {
            "turn": 0,
            "endpoint": "/v1/responses",
            "request": {"input": [], "tools": []},
            "response": {"error": {"message": "maximum context length"}},
            "upstream_status": 400,
            "error": None,
        }
    ]
    assert (
        classify_openclaw_agent_error(
            proxy_log=log, trajectory_events=[], subprocess_timed_out=False, subprocess_exit_code=1
        )
        == "context_window"
    )


def test_classify_other_for_unknown_upstream_4xx():
    log = [
        {
            "turn": 0,
            "endpoint": "/v1/responses",
            "request": {"input": [], "tools": []},
            "response": {"error": {"message": "some other failure"}},
            "upstream_status": 400,
            "error": None,
        }
    ]
    assert (
        classify_openclaw_agent_error(
            proxy_log=log, trajectory_events=[], subprocess_timed_out=False, subprocess_exit_code=1
        )
        == "other"
    )


def test_classify_max_iteration_from_proxy_error():
    """A refused turn carries the literal proxy error string 'max_iteration' (request/response
    null); classification must bucket it as max_iteration."""
    log = [
        {
            "turn": 0,
            "endpoint": "/v1/responses",
            "request": {"input": [], "tools": []},
            "response": {"output": []},
            "upstream_status": 200,
            "error": None,
        },
        {
            "turn": 1,
            "endpoint": "/v1/responses",
            "request": None,
            "response": None,
            "upstream_status": None,
            "error": "max_iteration",
        },
    ]
    assert (
        classify_openclaw_agent_error(
            proxy_log=log, trajectory_events=[], subprocess_timed_out=False, subprocess_exit_code=0
        )
        == "max_iteration"
    )


def test_classify_clean_run_returns_none():
    log = [
        {
            "turn": 0,
            "endpoint": "/v1/responses",
            "request": {"input": [], "tools": []},
            "response": {"output": []},
            "upstream_status": 200,
            "error": None,
        }
    ]
    assert (
        classify_openclaw_agent_error(
            proxy_log=log, trajectory_events=[], subprocess_timed_out=False, subprocess_exit_code=0
        )
        is None
    )


def test_classify_other_on_nonzero_exit_with_no_log_signal():
    assert (
        classify_openclaw_agent_error(
            proxy_log=[], trajectory_events=[], subprocess_timed_out=False, subprocess_exit_code=42
        )
        == "other"
    )


def test_classify_session_ended_fallback_max_iteration():
    """Clean-exit (no 4xx in proxy log) failures fall back to OpenClaw's session.ended event:
    a reason mentioning max+iter buckets as max_iteration."""
    events = [{"type": "session.ended", "data": {"reason": "stopped: max iterations reached"}}]
    assert (
        classify_openclaw_agent_error(
            proxy_log=[], trajectory_events=events, subprocess_timed_out=False, subprocess_exit_code=1
        )
        == "max_iteration"
    )


def test_classify_session_ended_fallback_context_window():
    events = [{"type": "session.ended", "data": {"reason": "context exceeded"}}]
    assert (
        classify_openclaw_agent_error(
            proxy_log=[], trajectory_events=events, subprocess_timed_out=False, subprocess_exit_code=1
        )
        == "context_window"
    )


def test_classify_subprocess_timed_out_returns_none():
    """A wall-clock timeout is surfaced separately (agent_timed_out); the error-kind is None
    regardless of log/exit so the two signals don't double-count."""
    assert (
        classify_openclaw_agent_error(
            proxy_log=[{"error": "max_iteration"}],
            trajectory_events=[],
            subprocess_timed_out=True,
            subprocess_exit_code=1,
        )
        is None
    )
