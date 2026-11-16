# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion-log conversion and the compaction condenser (offline: completion-log fixtures and stubbed OpenHands)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.config import ModelServerRef
from responses_api_agents.openhands_codeact_agent import compaction_condenser, conversion


SYSTEM = {"role": "system", "content": [{"type": "text", "text": "You are OpenHands."}]}
TASK = {"role": "user", "content": [{"type": "text", "text": "Write summary.txt from the brief."}]}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "run",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
]


def _tool_call(n):
    return {
        "id": f"call_{n}",
        "type": "function",
        "function": {"name": "execute_bash", "arguments": json.dumps({"command": f"echo {n}"})},
    }


def _assistant_request_message(n, *, stale_tokens=True):
    """Assistant message as OpenHands re-sends it: list content + the fork's token fields."""
    message = {"role": "assistant", "content": [], "tool_calls": [_tool_call(n)]}
    if stale_tokens:
        message.update(prompt_token_ids=[9, 9, 9], generation_token_ids=[8, 8], generation_log_probs=[-9.0, -9.0])
    return message


def _tool_message(n):
    return {
        "role": "tool",
        "content": [{"type": "text", "text": f"OBSERVATION {n}"}],
        "tool_call_id": f"call_{n}",
        "name": "execute_bash",
    }


def _reply(n, *, tool_call=True, content="", echo=True, topk=False):
    message = {"role": "assistant", "content": content or None}
    if tool_call:
        message["tool_calls"] = [_tool_call(n)]
    if echo:
        message.update(
            prompt_token_ids=list(range(1, 10 + n)),
            generation_token_ids=[1000 + n] * 3,
            generation_log_probs=[-0.1] * 3,
        )
        if topk:
            message["generation_topk_ids"] = [[1000 + n, 7]] * 3
            message["generation_topk_logprobs"] = [[-0.1, -2.0]] * 3
    return message


def _payload(n, messages, reply, *, tools=TOOLS, summary=False, marker=True, finish_reason=None):
    psf = {k: reply[k] for k in ("prompt_token_ids", "generation_token_ids", "generation_log_probs") if k in reply}
    out = {
        "messages": messages,
        "response": {
            "id": f"chatcmpl-{n}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": reply,
                    "finish_reason": finish_reason or ("tool_calls" if reply.get("tool_calls") else "stop"),
                }
            ],
            "usage": {"prompt_tokens": 100 + 50 * n, "completion_tokens": 3, "total_tokens": 103 + 50 * n},
        },
        "provider_specific_fields": psf,
        "kwargs": {"model": "policy", "tools": tools} if tools else {"model": "policy"},
        "timestamp": 1_700_000_000 + n,
    }
    if summary and marker:
        out[conversion.CALL_KIND_KEY] = conversion.SUMMARY_CALL_KIND
    return out


def _write(task: Path, payloads):
    """Completion logs as the agent server collects them: ``<trace>/completions``."""
    folder = task / "completions"
    folder.mkdir(parents=True, exist_ok=True)
    for n, payload in enumerate(payloads):
        (folder / f"policy-{payload['timestamp']}.{n}.json").write_text(json.dumps(payload))
    return task


def _summary_request(previous="", events=("EVENT 0", "EVENT 1")):
    prompt = compaction_condenser.build_summary_prompt(previous, list(enumerate(events)), max_event_chars=10_000)
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def compacting_rollout(*, echo=True, marker=True, topk=False):
    """Two agent calls, the summary call, two post-summary calls (the last one finishes)."""
    summary_text = "USER_CONTEXT: write summary.txt\nCOMPLETED: brief read"
    summary_msg = {"role": "user", "content": [{"type": "text", "text": summary_text}]}
    return [
        _payload(0, [SYSTEM, TASK], _reply(0, echo=echo, topk=topk)),
        _payload(1, [SYSTEM, TASK, _assistant_request_message(0), _tool_message(0)], _reply(1, echo=echo, topk=topk)),
        _payload(
            2,
            _summary_request(),
            _reply(2, tool_call=False, content=summary_text, echo=echo, topk=topk),
            tools=None,
            summary=True,
            marker=marker,
        ),
        _payload(3, [SYSTEM, TASK, summary_msg], _reply(3, echo=echo, topk=topk)),
        _payload(
            4,
            [SYSTEM, TASK, summary_msg, _assistant_request_message(3), _tool_message(3)],
            _reply(4, tool_call=False, content="Done, summary.txt written.", echo=echo, topk=topk),
        ),
    ]


MODEL = ModelServerRef(type="responses_api_models", name="policy_model")


# ---------------------------------------------------------------------------
# Completion logs -> response items, usage and compaction observations
# ---------------------------------------------------------------------------


def test_calls_are_read_in_dispatch_order_with_the_summary_marked(tmp_path):
    payloads = compacting_rollout()
    calls = conversion.load_calls(_write(tmp_path, list(reversed(payloads))) / "completions")
    assert [call["response_id"] for call in calls] == [f"chatcmpl-{n}" for n in range(5)]
    assert [call["kind"] for call in calls] == ["agent", "agent", "compaction_summary", "agent", "agent"]
    assert conversion.load_calls(tmp_path / "missing") == []


def test_summary_is_recognised_by_its_prompt_without_the_marker(tmp_path):
    calls = conversion.load_calls(_write(tmp_path, compacting_rollout(marker=False)) / "completions")
    assert calls[2]["kind"] == conversion.SUMMARY_CALL_KIND


def test_output_items_are_the_last_agent_context_without_tokens(tmp_path):
    calls = conversion.load_calls(_write(tmp_path, compacting_rollout()) / "completions")
    items, tools = conversion.output_items(calls)
    dumped = [item.model_dump() for item in items]
    assert [item["type"] for item in dumped] == ["function_call", "function_call_output", "message"]
    assert dumped[-1]["content"][0]["text"] == "Done, summary.txt written."
    # Training tokens come from Gym's capture; the sandbox log's token fields never reach the response.
    assert not any("generation_token_ids" in json.dumps(item) for item in dumped)
    assert [tool.name for tool in tools] == ["execute_bash"]
    assert conversion.output_items([]) == ([], [])


def test_each_summary_call_becomes_a_compaction_observation(tmp_path):
    calls = conversion.load_calls(_write(tmp_path, compacting_rollout()) / "completions")
    notes = [{"response_id": "chatcmpl-2", "trigger_usage": {"prompt_tokens": 150, "completion_tokens": 3}}]
    (record,) = conversion.compaction_records(calls, notes, invocation_id="rollout-1", model_ref=MODEL)
    assert record.invocation_id == "rollout-1" and record.outcome == "completed"
    assert record.before_model_call.response_id == "chatcmpl-1"
    assert [ref.response_id for ref in record.model_calls] == ["chatcmpl-2"]
    assert record.after_model_call.response_id == "chatcmpl-3"
    assert record.tokens_before == 153 and record.summary.startswith("USER_CONTEXT")


def test_usage_is_reported_only_when_every_answered_call_has_it():
    requests = [
        {"http_status": 200, "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        {"http_status": 503},
        {"http_status": 200, "usage": {"prompt_tokens": 20, "completion_tokens": 3}},
    ]
    usage = conversion.response_usage(requests)
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (30, 5, 35)
    requests.append({"http_status": 200})
    assert conversion.response_usage(requests) is None
    assert conversion.response_usage([]) is None


@pytest.mark.parametrize(
    "result,expected",
    [
        ({"finished": True}, ("completed", "completed")),
        ({"finished": False, "last_error": "context length limit has been exceeded"}, ("incomplete", "context_limit")),
        ({"finished": False, "last_error": "Agent reached maximum iteration 200"}, ("incomplete", "max_iterations")),
        (
            {"finished": False, "last_error": "AgentStuckInLoopError: Agent got stuck in a loop"},
            ("incomplete", "stuck_in_loop"),
        ),
        ({"finished": False, "last_error": "TypeError: bad tool argument"}, ("failed", "error")),
        (None, ("failed", "runner_error")),
    ],
)
def test_run_outcome_follows_the_controllers_last_error(result, expected):
    outcome = conversion.run_outcome(result, timed_out=False, rollout_stop=False)
    assert (outcome.status, outcome.stop_reason) == expected


def test_deadline_and_rollout_stop_end_the_run_incomplete():
    assert conversion.run_outcome(None, timed_out=True, rollout_stop=False).stop_reason == "wall_time"
    assert conversion.run_outcome({"finished": True}, timed_out=False, rollout_stop=True).stop_reason == "rollout_stop"


def test_failed_run_is_classified_from_the_last_model_request():
    assert conversion.classify_error({}, [{"http_status": 200}, {"http_status": 404}]) == ("model_request_error", 404)
    assert conversion.classify_error({}, [{"http_status": 404}, {"http_status": 200}]) == ("unknown", None)
    assert conversion.classify_error({}, [{"error_type": "ClientConnectorError"}]) == ("model_transport_error", None)
    assert conversion.classify_error({"last_error": "APITimeoutError: x"}, []) == ("model_transport_error", None)
    assert conversion.classify_error({"last_error": "TypeError: x"}, []) == ("unknown", None)


def test_partial_jsonl_lines_are_skipped(tmp_path):
    log = tmp_path / "model-requests.jsonl"
    log.write_text('{"http_status": 200}\n{"http_sta')
    assert conversion.read_jsonl(log) == [{"http_status": 200}]
    assert conversion.read_jsonl(tmp_path / "absent.jsonl") == []


# ---------------------------------------------------------------------------
# Sandbox-side condenser helpers (pure; OpenHands itself is not importable here)
# ---------------------------------------------------------------------------


class MessageAction(SimpleNamespace):
    pass


class SystemMessageAction(SimpleNamespace):
    pass


class CmdRunAction(SimpleNamespace):
    pass


def test_head_keeps_system_prompt_and_task():
    events = [
        SystemMessageAction(id=0, source="agent"),
        MessageAction(id=1, source="user"),
        CmdRunAction(id=2, source="agent"),
    ]
    assert compaction_condenser.head_length(events) == 2
    assert compaction_condenser.head_length([MessageAction(id=0, source="user")]) == 1
    assert compaction_condenser.head_length([CmdRunAction(id=0, source="agent")]) == 1
    # str(EventSource.USER) on a str-mixin Enum
    assert compaction_condenser.head_length([MessageAction(id=0, source="EventSource.USER")]) == 1


def test_compaction_trigger_matches_stirrup_and_never_refires_on_the_same_reply():
    usage = SimpleNamespace(prompt_tokens=350, completion_tokens=70)  # 420 / 600 = 0.7
    kw = dict(max_context=600, cutoff=0.7)
    assert compaction_condenser.should_compact(last_usage=usage, usage_count=3, compacted_at=0, **kw)
    assert not compaction_condenser.should_compact(
        last_usage=SimpleNamespace(prompt_tokens=350, completion_tokens=69), usage_count=3, compacted_at=0, **kw
    )
    # Same reply after the compaction: wait for a reply made inside the summarized context.
    assert not compaction_condenser.should_compact(last_usage=usage, usage_count=3, compacted_at=3, **kw)
    assert compaction_condenser.should_compact(last_usage=usage, usage_count=4, compacted_at=3, **kw)
    assert not compaction_condenser.should_compact(last_usage=None, usage_count=0, compacted_at=0, **kw)


def test_summary_prompt_is_the_openhands_prompt_with_truncated_events():
    prompt = compaction_condenser.build_summary_prompt(
        "old summary", [(5, "x" * 50), (6, "short")], max_event_chars=20
    )
    assert prompt.startswith(conversion.SUMMARY_PROMPT_PREFIX)
    assert "<PREVIOUS SUMMARY>\nold summary\n</PREVIOUS SUMMARY>" in prompt
    assert (
        "<EVENT id=5>\n" + "x" * 10 + "\n[... Observation truncated due to length ...]\n" + "x" * 10 + "\n</EVENT>"
        in prompt
    )
    assert "<EVENT id=6>\nshort\n</EVENT>\nNow summarize the events using the rules above." in prompt
    # The agent server's fallback detection keys on this prefix.
    assert conversion.call_kind({"messages": [{"role": "user", "content": prompt}]}) == conversion.SUMMARY_CALL_KIND


# ---------------------------------------------------------------------------
# Condenser wiring against stubbed OpenHands modules
# ---------------------------------------------------------------------------


def _stub_openhands(monkeypatch, tmp_path, *, reply_content="SUMMARY TEXT"):
    """The handful of OpenHands symbols ``compaction_condenser.install`` imports, as stubs."""
    import sys
    import types

    posted = []

    class NemoGymClient:
        def __init__(self, llm):
            self.llm = llm

        async def model_call(self, messages, tools=None):
            params = {"messages": [m.model_dump() for m in messages], **self.llm._nemo_gym_llm_kwargs}
            response_json = {
                "id": "chatcmpl-summary",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": reply_content,
                            "prompt_token_ids": [1, 2],
                            "generation_token_ids": [3],
                            "generation_log_probs": [-0.5],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
            self.llm.metrics.add_token_usage(2, 1)
            posted.append((params, tools))
            self._log_completion(
                messages,
                response_json,
                {"prompt_token_ids": [1, 2], "generation_token_ids": [3], "generation_log_probs": [-0.5]},
                params,
            )
            return SimpleNamespace(
                id="chatcmpl-summary",
                choices=[SimpleNamespace(message=SimpleNamespace(content=reply_content))],
                model_dump=lambda: response_json,
            )

    class TextContent(SimpleNamespace):
        pass

    class Message:
        def __init__(self, role, content):
            self.role, self.content = role, content

        def model_dump(self):
            return {"role": self.role, "content": [{"type": "text", "text": c.text} for c in self.content]}

    class CondensationAction(SimpleNamespace):
        pass

    class AgentCondensationObservation(SimpleNamespace):
        pass

    class Metrics:
        def __init__(self, model_name="default"):
            self.token_usages = []

        def add_token_usage(self, prompt_tokens, completion_tokens, **_):
            self.token_usages.append(
                SimpleNamespace(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model_dump=lambda p=prompt_tokens, c=completion_tokens: {
                        "prompt_tokens": p,
                        "completion_tokens": c,
                    },
                )
            )

        def get(self):
            return {"calls": len(self.token_usages)}

    class View(SimpleNamespace):
        def __len__(self):
            return len(self.events)

    class Condensation(SimpleNamespace):
        pass

    class RollingCondenser:
        def __init__(self):
            self.metadata = {}

        def add_metadata(self, key, value):
            self.metadata[key] = value

    class LLMSummarizingCondenser(RollingCondenser):
        @classmethod
        def from_config(cls, config, llm_registry):
            raise AssertionError("not patched")

    modules = {
        "openhands": {},
        "openhands.agenthub": {},
        "openhands.core": {},
        "openhands.events": {},
        "openhands.events.action": {},
        "openhands.events.observation": {},
        "openhands.llm": {},
        "openhands.memory": {},
        "openhands.memory.condenser": {},
        "openhands.memory.condenser.impl": {},
        "openhands.agenthub.nemo_gym_client": {"NemoGymClient": NemoGymClient},
        "openhands.core.message": {"Message": Message, "TextContent": TextContent},
        "openhands.events.action.agent": {"CondensationAction": CondensationAction},
        "openhands.events.observation.agent": {"AgentCondensationObservation": AgentCondensationObservation},
        "openhands.llm.metrics": {"Metrics": Metrics},
        "openhands.memory.condenser.condenser": {
            "Condensation": Condensation,
            "RollingCondenser": RollingCondenser,
            "View": View,
        },
        "openhands.memory.condenser.impl.llm_summarizing_condenser": {
            "LLMSummarizingCondenser": LLMSummarizingCondenser
        },
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    folder = tmp_path / "completions"
    folder.mkdir()
    agent_llm = SimpleNamespace(
        config=SimpleNamespace(model="policy", base_url="http://policy/v1", log_completions_folder=str(folder)),
        metrics=Metrics(),
        _nemo_gym_llm_kwargs={"model": "policy", "seed": None},
    )
    registry = SimpleNamespace(service_to_llm={"agent": agent_llm})
    return SimpleNamespace(
        posted=posted,
        agent_llm=agent_llm,
        registry=registry,
        folder=folder,
        View=View,
        AgentCondensationObservation=AgentCondensationObservation,
        LLMSummarizingCondenser=LLMSummarizingCondenser,
    )


def test_condenser_summarizes_through_the_direct_client_and_marks_the_log(monkeypatch, tmp_path):
    stub = _stub_openhands(monkeypatch, tmp_path)
    compaction_condenser.install(
        cutoff=0.7, max_context=1000, max_event_chars=10_000, log_path=str(tmp_path / "compactions.jsonl")
    )
    condenser = stub.LLMSummarizingCondenser.from_config(None, stub.registry)

    events = [SystemMessageAction(id=0, source="agent"), MessageAction(id=1, source="user")]
    for n in range(2, 8):
        events.append(CmdRunAction(id=n, source="agent", __str__=None))
    view = stub.View(events=events, unhandled_condensation_request=False)
    # No agent call yet: nothing to trigger on.
    assert condenser.should_condense(view) is False
    stub.agent_llm.metrics.add_token_usage(500, 100)  # 0.6 < 0.7
    assert condenser.should_condense(view) is False
    stub.agent_llm.metrics.add_token_usage(650, 100)  # 0.75
    assert condenser.should_condense(view) is True

    condensation = condenser.get_condensation(view)
    action = condensation.action
    assert (action.forgotten_events_start_id, action.forgotten_events_end_id, action.summary_offset) == (2, 7, 2)
    assert action.summary == "SUMMARY TEXT"
    assert condenser.compactions == 1 and condenser.metadata["response"]["id"] == "chatcmpl-summary"
    # The summary call is one user message with the upstream prompt and no tools.
    ((params, tools),) = stub.posted
    assert tools is None and params["model"] == "policy"
    assert len(params["messages"]) == 1 and params["messages"][0]["role"] == "user"
    assert params["messages"][0]["content"][0]["text"].startswith(conversion.SUMMARY_PROMPT_PREFIX)
    assert "<EVENT id=2>" in params["messages"][0]["content"][0]["text"]
    # Its completion log carries the marker the agent server reports the compaction from.
    logs = list(stub.folder.glob("*.json"))
    assert len(logs) == 1
    payload = json.loads(logs[0].read_text())
    assert payload[conversion.CALL_KIND_KEY] == conversion.SUMMARY_CALL_KIND
    assert conversion.call_kind(payload) == conversion.SUMMARY_CALL_KIND
    # The summary call's usage never lands on the agent LLM, and the same reply cannot re-trigger.
    assert len(stub.agent_llm.metrics.token_usages) == 2
    after = stub.View(
        events=events[:2] + [stub.AgentCondensationObservation(id=-1, message="SUMMARY TEXT")],
        unhandled_condensation_request=False,
    )
    assert condenser.should_condense(after) is False
    after.events.append(CmdRunAction(id=9, source="agent"))
    assert condenser.should_condense(after) is False
    stub.agent_llm.metrics.add_token_usage(900, 50)
    assert condenser.should_condense(after) is True
    second = condenser.get_condensation(after)
    assert (second.action.forgotten_events_start_id, second.action.forgotten_events_end_id) == (9, 9)
    assert (
        "<PREVIOUS SUMMARY>\nSUMMARY TEXT\n</PREVIOUS SUMMARY>"
        in stub.posted[1][0]["messages"][0]["content"][0]["text"]
    )
    notes = [json.loads(line) for line in (tmp_path / "compactions.jsonl").read_text().splitlines()]
    assert [n["compaction"] for n in notes] == [1, 2] and notes[0]["forgotten_events"] == 6


def test_condenser_requires_the_agent_llm_and_a_valid_cutoff(monkeypatch, tmp_path):
    stub = _stub_openhands(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="cutoff"):
        compaction_condenser.install(cutoff=1.5, max_context=10, max_event_chars=100)
    compaction_condenser.install(cutoff=0.5, max_context=10, max_event_chars=100)
    with pytest.raises(RuntimeError, match="agent LLM"):
        stub.LLMSummarizingCondenser.from_config(None, SimpleNamespace(service_to_llm={}))
