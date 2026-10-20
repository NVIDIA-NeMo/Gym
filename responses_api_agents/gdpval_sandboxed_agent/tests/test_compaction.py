# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context-compaction segments of an OpenHands rollout (offline: completion-log fixtures only).

Run with the agent venv python:
  .venv/bin/python -m pytest responses_api_agents/gdpval_sandboxed_agent/tests/test_compaction.py
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from responses_api_agents.gdpval_sandboxed_agent import compaction, compaction_condenser

SYSTEM = {"role": "system", "content": [{"type": "text", "text": "You are OpenHands."}]}
TASK = {"role": "user", "content": [{"type": "text", "text": "Write summary.txt from the brief."}]}
TOOLS = [{"type": "function", "function": {"name": "execute_bash", "description": "run", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def _tool_call(n):
    return {"id": f"call_{n}", "type": "function", "function": {"name": "execute_bash", "arguments": json.dumps({"command": f"echo {n}"})}}


def _assistant_request_message(n, *, stale_tokens=True):
    """Assistant message as OpenHands re-sends it: list content + the fork's token fields."""
    message = {"role": "assistant", "content": [], "tool_calls": [_tool_call(n)]}
    if stale_tokens:
        message.update(prompt_token_ids=[9, 9, 9], generation_token_ids=[8, 8], generation_log_probs=[-9.0, -9.0])
    return message


def _tool_message(n):
    return {"role": "tool", "content": [{"type": "text", "text": f"OBSERVATION {n}"}], "tool_call_id": f"call_{n}", "name": "execute_bash"}


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
    psf = {k: reply[k] for k in compaction.TOKEN_KEYS + compaction.OPTIONAL_TOKEN_KEYS if k in reply}
    out = {
        "messages": messages,
        "response": {
            "id": f"chatcmpl-{n}",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": reply, "finish_reason": finish_reason or ("tool_calls" if reply.get("tool_calls") else "stop")}],
            "usage": {"prompt_tokens": 100 + 50 * n, "completion_tokens": 3, "total_tokens": 103 + 50 * n},
        },
        "provider_specific_fields": psf,
        "kwargs": {"model": "gdpval-policy", "tools": tools} if tools else {"model": "gdpval-policy"},
        "timestamp": 1_700_000_000 + n,
    }
    if summary and marker:
        out[compaction.CALL_KIND_KEY] = compaction.SUMMARY_CALL_KIND
    return out


def _write(task: Path, payloads):
    """Completion logs where ``app._download_trace`` puts the sandbox trace: ``task/trace/completions``."""
    folder = task / "trace" / "completions"
    folder.mkdir(parents=True, exist_ok=True)
    for n, payload in enumerate(payloads):
        (folder / f"gdpval-policy-{payload['timestamp']}.{n}.json").write_text(json.dumps(payload))
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
        _payload(2, _summary_request(), _reply(2, tool_call=False, content=summary_text, echo=echo, topk=topk), tools=None, summary=True, marker=marker),
        _payload(3, [SYSTEM, TASK, summary_msg], _reply(3, echo=echo, topk=topk)),
        _payload(4, [SYSTEM, TASK, summary_msg, _assistant_request_message(3), _tool_message(3)],
                 _reply(4, tool_call=False, content="Done, summary.txt written.", echo=echo, topk=topk)),
    ]


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def test_calls_are_split_at_the_marked_summary_call(tmp_path):
    calls = compaction.load_calls(_write(tmp_path, compacting_rollout()))
    assert [c["kind"] for c in calls] == ["agent", "agent", "compaction_summary", "agent", "agent"]
    assert [c["response_id"] for c in calls] == [f"chatcmpl-{n}" for n in range(5)]
    segments = compaction.segments(calls)
    assert [(s["index"], s["reason"], len(s["calls"])) for s in segments] == [
        (0, "", 2), (1, "compaction", 1), (2, "post_compaction", 2),
    ]


def test_summary_call_is_recognised_by_its_prompt_without_the_marker(tmp_path):
    calls = compaction.load_calls(_write(tmp_path, compacting_rollout(marker=False)))
    assert calls[2]["kind"] == compaction.SUMMARY_CALL_KIND
    assert [s["reason"] for s in compaction.segments(calls)] == ["", "compaction", "post_compaction"]


def test_two_compactions_alternate_summary_and_continuation_segments():
    payloads = compacting_rollout()[:4] + [
        _payload(5, _summary_request("S1", ("EVENT 3",)), _reply(5, tool_call=False, content="S2"), tools=None, summary=True),
        _payload(6, [SYSTEM, TASK, {"role": "user", "content": [{"type": "text", "text": "S2"}]}], _reply(6, tool_call=False, content="done")),
    ]
    calls = _calls_from(payloads)
    assert [(s["reason"], len(s["calls"])) for s in compaction.segments(calls)] == [
        ("", 2), ("compaction", 1), ("post_compaction", 1), ("compaction", 1), ("post_compaction", 1),
    ]


def _calls_from(payloads):
    return [
        {
            "kind": compaction.call_kind(p),
            "response_id": p["response"]["id"],
            "messages": [dict(m) for m in p["messages"]],
            "tools": p["kwargs"].get("tools") or [],
            "reply": compaction._reply_from_payload(p),
            "finish_reason": p["response"]["choices"][0]["finish_reason"],
            "timestamp": p["timestamp"],
        }
        for p in sorted(payloads, key=lambda x: x["timestamp"])
    ]


def test_summary_call_before_any_agent_call_is_rejected():
    payloads = [compacting_rollout()[2]]
    with pytest.raises(ValueError, match="before any agent call"):
        compaction.segments(_calls_from(payloads))


def test_no_compaction_yields_one_plain_segment():
    calls = _calls_from(compacting_rollout()[:2])
    assert [(s["index"], s["reason"], len(s["calls"])) for s in compaction.segments(calls)] == [(0, "", 2)]


# ---------------------------------------------------------------------------
# Segment messages: context prefix, per-call token bundles
# ---------------------------------------------------------------------------


def test_segment_messages_attach_each_calls_echo_after_the_context_prefix():
    segments = compaction.segments(_calls_from(compacting_rollout(topk=True)))
    messages0, prefix0 = compaction.segment_messages(segments[0])
    assert prefix0 == 2 and [m["role"] for m in messages0[prefix0:]] == ["assistant", "tool", "assistant"]
    assistants0 = [m for m in messages0[prefix0:] if m["role"] == "assistant"]
    # The stale token fields OpenHands re-sent on the first assistant message are replaced by
    # the bundle of the call that produced it; the final reply keeps its own.
    assert [m["generation_token_ids"] for m in assistants0] == [[1000] * 3, [1001] * 3]
    assert [m["prompt_token_ids"] for m in assistants0] == [list(range(1, 10)), list(range(1, 11))]
    assert all(m["generation_topk_ids"] for m in assistants0)
    assert all("prompt_token_ids" not in m for m in messages0[:prefix0])

    messages1, prefix1 = compaction.segment_messages(segments[1])
    assert prefix1 == 1 and messages1[0]["role"] == "user" and messages1[1]["role"] == "assistant"
    assert messages1[1]["generation_token_ids"] == [1002] * 3 and messages1[1]["content"].startswith("USER_CONTEXT")

    messages2, prefix2 = compaction.segment_messages(segments[2])
    assert [m["role"] for m in messages2[:prefix2]] == ["system", "user", "user"]
    assistants2 = [m for m in messages2[prefix2:] if m["role"] == "assistant"]
    assert [m["generation_token_ids"][0] for m in assistants2] == [1003, 1004]


def test_segment_messages_without_echo_are_text_only():
    segments = compaction.segments(_calls_from(compacting_rollout(echo=False)))
    for segment in segments:
        messages, prefix = compaction.segment_messages(segment)
        assert messages[prefix:] and all("prompt_token_ids" not in m for m in messages)


def test_partial_echo_is_an_error():
    payloads = compacting_rollout()
    payloads[1]["response"]["choices"][0]["message"].pop("prompt_token_ids")
    payloads[1]["provider_specific_fields"] = {}
    segments = compaction.segments(_calls_from(payloads))
    with pytest.raises(ValueError, match="some calls only"):
        compaction.segment_messages(segments[0])


def test_misaligned_logprobs_are_an_error():
    payloads = compacting_rollout()
    payloads[0]["response"]["choices"][0]["message"]["generation_log_probs"] = [-0.1]
    payloads[0]["provider_specific_fields"]["generation_log_probs"] = [-0.1]
    with pytest.raises(ValueError, match="misaligned"):
        compaction.segment_messages(compaction.segments(_calls_from(payloads))[0])


def test_assistant_count_must_match_the_calls():
    payloads = compacting_rollout()
    # The second request "forgot" the first assistant turn: one reply has no message to carry it.
    payloads[1]["messages"] = [SYSTEM, TASK, _tool_message(0)]
    with pytest.raises(ValueError, match="1 assistant messages after the prefix but 2 policy calls"):
        compaction.segment_messages(compaction.segments(_calls_from(payloads))[0])


# ---------------------------------------------------------------------------
# Export: NeMo RL compaction-trace contract
# ---------------------------------------------------------------------------


def _session_traces_module():
    """NeMo RL's dependency-free validator, when the compaction NeMo RL tree sits next to this Gym tree."""
    here = Path(__file__).resolve()
    for root in here.parents:
        candidate = root / "nemo_rl_cc" / "nemo_rl" / "environments" / "session_traces.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("session_traces", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


def test_export_follows_the_compaction_trace_contract(tmp_path):
    responses = compaction.export_segments_from_trace(
        _write(tmp_path, compacting_rollout(topk=True)), model="gdpval-policy", session_id="task-7"
    )
    assert [r.metadata["segment_boundary_reason"] for r in responses] == ["", "compaction", "post_compaction"]
    assert [r.metadata["segment_index"] for r in responses] == ["0", "1", "2"]
    assert [r.id for r in responses] == ["gdpval-task-7-seg0", "gdpval-task-7-seg1", "gdpval-task-7-seg2"]
    owned = []
    for r in responses:
        meta = r.metadata
        assert all(isinstance(v, str) for v in meta.values())
        assert meta["training_mode"] == "compaction" and meta["session_id"] == meta["root_session_id"] == "task-7"
        assert meta["parent_session_id"] == "" and meta["session_role"] == "parent" and meta["spawn_call_id"] == ""
        ids = json.loads(meta["request_ids"])
        assert ids and len(ids) == int(meta["policy_calls"])
        owned += ids
        assert r.status == "completed" and r.model == "gdpval-policy"
    assert owned == [f"chatcmpl-{n}" for n in range(5)]  # every served completion owned exactly once
    # Tools: the agent segments carry the function tools, the summary call offered none.
    assert [len(r.tools) for r in responses] == [1, 0, 1]

    trained = [[i for i in (item.model_dump() for item in r.output) if i.get("generation_token_ids")] for r in responses]
    assert [len(t) for t in trained] == [2, 1, 2]
    for items in trained:
        for item in items:
            assert len(item["generation_token_ids"]) == len(item["generation_log_probs"]) == 3
            assert item["prompt_token_ids"] and item["generation_topk_ids"]
    # Output items cover only the messages after each segment's context prefix.
    kinds0 = [item.type for item in responses[0].output]
    assert kinds0[0] == "function_call" and "function_call_output" in kinds0
    assert len(responses[1].output) == 1 and responses[1].output[0].type == "message"

    validator = _session_traces_module()
    if validator is None:
        pytest.skip("nemo_rl_cc tree not found next to this Gym tree")
    validator.validate_session_trace_metadata([r.metadata for r in responses], training_mode="compaction")
    with pytest.raises(ValueError):
        validator.validate_session_trace_metadata([r.metadata for r in responses], training_mode="plain")


def test_export_without_echo_is_text_only(tmp_path):
    responses = compaction.export_segments_from_trace(
        _write(tmp_path, compacting_rollout(echo=False)), model="gdpval-policy", session_id="t"
    )
    assert len(responses) == 3
    for r in responses:
        assert r.output and all("prompt_token_ids" not in item.model_dump() for item in r.output)


def test_export_requires_completion_logs(tmp_path):
    with pytest.raises(ValueError, match="no completion logs"):
        compaction.export_segments_from_trace(tmp_path, model="m", session_id="t")
    # Regression (smoke-02, 2026-10-18): logs under ``task/completions`` are not where the
    # downloaded trace lives; only ``task/trace/completions`` counts.
    (tmp_path / "completions").mkdir()
    (tmp_path / "completions" / "x.json").write_text(json.dumps(compacting_rollout()[0]))
    with pytest.raises(ValueError, match="no completion logs"):
        compaction.export_segments_from_trace(tmp_path, model="m", session_id="t")
    assert compaction.load_calls(_write(tmp_path, compacting_rollout()))


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
    events = [SystemMessageAction(id=0, source="agent"), MessageAction(id=1, source="user"), CmdRunAction(id=2, source="agent")]
    assert compaction_condenser.head_length(events) == 2
    assert compaction_condenser.head_length([MessageAction(id=0, source="user")]) == 1
    assert compaction_condenser.head_length([CmdRunAction(id=0, source="agent")]) == 1
    # str(EventSource.USER) on a str-mixin Enum
    assert compaction_condenser.head_length([MessageAction(id=0, source="EventSource.USER")]) == 1


def test_compaction_trigger_matches_stirrup_and_never_refires_on_the_same_reply():
    usage = SimpleNamespace(prompt_tokens=350, completion_tokens=70)  # 420 / 600 = 0.7
    kw = dict(max_context=600, cutoff=0.7)
    assert compaction_condenser.should_compact(last_usage=usage, usage_count=3, compacted_at=0, **kw)
    assert not compaction_condenser.should_compact(last_usage=SimpleNamespace(prompt_tokens=350, completion_tokens=69), usage_count=3, compacted_at=0, **kw)
    # Same reply after the compaction: wait for a reply made inside the summarized context.
    assert not compaction_condenser.should_compact(last_usage=usage, usage_count=3, compacted_at=3, **kw)
    assert compaction_condenser.should_compact(last_usage=usage, usage_count=4, compacted_at=3, **kw)
    assert not compaction_condenser.should_compact(last_usage=None, usage_count=0, compacted_at=0, **kw)


def test_summary_prompt_is_the_openhands_prompt_with_truncated_events():
    prompt = compaction_condenser.build_summary_prompt("old summary", [(5, "x" * 50), (6, "short")], max_event_chars=20)
    assert prompt.startswith(compaction.SUMMARY_PROMPT_PREFIX)
    assert "<PREVIOUS SUMMARY>\nold summary\n</PREVIOUS SUMMARY>" in prompt
    assert "<EVENT id=5>\n" + "x" * 10 + "\n[... Observation truncated due to length ...]\n" + "x" * 10 + "\n</EVENT>" in prompt
    assert "<EVENT id=6>\nshort\n</EVENT>\nNow summarize the events using the rules above." in prompt
    # The agent server's fallback detection keys on this prefix.
    assert compaction.call_kind({"messages": [{"role": "user", "content": prompt}]}) == compaction.SUMMARY_CALL_KIND


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
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply_content,
                                                     "prompt_token_ids": [1, 2], "generation_token_ids": [3],
                                                     "generation_log_probs": [-0.5]}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
            self.llm.metrics.add_token_usage(2, 1)
            posted.append((params, tools))
            self._log_completion(messages, response_json, {"prompt_token_ids": [1, 2], "generation_token_ids": [3], "generation_log_probs": [-0.5]}, params)
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
            self.token_usages.append(SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                                     model_dump=lambda p=prompt_tokens, c=completion_tokens: {"prompt_tokens": p, "completion_tokens": c}))

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
        "openhands": {}, "openhands.agenthub": {}, "openhands.core": {}, "openhands.events": {},
        "openhands.events.action": {}, "openhands.events.observation": {}, "openhands.llm": {},
        "openhands.memory": {}, "openhands.memory.condenser": {}, "openhands.memory.condenser.impl": {},
        "openhands.agenthub.nemo_gym_client": {"NemoGymClient": NemoGymClient},
        "openhands.core.message": {"Message": Message, "TextContent": TextContent},
        "openhands.events.action.agent": {"CondensationAction": CondensationAction},
        "openhands.events.observation.agent": {"AgentCondensationObservation": AgentCondensationObservation},
        "openhands.llm.metrics": {"Metrics": Metrics},
        "openhands.memory.condenser.condenser": {"Condensation": Condensation, "RollingCondenser": RollingCondenser, "View": View},
        "openhands.memory.condenser.impl.llm_summarizing_condenser": {"LLMSummarizingCondenser": LLMSummarizingCondenser},
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    folder = tmp_path / "completions"
    folder.mkdir()
    agent_llm = SimpleNamespace(
        config=SimpleNamespace(model="gdpval-policy", base_url="http://policy/v1", log_completions_folder=str(folder)),
        metrics=Metrics(), _nemo_gym_llm_kwargs={"model": "gdpval-policy", "seed": None},
    )
    registry = SimpleNamespace(service_to_llm={"agent": agent_llm})
    return SimpleNamespace(
        posted=posted, agent_llm=agent_llm, registry=registry, folder=folder, View=View,
        AgentCondensationObservation=AgentCondensationObservation, LLMSummarizingCondenser=LLMSummarizingCondenser,
    )


def test_condenser_summarizes_through_the_direct_client_and_marks_the_log(monkeypatch, tmp_path):
    stub = _stub_openhands(monkeypatch, tmp_path)
    compaction_condenser.install(cutoff=0.7, max_context=1000, max_event_chars=10_000, log_path=str(tmp_path / "compactions.jsonl"))
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
    (params, tools), = stub.posted
    assert tools is None and params["model"] == "gdpval-policy"
    assert len(params["messages"]) == 1 and params["messages"][0]["role"] == "user"
    assert params["messages"][0]["content"][0]["text"].startswith(compaction.SUMMARY_PROMPT_PREFIX)
    assert "<EVENT id=2>" in params["messages"][0]["content"][0]["text"]
    # Its completion log carries the marker the agent server splits on, with the echoed tokens.
    logs = list(stub.folder.glob("*.json"))
    assert len(logs) == 1
    payload = json.loads(logs[0].read_text())
    assert payload[compaction.CALL_KIND_KEY] == compaction.SUMMARY_CALL_KIND
    assert compaction.call_kind(payload) == compaction.SUMMARY_CALL_KIND
    assert compaction._reply_from_payload(payload)["generation_token_ids"] == [3]
    # The summary call's usage never lands on the agent LLM, and the same reply cannot re-trigger.
    assert len(stub.agent_llm.metrics.token_usages) == 2
    after = stub.View(events=events[:2] + [stub.AgentCondensationObservation(id=-1, message="SUMMARY TEXT")], unhandled_condensation_request=False)
    assert condenser.should_condense(after) is False
    after.events.append(CmdRunAction(id=9, source="agent"))
    assert condenser.should_condense(after) is False
    stub.agent_llm.metrics.add_token_usage(900, 50)
    assert condenser.should_condense(after) is True
    second = condenser.get_condensation(after)
    assert (second.action.forgotten_events_start_id, second.action.forgotten_events_end_id) == (9, 9)
    assert "<PREVIOUS SUMMARY>\nSUMMARY TEXT\n</PREVIOUS SUMMARY>" in stub.posted[1][0]["messages"][0]["content"][0]["text"]
    notes = [json.loads(line) for line in (tmp_path / "compactions.jsonl").read_text().splitlines()]
    assert [n["compaction"] for n in notes] == [1, 2] and notes[0]["forgotten_events"] == 6


def test_condenser_requires_the_agent_llm_and_a_valid_cutoff(monkeypatch, tmp_path):
    stub = _stub_openhands(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="cutoff"):
        compaction_condenser.install(cutoff=1.5, max_context=10, max_event_chars=100)
    compaction_condenser.install(cutoff=0.5, max_context=10, max_event_chars=100)
    with pytest.raises(RuntimeError, match="agent LLM"):
        stub.LLMSummarizingCondenser.from_config(None, SimpleNamespace(service_to_llm={}))
