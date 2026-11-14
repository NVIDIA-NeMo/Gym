# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A bounded, joined Explore worker over frozen files; no native harness."""

import copy
import hashlib
import json
from time import perf_counter, time

from responses_api_agents.ngc_recorded_context.observation_hooks import tool_observation
from responses_api_agents.ngc_recorded_context.readonly_tools import CoverageGap
from responses_api_agents.ngc_recorded_context.replay_fixtures import shell_tokens


async def explore(arguments, *, state, case, config, model, directory, invoke, write, observe_tool=None):
    if arguments.get("subagent_type") != "Explore" or arguments.get("model") or arguments.get("isolation"):
        return "Only inherited-model readonly Explore is available in this API replay.", "fixture_gap"
    if arguments.get("run_in_background"):
        return "Background agent scheduling is unavailable in this API replay.", "fixture_gap"
    directory.mkdir()
    tools = [copy.deepcopy(t) for t in case["tools"] if t["function"]["name"] in {"Read", "Bash"}]
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    messages = [
        {
            "role": "system",
            "content": "You are a read-only Explore worker in an API replay. Inspect the pinned source using Read and Bash; only bounded file reads, listings and source search are available. No network, writes, shell execution, other agents or native reviewer runtime. Return concise findings with paths and lines backed by returned tool output. Clearly identify unavailable evidence. Your response is returned to the parent; do not write its answer files.",
        },
        {"role": "user", "content": arguments["prompt"]},
    ]
    events, termination = [], "delegate_turn_limit"
    seen = set()
    final = ""
    for turn in range(1, config.max_delegate_turns + 1):
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
        }
        prefix = directory / f"call-{turn:02d}"
        write(prefix.with_suffix(".request.json"), payload)
        try:
            status, response = await invoke(payload, config.timeout_s, prefix)
            write(prefix.with_suffix(".response.json"), response)
            write(prefix.with_suffix(".receipt.json"), {"http_status": status})
            if status != 200 or response.get("model") != model or len(response.get("choices", [])) != 1:
                raise ValueError("Invalid delegated provider response")
            choice = response["choices"][0]
            msg = choice["message"]
            if (
                choice["finish_reason"] not in {"stop", "tool_calls"}
                or msg.get("refusal")
                or msg["role"] != "assistant"
            ):
                raise ValueError("Incomplete delegated response")
            assistant = {k: msg[k] for k in ("role", "content", "reasoning_content") if k in msg}
            calls = msg.get("tool_calls") or []
            if calls:
                assistant["tool_calls"] = [{k: c[k] for k in ("id", "type", "function")} for c in calls]
            messages.append(assistant)
            if not calls:
                final = msg.get("content") or ""
                if not final.strip():
                    raise ValueError("Empty delegated answer")
                termination = "final_response"
                break
            for call in calls:
                observed_at, observed_monotonic = time(), perf_counter()
                if call["type"] != "function" or not call["id"] or call["id"] in seen:
                    raise ValueError("Invalid delegated tool ID/type")
                seen.add(call["id"])
                # Dispatch validates the unchanged registered argument schema.
                if call["function"]["name"] not in schemas:
                    result, kind = "Worker tool is not available.", "fixture_gap"
                elif call["function"]["name"] == "Bash":
                    import jsonschema

                    args = json.loads(call["function"]["arguments"])
                    jsonschema.Draft202012Validator(schemas["Bash"]).validate(args)
                    try:
                        result, kind = state.readonly.run(shell_tokens(state.path(args["command"])))
                    except CoverageGap as error:
                        result, kind = f"Readonly worker coverage unavailable: {error}", "fixture_gap"
                else:
                    result, kind = state.dispatch(call, schemas)
                if observe_tool is not None:
                    observe_tool(
                        tool_observation(
                            call,
                            result,
                            kind,
                            state,
                            observed_at,
                            observed_monotonic,
                            conversation_id=directory.name,
                        )
                    )
                events.append(
                    {
                        "turn": turn,
                        "tool_call_id": call["id"],
                        "function": call["function"],
                        "kind": kind,
                        "result": result,
                    }
                )
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            if any(e["kind"] in {"fixture_gap", "invalid_call"} for e in events if e["turn"] == turn):
                termination = "fixture_gap"
                break
        except Exception as error:
            write(directory / "error.json", {"type": type(error).__name__, "detail": str(error)})
            termination = "delegate_error"
            break
    write(directory / "conversation.json", messages)
    write(directory / "events.json", events)
    partial = termination == "delegate_turn_limit" and bool(events)
    write(
        directory / "result.json",
        {
            "termination": termination,
            "model": model,
            "readonly": True,
            "native_harness": False,
            "completed": termination == "final_response",
            "partial_evidence_returned": partial,
        },
    )
    if partial:
        # A bounded worker can return its actual intermediate evidence without
        # claiming it finished. The parent still owns the answer and IF score.
        evidence = [
            {
                "turn": event["turn"],
                "function": event["function"],
                "kind": event["kind"],
                "result_excerpt": event["result"][:2400],
                "result_truncated": len(event["result"]) > 2400,
                "result_sha256": hashlib.sha256(event["result"].encode()).hexdigest(),
            }
            for event in events
        ]
        return (
            "Readonly Explore reached its response limit without a final summary. These are its actual returned tool results, not a completed investigation or reviewer approval. Full results are retained in the worker capture.\n\n"
            + json.dumps(evidence, ensure_ascii=False, indent=2),
            "delegate_partial",
        )
    if termination != "final_response":
        return (
            f"Readonly Explore worker did not complete: {termination}. No reviewer approval or investigation finding is inferred.",
            "fixture_gap",
        )
    return (
        "Readonly Explore completed synchronously in the API replay using the parent model. No native harness or review approval is implied.\n\n"
        + final,
        "delegate_result",
    )
