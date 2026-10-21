# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""runner.py plus crash resilience, for the background-exec path of app.py.

Differences from runner.py (kept untouched so in-flight evals that already use it are unaffected):
  - writes a partial result file every `partial_every` events, atomically, so a runner that is killed
    (sandbox OOM, TTL, lost exec) still leaves a usable trajectory;
  - records ConversationErrorEvent codes (e.g. MaxIterationsReached) so run endings can be classified.

Usage: python runner_v2.py <request.json> <result.json>
"""

import json
import os
import sys
import time
import traceback

_CHAT_KWARGS = dict(
    cache_enabled=False,
    vision_enabled=False,
    function_calling_enabled=True,
    force_string_serializer=True,
    send_reasoning_content=True,
)


def _write_atomic(path: str, result: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, default=str)
    os.replace(tmp, path)


def main(request_path: str, result_path: str) -> None:
    req = json.load(open(request_path))
    result = {"finished": False, "partial": True, "error": None, "messages": [], "metrics": None, "errors": []}
    started = time.time()
    snap = {}
    try:
        import openhands.sdk.llm.llm as _llm_module
        from openhands.sdk import LLM, Agent, Conversation, Tool
        from openhands.sdk.event import LLMConvertibleEvent
        from openhands.sdk.event.conversation_error import ConversationErrorEvent
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.task_tracker import TaskTrackerTool
        from openhands.tools.terminal import TerminalTool

        # Gym's model server validates chat messages strictly; drop the SDK's extra `name` on tool results.
        _orig_completion = _llm_module.litellm_completion

        def _completion_without_tool_names(*args, **kwargs):
            for message in kwargs.get("messages") or []:
                if isinstance(message, dict) and message.get("role") == "tool":
                    message.pop("name", None)
            return _orig_completion(*args, **kwargs)

        _llm_module.litellm_completion = _completion_without_tool_names

        llm_kwargs = dict(
            model=f"openai/{req['model']}",
            base_url=req["base_url"],
            api_key="dummy_key",  # pragma: allowlist secret
            timeout=req.get("model_call_timeout_s"),
            num_retries=req.get("num_retries", 5),
            native_tool_calling=True,
            stream=False,
        )
        for key in ("temperature", "top_p", "max_output_tokens", "reasoning_effort"):
            if req.get(key) is not None:
                llm_kwargs[key] = req[key]
        if req.get("litellm_extra_body"):
            llm_kwargs["litellm_extra_body"] = req["litellm_extra_body"]
        llm = LLM(**llm_kwargs)

        agent = Agent(
            llm=llm,
            tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name), Tool(name=TaskTrackerTool.name)],
        )
        events = []
        partial_every = req.get("partial_every", 25)

        def snapshot(final: bool) -> None:
            convertible = [e for e in events if isinstance(e, LLMConvertibleEvent)]
            result["messages"] = [
                m.to_chat_dict(**_CHAT_KWARGS) for m in LLMConvertibleEvent.events_to_messages(convertible)
            ]
            result["errors"] = [
                {"code": e.code, "detail": e.detail} for e in events if isinstance(e, ConversationErrorEvent)
            ]
            result["n_events"] = len(events)
            result["duration_s"] = time.time() - started
            result["partial"] = not final
            _write_atomic(result_path, result)

        snap["fn"] = snapshot

        def on_event(event) -> None:
            events.append(event)
            if len(events) % partial_every == 0:
                try:
                    snapshot(final=False)
                except Exception:  # noqa: BLE001 - never let a snapshot kill the run
                    pass

        conversation = Conversation(
            agent=agent,
            workspace=req["workdir"],
            max_iteration_per_run=req["max_iterations"],
            callbacks=[on_event],
            visualizer=None,
        )
        conversation.send_message(req["prompt"])
        conversation.run()

        state = conversation.state
        result["execution_status"] = str(getattr(state.execution_status, "value", state.execution_status))
        result["finished"] = result["execution_status"] == "finished"
        metrics = conversation.conversation_stats.get_combined_metrics()
        usage = metrics.accumulated_token_usage
        result["metrics"] = {
            "cost": metrics.accumulated_cost,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "cache_read_tokens": getattr(usage, "cache_read_tokens", 0),
            "reasoning_tokens": getattr(usage, "reasoning_tokens", 0),
            "n_llm_calls": len(metrics.token_usages),
        }
        snapshot(final=True)
        conversation.close()
    except BaseException as exc:  # noqa: BLE001 - always write a result file
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]
        result["duration_s"] = time.time() - started
        try:
            if "fn" in snap:
                snap["fn"](final=False)  # keeps the trajectory gathered so far
            _write_atomic(result_path, result | {"partial": True})
        except Exception:  # noqa: BLE001
            pass
    print("OpenHands run finished" if result["finished"] else f"OpenHands run ended: {result['error'] or result.get('execution_status')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
