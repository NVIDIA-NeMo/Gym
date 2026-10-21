# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runs inside the task sandbox (python>=3.12 venv with openhands-sdk/openhands-tools 1.50.0).

Usage: python runner.py <request.json> <result.json>

Mirrors the harness that produced the vahid_1016 SFT data (OpenHands SDK 1.50: terminal,
file_editor, task_tracker + built-in finish/think, native tool calling, no condenser). The public
SDK reproduces the SFT rows' system prompt and the terminal/task_tracker/finish/think tool schemas
exactly; the patched `+nl2repo.tools2` build only differs in file_editor's description.
"""

import json
import sys
import time
import traceback


def main(request_path: str, result_path: str) -> None:
    req = json.load(open(request_path))
    result = {"finished": False, "error": None, "messages": [], "metrics": None}
    started = time.time()
    try:
        from openhands.sdk import LLM, Agent, Conversation, Tool
        from openhands.sdk.event import LLMConvertibleEvent
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.task_tracker import TaskTrackerTool
        from openhands.tools.terminal import TerminalTool

        # Gym's model server validates chat messages strictly (NeMoGymChatCompletionToolMessageParam
        # forbids extra fields), but the SDK adds `name` to tool-result messages. The chat template
        # doesn't use it, so drop it on the wire.
        import openhands.sdk.llm.llm as _llm_module

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
        conversation = Conversation(
            agent=agent,
            workspace=req["workdir"],
            max_iteration_per_run=req["max_iterations"],
            callbacks=[events.append],
            visualizer=None,
        )
        conversation.send_message(req["prompt"])
        conversation.run()

        state = conversation.state
        result["execution_status"] = str(getattr(state.execution_status, "value", state.execution_status))
        result["finished"] = result["execution_status"] == "finished"
        convertible = [e for e in state.events if isinstance(e, LLMConvertibleEvent)]
        result["messages"] = [
            m.to_chat_dict(
                cache_enabled=False,
                vision_enabled=False,
                function_calling_enabled=True,
                force_string_serializer=True,
                send_reasoning_content=True,
            )
            for m in LLMConvertibleEvent.events_to_messages(convertible)
        ]
        result["event_types"] = [type(e).__name__ for e in state.events]
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
        conversation.close()
    except BaseException as exc:  # noqa: BLE001 - always write a result file
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]
    result["duration_s"] = time.time() - started
    with open(result_path, "w") as f:
        json.dump(result, f, default=str)
    print("OpenHands run finished" if result["finished"] else f"OpenHands run ended: {result['error'] or result.get('execution_status')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
