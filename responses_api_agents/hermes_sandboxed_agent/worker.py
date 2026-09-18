# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run native Hermes in an isolated process with Gym model and sandbox transports."""

import json
import sys
from collections import defaultdict, deque
from concurrent.futures import Future
from itertools import count
from pathlib import Path
from threading import Lock, Thread
from time import time
from types import SimpleNamespace


class GymTransport:
    def __init__(self):
        self.lock = Lock()
        self.output = sys.stdout
        sys.stdout = sys.stderr
        self.pending = {}
        self.ids = count()
        Thread(target=self.receive, daemon=True).start()

    def receive(self):
        try:
            for line in sys.stdin:
                reply = json.loads(line)
                with self.lock:
                    future = self.pending.pop(reply["id"])
                if "error" in reply:
                    future.set_exception(RuntimeError(reply["error"]))
                else:
                    future.set_result(reply["result"])
        finally:
            with self.lock:
                for future in self.pending.values():
                    future.set_exception(RuntimeError("Gym transport closed"))
                self.pending.clear()

    def call(self, operation, **payload):
        with self.lock:
            request_id = next(self.ids)
            future = self.pending[request_id] = Future()
            self.output.write(json.dumps({"id": request_id, "operation": operation, **payload}) + "\n")
            self.output.flush()
        return future.result()


class GymModel:
    def __init__(self, transport):
        self.transport = transport
        self.chat = SimpleNamespace(completions=self)
        self.is_closed = False

    def close(self):
        self.is_closed = True

    def create(self, **kwargs):
        from openai.types.chat import ChatCompletion

        return ChatCompletion.model_validate(self.transport.call("model", kwargs=kwargs))


class SandboxEnvironment:
    def __init__(self, transport, context, timeout):
        self.transport = transport
        self.cwd = context["workdir"] or "/"
        self.timeout = timeout

    def execute(self, command, cwd="", *, timeout=None, stdin_data=None):
        return self.transport.call(
            "sandbox",
            command=command,
            cwd=cwd or self.cwd,
            timeout=min(timeout or self.timeout, self.timeout),
            stdin_data=stdin_data,
        )

    def cleanup(self):
        # Gym owns the sandbox lifecycle, including process cleanup.
        pass


def main():
    payload = json.loads(sys.stdin.readline())
    transport = GymTransport()
    from run_agent import AIAgent
    from tools.process_registry import process_registry
    from tools.registry import registry
    from tools.terminal_tool import _active_environments, _env_lock, _last_activity, register_task_env_overrides
    from toolsets import resolve_toolset

    context = payload["context"]
    task_id = context["session_id"]
    environment = SandboxEnvironment(transport, context, payload["step_timeout"])
    # The pinned native registry accepts a backend for each task; an unknown
    # backend type fails closed if the registration is ever lost.
    with _env_lock:
        _active_environments[task_id] = environment
        _last_activity[task_id] = time()
    register_task_env_overrides(task_id, {"cwd": environment.cwd})
    completed_tools = set()

    def tool_start(call_id, name, args):
        transport.call("tool_start", call_id=call_id, name=name, args=args)

    def tool_complete(call_id, name, args, result):
        from run_agent import _detect_tool_failure

        if call_id in completed_tools:
            return
        transport.call(
            "tool_complete",
            call_id=call_id,
            name=name,
            args=args,
            result=result,
            failed=_detect_tool_failure(name, result)[0],
        )
        completed_tools.add(call_id)

    class SandboxedAgent(AIAgent):
        def _create_openai_client(self, client_kwargs, *, reason, shared):
            return GymModel(transport)

        def _execute_tool_calls(self, assistant_message, messages, effective_task_id, api_call_count=0):
            self.pending_calls = defaultdict(deque)
            self.calls_lock = Lock()
            for call in assistant_message.tool_calls:
                key = (call.function.name, json.dumps(json.loads(call.function.arguments), sort_keys=True))
                self.pending_calls[key].append(call.id)
            transport.call("messages", messages=messages)
            super()._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)
            error = transport.call("messages", messages=messages)
            if error:
                self.interrupt(error)

        def _invoke_tool(self, function_name, function_args, effective_task_id):
            # Native parallel batches delay completion callbacks until all tools finish.
            with self.calls_lock:
                key = (function_name, json.dumps(function_args, sort_keys=True))
                call_id = self.pending_calls[key].popleft()
            result = super()._invoke_tool(function_name, function_args, effective_task_id)
            tool_complete(call_id, function_name, function_args, result)
            return result

    agent = SandboxedAgent(
        **payload["agent_config"],
        api_mode="chat_completions",
        use_streaming=False,
        enabled_toolsets=[],
        skip_context_files=True,
        skip_memory=True,
        persist_session=False,
        save_trajectories=False,
        checkpoints_enabled=False,
        base_url="http://gym.invalid/v1",
        api_key="dummy-key",
        model=payload["model_name"],
        max_iterations=payload["max_turns"],
        max_tokens=payload["max_tokens"],
        tool_start_callback=tool_start,
        tool_complete_callback=tool_complete,
    )
    # Native compression uses an auxiliary client outside Gym's model transport.
    agent.compression_enabled = False
    tool_names = {name for toolset in payload["toolsets"] for name in resolve_toolset(toolset)}
    agent.tools = [{"type": "function", "function": registry.get_schema(name)} for name in sorted(tool_names)]
    agent.valid_tool_names = tool_names
    try:
        result = agent.run_conversation(payload["instruction"], task_id=task_id)
        Path(payload["trajectory"]).write_text(json.dumps({"messages": result.get("messages", [])}, indent=2))
        process_registry.kill_all(task_id=task_id)
        transport.call("result", result=result)
    finally:
        agent.client.close()


if __name__ == "__main__":
    main()
