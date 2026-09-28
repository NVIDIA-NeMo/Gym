# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one Hermes conversation inside a Gym sandbox."""

from __future__ import annotations

import asyncio
import functools
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from run_agent import AIAgent


try:
    from .sandbox_observer import SandboxHermesObserver
except ImportError:
    from sandbox_observer import SandboxHermesObserver


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)


_RELAY_HOST = "nemo-gym-model-relay.invalid"
_RELAY_BASE_URL = f"http://{_RELAY_HOST}/v1"


class FileModelRelay:
    """Exchange model requests with the controlling agent server, which answers them in index order."""

    def __init__(self, exchange_dir: Path) -> None:
        self.exchange_dir = exchange_dir
        self.request_index = 0
        # Delegated children call the model from their own threads.
        self._index_lock = threading.Lock()

    def exchange(self, body: dict[str, Any]) -> dict[str, Any]:
        with self._index_lock:
            request_id = self.request_index
            self.request_index += 1
        request_path = self.exchange_dir / f"model-request-{request_id}.json"
        response_path = self.exchange_dir / f"model-response-{request_id}.json"
        _write_atomic(request_path, body)

        while not response_path.exists():
            time.sleep(0.1)

        payload = json.loads(response_path.read_text())
        response_path.unlink()
        return payload

    def respond(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
            return _relay_error(request, 404, f"The model relay serves only chat completions: {request.url.path}")
        body = json.loads(request.content)
        if body.get("stream"):
            return _relay_error(request, 400, "The model relay does not support streaming")
        payload = self.exchange(body)
        if payload.get("error") is not None:
            return _relay_error(request, payload.get("status") or 502, str(payload["error"]))
        return httpx.Response(200, json=payload["response"], request=request)


def _relay_error(request: httpx.Request, status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message}}, request=request)


def _route_model_clients_through(relay: FileModelRelay) -> None:
    """Send every model request Hermes makes in this process through the relay.

    The root agent, its iteration-limit summary, delegated children, and auxiliary clients such as
    context compression each build their own OpenAI client, so the relay sits under all of them at
    the HTTP transport. Requests to other hosts are sent normally.
    """
    os.environ["OPENAI_BASE_URL"] = _RELAY_BASE_URL
    os.environ["OPENAI_API_KEY"] = "model-relay"
    handle_request = httpx.HTTPTransport.handle_request
    handle_async_request = httpx.AsyncHTTPTransport.handle_async_request

    def relay_request(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        if request.url.host != _RELAY_HOST:
            return handle_request(transport, request)
        request.read()
        return relay.respond(request)

    async def relay_async_request(transport: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        if request.url.host != _RELAY_HOST:
            return await handle_async_request(transport, request)
        await request.aread()
        return await asyncio.to_thread(relay.respond, request)

    httpx.HTTPTransport.handle_request = relay_request
    httpx.AsyncHTTPTransport.handle_async_request = relay_async_request

    # The relay exchanges whole responses, so no agent may stream, including the children Hermes builds.
    initialize = AIAgent.__init__

    @functools.wraps(initialize)
    def initialize_without_streaming(agent: AIAgent, *args: Any, **kwargs: Any) -> None:
        initialize(agent, *args, **{**kwargs, "use_streaming": False})

    AIAgent.__init__ = initialize_without_streaming


def _run(payload: dict[str, Any], exchange_dir: Path) -> dict[str, Any]:
    hermes_home = exchange_dir / "hermes-home"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(payload["config_yaml"])
    os.environ["HERMES_HOME"] = str(hermes_home)
    os.environ["TERMINAL_ENV"] = "local"
    os.environ["TERMINAL_TIMEOUT"] = str(payload["terminal_timeout"])
    _route_model_clients_through(FileModelRelay(exchange_dir))

    agent = AIAgent(
        base_url=_RELAY_BASE_URL,
        api_key="model-relay",
        model=payload["model"],
        temperature=payload["temperature"],
        insert_reasoning=True,
        max_iterations=payload["max_turns"],
        max_tokens=payload["max_tokens"],
        enabled_toolsets=payload["enabled_toolsets"],
        disabled_toolsets=payload["disabled_toolsets"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        persist_session=False,
        save_trajectories=False,
    )
    observer = SandboxHermesObserver().instrument(agent)

    original_build_api_kwargs = agent._build_api_kwargs

    def build_api_kwargs(api_messages: list[dict[str, Any]]) -> dict[str, Any]:
        kwargs = original_build_api_kwargs(api_messages)
        if not payload["chat_template_kwargs_enabled"]:
            return kwargs
        chat_template_kwargs = kwargs.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})
        chat_template_kwargs.setdefault("enable_thinking", True)
        chat_template_kwargs["truncate_history_thinking"] = False
        return kwargs

    agent._build_api_kwargs = build_api_kwargs
    result = None
    error = None
    try:
        result = agent.run_conversation(
            payload["user_message"],
            payload["system_message"],
            payload["history"],
            task_id=payload["agent_session_id"],
        )
    except BaseException as exception:
        error = exception
        setattr(exception, "_sandbox_observations", observer.finish(result, error))
        raise
    return {
        "observations": observer.finish(result, error),
        "result": result,
        "runtime": {
            "hostname": os.uname().nodename,
            "pid": os.getpid(),
            "python": sys.executable,
        },
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: sandbox_runner.py INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    exchange_dir = input_path.parent
    try:
        output = _run(json.loads(input_path.read_text()), exchange_dir)
    except BaseException as error:
        output = {
            "error": str(error),
            "error_type": type(error).__name__,
            "observations": getattr(error, "_sandbox_observations", None),
            "traceback": traceback.format_exc(),
            "runtime": {
                "hostname": os.uname().nodename,
                "pid": os.getpid(),
                "python": sys.executable,
            },
        }
        _write_atomic(output_path, output)
        return 1

    _write_atomic(output_path, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
