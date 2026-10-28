# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the pinned DeepAgents SDK inside a Gym-supervised task sandbox."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


# This file is staged below a private package root; no task files are imported as code.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse
from langchain_core.runnables import ensure_config
from langgraph.errors import GraphRecursionError

from responses_api_agents.langchain_deepagents_agent.runtime_messages import from_items
from responses_api_agents.langchain_deepagents_agent.runtime_model import (
    ModelRequestError,
    SandboxResponsesModel,
    model_http_session,
)
from responses_api_agents.langchain_deepagents_agent.runtime_observer import (
    DeepAgentsObserver,
    observe_native_summarization,
)


class SupervisedShellBackend(LocalShellBackend):
    """Native file operations with bounded shell calls that tolerate background services.

    The enclosing Gym sandbox supplies isolation. Files replace capture pipes so a
    completed command's background child cannot hold the caller's stdout open.
    """

    def __init__(self, *, workdir: str, scratch: Path, timeout: int) -> None:
        super().__init__(root_dir=workdir, virtual_mode=False, inherit_env=True, timeout=timeout)
        self.scratch = scratch
        self.results: dict[str, int] = {}

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        limit = timeout if timeout is not None else self._default_timeout
        if limit <= 0:
            raise ValueError("Shell timeout must be positive")
        callbacks = ensure_config().get("callbacks")
        run_id = str(getattr(callbacks, "parent_run_id", ""))
        with tempfile.TemporaryFile(dir=self.scratch) as output:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=self.cwd,
                env=self._env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
                code = 124
            output.seek(0)
            raw = output.read(self._max_output_bytes + 1)
        self.results[run_id] = code
        truncated = len(raw) > self._max_output_bytes
        text = raw[: self._max_output_bytes].decode("utf-8", errors="replace")
        if code == 124:
            text += f"\nCommand timed out after {limit} seconds"
        return ExecuteResponse(output=text, exit_code=code, truncated=truncated)


async def run(payload: dict[str, Any], observer: DeepAgentsObserver) -> None:
    """Execute one graph, retaining partial callbacks when the supervisor stops it."""
    observe_native_summarization()
    workdir = payload["workdir"]
    os.chdir(workdir)
    os.environ.update(payload.get("task_env") or {})
    backend = SupervisedShellBackend(
        workdir=workdir, scratch=observer.output.parent, timeout=payload["shell_timeout_seconds"]
    )
    observer.shell_results = backend.results
    async with model_http_session() as http:
        model = SandboxResponsesModel(
            http=http,
            observer=observer,
            url=payload["model_url"],
            request_defaults=payload["model_defaults"],
            assistant_message_header=payload["assistant_message_header"],
            profile={"max_input_tokens": payload["max_input_tokens"]},
        )
        history = observer.output.parent / "conversation-history"
        history.mkdir(exist_ok=True)
        routed_backend = CompositeBackend(
            default=backend, routes={"/conversation_history/": FilesystemBackend(root_dir=history, virtual_mode=True)}
        )
        graph = create_deep_agent(model=model, backend=routed_backend, system_prompt=payload.get("system_prompt"))
        config = {
            "callbacks": [observer],
            "recursion_limit": payload["recursion_limit"],
            "configurable": {"thread_id": observer.rollout_id},
        }
        try:
            await graph.ainvoke({"messages": from_items(payload["input"])}, config=config)
            observer.status = "completed"
        except ModelRequestError as error:
            observer.status, observer.error, observer.error_status = "failed", str(error), error.status
        except GraphRecursionError as error:
            observer.status, observer.stop_reason, observer.error = "incomplete", "max_iterations", str(error)
        except asyncio.CancelledError:
            observer.status = "incomplete"
            observer.stop_reason = "cancelled" if Path(payload["stop_request_path"]).exists() else "wall_time"
        except Exception as error:
            observer.status, observer.error = "failed", f"{type(error).__name__}: {error}"
        finally:
            observer.invocations["root"]["status"] = observer.status
            if observer.stop_reason:
                observer.invocations["root"]["stop_reason"] = observer.stop_reason
            for tool in observer.tools.values():
                if tool["status"] == "incomplete":
                    tool["error_type"] = "Interrupted"
            observer.save()


def main(input_path: str, output_path: str) -> None:
    """Load private input, publish runtime provenance, and honor supervisor SIGTERM."""
    payload = json.loads(Path(input_path).read_text())
    Path(input_path).unlink()
    output = Path(output_path)
    runtime = {
        "pid": os.getpid(),
        "python": sys.executable,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("deepagents", "langchain", "langchain-core", "langgraph", "aiohttp")
        },
    }
    output.with_name("runtime.json").write_text(json.dumps(runtime))
    observer = DeepAgentsObserver(
        output=output,
        task_id=payload["task_id"],
        rollout_id=payload["rollout_id"],
        model_ref=payload["model_ref"],
        sandbox_id=payload["sandbox_id"],
    )
    observer.runtime = runtime
    observer.save()

    async def start() -> None:
        task = asyncio.create_task(run(payload, observer))
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
        await task

    asyncio.run(start())


if __name__ == "__main__":
    main(*sys.argv[1:])
