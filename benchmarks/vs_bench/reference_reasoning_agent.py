# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the author's unmodified reasoning agent over a framed model-call bridge.

External source: https://github.com/zelaix/VS-Bench, MIT, revision
2c3531cdf91d1b620b4a9015126a242eb305a4d0. This bridge invokes the authors'
``agents/prompt_agent.py``, ``models/openai.py`` and game modules directly.
The parent owns model transport and receives every model call over the pipe.
"""

import asyncio
import importlib
import json
import struct
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace

from reference_reasoning import GAMES


def receive(stream):
    header = stream.read(4)
    if len(header) != 4:
        raise EOFError("Missing model bridge frame")
    length = struct.unpack("!I", header)[0]
    data = stream.read(length)
    if len(data) != length:
        raise EOFError("Incomplete model bridge frame")
    return json.loads(data)


def send(stream, value):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    stream.write(struct.pack("!I", len(data)))
    stream.write(data)
    stream.flush()


async def run(author_root, task, incoming, outgoing):
    if task["game"] not in GAMES:
        raise ValueError("Unknown game")
    for name in ("agents", "envs", "models"):
        package = ModuleType(name)
        package.__path__ = [str(author_root / name)]
        sys.modules[name] = package
    sys.path.insert(0, str(author_root))
    from agents.prompt_agent import PromptAgent
    from envs.base_env import BaseEnv
    from models.openai import OpenAIModel

    importlib.import_module("envs." + task["game"])
    environment = object.__new__(BaseEnv._registry[task["game"]])
    model = object.__new__(OpenAIModel)
    model.name = "vs-bench-gym-policy"
    model.version = "policy_model"
    model.reasoning_model = task["reasoning_model"]
    model.stream = False
    model.temperature = task["temperature"]
    model.response_tokens = 8192
    model.reasoning_tokens = 16000
    model.max_response_tokens = task["max_response_tokens"]
    model.max_reasoning_tokens = task["max_reasoning_tokens"]
    model.input_price_per_1M_tokens = 0.0
    model.output_price_per_1M_tokens = 0.0
    calls = 0

    async def create(**kwargs):
        nonlocal calls
        calls += 1
        send(outgoing, {"type": "model_request", "request": kwargs})
        reply = receive(incoming)
        if reply.get("type") != "model_response":
            raise RuntimeError("Model bridge did not return a response")
        usage = SimpleNamespace(
            prompt_tokens=reply.get("prompt_tokens"), completion_tokens=reply.get("completion_tokens")
        )
        choice = SimpleNamespace(
            finish_reason=reply["finish_reason"], message=SimpleNamespace(content=reply["content"])
        )
        return SimpleNamespace(choices=[choice], usage=usage)

    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    agent = object.__new__(PromptAgent)
    agent.env_name = task["game"]
    agent.model = model
    agent.visual_obs = True
    agent.max_errors = task["max_errors"]
    if task.get("track", "reasoning") == "reasoning":
        action, info = await agent.predict(task["dataset_row"], environment.regex_patterns)
        result = {"action": action, "correct": action in task["dataset_row"]["next_action"]}
    elif task["track"] == "perception":
        response, info = await agent.perception(task["dataset_row"]["prompt"], environment.schema)
        result = {"answer": response}
    else:
        raise ValueError("Unknown benchmark track")
    return {
        "type": "complete",
        **result,
        "model_calls": calls,
        "exception_types": [item["type"] for item in info["exception"].values()],
    }


if __name__ == "__main__":
    incoming, outgoing = sys.stdin.buffer, sys.stdout.buffer
    try:
        with redirect_stdout(sys.stderr):
            task = receive(incoming)
            result = asyncio.run(run(Path(sys.argv[1]).resolve(), task, incoming, outgoing))
        send(outgoing, result)
    except Exception as exc:
        send(outgoing, {"type": "error", "error_type": type(exc).__name__})
        raise SystemExit(1) from None
