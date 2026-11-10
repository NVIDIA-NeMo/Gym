# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the unmodified VS-Bench episode loop with a framed policy bridge.

Source: https://github.com/zelaix/VS-Bench, MIT, revision
2c3531cdf91d1b620b4a9015126a242eb305a4d0. The original `run_episode`,
environment, opponents, agent and prompts run directly from that checkout.
Only model transport and lightweight episode logging are supplied here.
"""

import asyncio
import importlib
import json
import os
import struct
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace

from reference_reasoning import GAMES


def send(stream, value):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    stream.write(struct.pack("!I", len(data)))
    stream.write(data)
    stream.flush()


async def receive(reader):
    size = struct.unpack("!I", await reader.readexactly(4))[0]
    return json.loads(await reader.readexactly(size))


async def run(author_root, task, incoming, outgoing):
    game = task["game"]
    if game not in GAMES or not isinstance(task["episode_id"], int) or task["episode_id"] < 0:
        raise ValueError("Invalid VS-Bench game or episode")
    root = Path(task["episode_dir"])
    if not root.is_absolute():
        raise ValueError("Episode images require an explicit path")
    root.mkdir(parents=True, exist_ok=False)
    for name in ("agents", "envs", "models"):
        package = ModuleType(name)
        package.__path__ = [str(author_root / name)]
        sys.modules[name] = package
    sys.path.insert(0, str(author_root))
    os.chdir(author_root)
    from models.base_model import BaseModel
    from models.openai import OpenAIModel
    from utils.helper import load_decision_config

    importlib.import_module("envs." + game)
    for module in ("prompt_agent", "builtin_agent", "cfr_agent", "mcts_agent"):
        importlib.import_module("agents." + module)
    from eval import decision_making

    model_calls = 0
    next_id = 0
    pending = {}

    async def reader_loop():
        while True:
            frame = await receive(incoming)
            future = pending.pop(frame["request_id"])
            if frame["type"] == "model_response":
                future.set_result(frame)
            else:
                future.set_exception(RuntimeError("Model bridge transport failed"))

    reader_task = asyncio.create_task(reader_loop())

    async def create(**kwargs):
        nonlocal model_calls, next_id
        request_id = next_id
        next_id += 1
        model_calls += 1
        future = asyncio.get_running_loop().create_future()
        pending[request_id] = future
        send(outgoing, {"type": "model_request", "request_id": request_id, "request": kwargs})
        reply = await future
        usage = SimpleNamespace(
            prompt_tokens=reply.get("prompt_tokens"), completion_tokens=reply.get("completion_tokens")
        )
        choice = SimpleNamespace(
            finish_reason=reply["finish_reason"], message=SimpleNamespace(content=reply["content"])
        )
        return SimpleNamespace(choices=[choice], usage=usage)

    def create_author_model(_cls, _spec):
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
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return model

    BaseModel.from_config = classmethod(create_author_model)

    class Collector:
        def __init__(self):
            self.steps = 0
            self.returns = None

        def get_episode_dir(self, _):
            return str(root)

        def episode_start(self, _):
            pass

        def step(self, _, step_info):
            self.steps += 1

        def episode_end(self, _, env_info):
            self.returns = [float(value) for value in env_info["returns"]]

    try:
        if task.get("mode", "episode") == "suite":
            original_logger = decision_making.Logger
            captured = {}

            def create_logger(config):
                config["experiment"]["results_dir"] = str(root)
                captured["model_seats"] = [
                    i for i, agent in enumerate(config["agents"]) if agent["type"] == "prompt_agent"
                ]
                for agent in config["agents"]:
                    model = agent.get("params", {}).get("model")
                    if model:
                        model["params"]["name"] = "vs-bench-gym-policy"
                        model["params"]["version"] = "policy_model"
                logger = original_logger(config)
                captured["logger"] = logger
                return logger

            decision_making.Logger = create_logger
            await decision_making.run_decision_making(game, model=None)
            logger = captured["logger"]
            return {
                "type": "complete",
                "game": game,
                "mode": "suite",
                "model_seats": captured["model_seats"],
                "returns": logger.returns.tolist(),
                "steps": logger.steps.tolist(),
                "model_calls": model_calls,
                "source_results_path": logger.json_file,
            }
        collector = Collector()
        config = load_decision_config(game, None)
        episode = task.get("paper_episode")
        if episode is not None:
            import numpy as np

            seat = episode["model_seat"]
            if game not in ("kuhn_poker", "breakthrough") or seat not in (0, 1):
                raise ValueError("Invalid competitive episode")
            agents = config["agents"]
            if len(agents) != 2 or sum(a["type"] == "prompt_agent" for a in agents) != 1:
                raise ValueError("Expected one model-controlled player")
            if agents[seat]["type"] != "prompt_agent":
                agents.reverse()
            # Each episode runs in an isolated process. Set deterministic
            # opponent randomness without coupling the deal to seat parity.
            opponent_seed = episode["block"] * 1000 + episode["index"]
            np.random.seed(opponent_seed)
            config["experiment"]["seed"] = episode["environment_seed"]
            episode_id = 0
        else:
            seat = None
            episode_id = task["episode_id"]
        await decision_making.run_episode(episode_id, config, collector)
        return {
            "type": "complete",
            "game": game,
            "episode_id": task["episode_id"],
            "returns": collector.returns,
            "steps": collector.steps,
            "model_seats": [seat] if seat is not None else [],
            "model_calls": model_calls,
            "source_results_path": str(root),
        }
    finally:
        reader_task.cancel()
        await asyncio.gather(reader_task, return_exceptions=True)


async def main(outgoing):
    reader = asyncio.StreamReader()
    loop = asyncio.get_running_loop()
    transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer)
    try:
        task = await receive(reader)
        result = await run(Path(sys.argv[1]).resolve(), task, reader, outgoing)
        send(outgoing, result)
    finally:
        transport.close()


if __name__ == "__main__":
    outgoing = sys.stdout.buffer
    try:
        with redirect_stdout(sys.stderr):
            asyncio.run(main(outgoing))
    except Exception as exc:
        send(outgoing, {"type": "error", "error_type": type(exc).__name__, "message": str(exc)})
        raise SystemExit(1) from None
