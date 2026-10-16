# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CaP-X environment worker.

Runs under the CaP-X virtualenv (Python 3.10-3.12), not the Gym venv, so it must not import ``nemo_gym``.
It speaks a JSON-lines protocol: one request per stdin line, one response per line on the original stdout.
Everything else written to stdout (simulator logs, user-code prints) is redirected to stderr so it cannot
corrupt the protocol; user-code output is still captured by CaP-X and returned in step responses.

Requests:
    {"op": "reset", "config_path": str, "seed": int}
        -> {"ok": true, "user_prompt": str, "multi_turn_prompt": str | null}
    {"op": "step", "code": str}
        -> {"ok": true, "reward": float, "task_completed": bool | null, "sandbox_rc": int,
            "stdout": str, "stderr": str, "terminated": bool, "truncated": bool}
Failures return {"ok": false, "error": str}.
"""

import copy
import json
import os
import traceback


def _prompt_text(message: dict) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return "\n".join(part["text"] for part in content if part.get("type") == "text")


class _Worker:
    def __init__(self) -> None:
        self.env = None
        self.config_path = None
        self.multi_turn_prompt = None

    def reset(self, config_path: str, seed: int) -> dict:
        from capx.envs.configs.instantiate import instantiate
        from capx.envs.configs.loader import DictLoader
        from capx.envs.trial import _patch_libero_goal

        if config_path != self.config_path:
            if self.env is not None and hasattr(self.env, "close"):
                self.env.close()
            self.env = None
            self.config_path = None
            env_factory = DictLoader.load([config_path])["env"]
            self.env = instantiate(env_factory)
            self.multi_turn_prompt = env_factory["cfg"].get("multi_turn_prompt")
            self.config_path = config_path

        # Mirrors capx.envs.trial._run_single_trial so seeds map to the same CaP-Bench trials.
        obs, _ = self.env.reset(options={"trial": seed}, seed=seed)
        obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])
        _patch_libero_goal(self.env, obs)
        # full_prompt is [system, user]; the system prompt is the same for every task and lives in the dataset.
        return {
            "user_prompt": _prompt_text(obs["full_prompt"][-1]),
            "multi_turn_prompt": self.multi_turn_prompt,
        }

    def step(self, code: str) -> dict:
        if self.env is None:
            raise RuntimeError("step called before reset")
        _, reward, terminated, truncated, info = self.env.step(code)
        task_completed = info.get("task_completed")
        return {
            "reward": float(reward),
            "task_completed": None if task_completed is None else bool(task_completed),
            "sandbox_rc": int(info["sandbox_rc"]),
            "stdout": info["stdout"],
            "stderr": info["stderr"],
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }


def main() -> None:
    # Generated code runs in this process, so keep the protocol on private descriptors it cannot read, write, or close.
    protocol_in = os.fdopen(os.dup(0), "r")
    protocol_out = os.fdopen(os.dup(1), "w", buffering=1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)

    worker = _Worker()
    for line in protocol_in:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            op = request["op"]
            if op == "reset":
                response = worker.reset(request["config_path"], int(request["seed"]))
            elif op == "step":
                response = worker.step(request["code"])
            else:
                raise ValueError(f"unknown op {op!r}")
            response["ok"] = True
        except Exception:  # Process boundary: report the failure to the resources server instead of dying.
            response = {"ok": False, "error": traceback.format_exc()}
        protocol_out.write(json.dumps(response) + "\n")


if __name__ == "__main__":
    main()
