# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collaborative Intelligence (Zou group, arXiv 2611.02633) as a Gym agent.

One row = one released scenario and test seed. The agent runs the upstream harness CLI (collab_intel.harness.cli, from the
pinned repo that benchmarks/collab_intel/prepare.py downloads, in that repo's own uv venv since it needs openai>=3) twice:
the scenario alone (the gate), then, only if the policy passes alone, the test version with the scripted teammate.
The policy is the Gym model server (its /v1/chat/completions). The partner seats and the referee judges are one
self-hosted OpenAI-compatible endpoint (judge_base_url, e.g. a vLLM server), never a hosted API. Reward = the test
run's pass; rows failed alone carry alone_pass=false and are excluded from the score (upstream: "scored only on the
scenarios it passes alone"). Partners and judges differ from the upstream leaderboard, so scores are not comparable to it.
"""

import asyncio
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from pydantic import ConfigDict

from nemo_gym.agents import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client import get_response_json, get_server_url, raise_for_status
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse


PARTNERS = ["ci-partner-1", "ci-partner-2", "ci-partner-3"]
JUDGES = ["ci-judge-1", "ci-judge-2", "ci-judge-3"]
POLICY = "ci-policy"
THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


class CollabIntelAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    resources_server: ResourcesServerRef
    data_root: str = "benchmarks/collab_intel/data/repo"
    judge_base_url: str
    judge_model: str = "deepseek-v41-flash"
    judge_extra: dict[str, Any] = {"chat_template_kwargs": {"thinking": False}}
    policy_model: str = "policy_model"
    policy_extra: dict[str, Any] = {}
    max_concurrent: int = 16
    episode_timeout_s: int = 7200


class CollabIntelRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    dimension: str
    scenario_path: str
    gate_path: str
    gate_seed: int = 0
    seed: int = 0


class CollabIntelVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class CollabIntelAgent(SimpleResponsesAPIAgent):
    config: CollabIntelAgentConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        root = Path(self.config.data_root)
        self._root = (root if root.is_absolute() else Path(__file__).resolve().parents[2] / root).resolve()
        subprocess.run(["uv", "sync", "--frozen", "--project", str(self._root)], check=True, capture_output=True)
        self._python = str(self._root / ".venv" / "bin" / "python")
        self._tmp = Path(tempfile.mkdtemp(prefix="collab_intel_"))
        self._sem = asyncio.Semaphore(self.config.max_concurrent)

    def _env(self) -> dict[str, str]:
        seat = {
            "base_url": self.config.judge_base_url,
            "api_key_env": "CI_GYM_KEY",
            "model": self.config.judge_model,
            "extra": self.config.judge_extra,
            "price": [0, 0],
        }
        # through this agent's /ci_policy route, which strips the reasoning Gym's chat completions put in content
        policy = {
            "base_url": get_server_url(self.config.name).rstrip("/") + "/ci_policy/v1",
            "api_key_env": "CI_GYM_KEY",
            "model": self.config.policy_model,
            "extra": self.config.policy_extra,
            "price": [0, 0],
        }
        compat = {POLICY: policy, **{name: seat for name in PARTNERS + JUDGES}}
        judge, second, tiebreak = JUDGES
        return os.environ | {
            "CI_GYM_KEY": "dummy",
            "HARNESS_OPENAI_COMPAT": json.dumps(compat),
            "HARNESS_JUDGE_MODEL": judge,
            "HARNESS_SECOND_RATER": second,
            "HARNESS_TIEBREAK_RATER": tiebreak,
            "HARNESS_BUDGET_USD": "1000000",
            "HARNESS_PRICES": json.dumps({n: [0, 0] for n in compat}),
            "HARNESS_USAGE_LOG": str(self._tmp / "usage.jsonl"),
        }

    async def _run(self, path: Path, version: str, seed: int, out: Path) -> dict[str, Any]:
        """One upstream episode (collab_intel.harness.cli): returns its pass and outcome from the saved transcript."""
        cmd = [
            self._python,
            "-m",
            "collab_intel.harness.cli",
            str(path),
            "--version",
            version,
            "--seed",
            str(seed),
            "--evaluated",
            POLICY,
            "--partners",
            ",".join(PARTNERS),
            "--out-dir",
            str(out),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=self._env(),
            cwd=str(self._root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            log, _ = await asyncio.wait_for(proc.communicate(), timeout=self.config.episode_timeout_s)
        except TimeoutError:
            # unfinished, not failed: the row is flagged (error) and left out of the score
            proc.kill()
            await proc.wait()
            return {"pass": False, "error": f"episode timeout after {self.config.episode_timeout_s} s"}
        runs = list(out.glob(f"*/{version}_seed{seed}.json"))
        if proc.returncode or not runs:
            return {"pass": False, "error": log.decode(errors="replace")[-2000:]}
        t = json.loads(runs[0].read_text())
        score = t.get("score") or {}
        return {
            "pass": bool(score.get("pass")),
            "outcome": score.get("outcome"),
            "ended": t.get("ended"),
            "rounds": t.get("rounds_run"),
            "judge_failed": "score" not in t,
            "transcript": str(runs[0]),
        }

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/ci_policy/v1/chat/completions")(self.policy_chat)
        return app

    async def policy_chat(self, request: Request) -> dict[str, Any]:
        """The policy seat's calls: the Gym model server's /v1/chat/completions returns reasoning wrapped in <think> tags
        at the start of content (vllm_model with uses_reasoning_parser). The harness would post that into the team chat,
        so the partners and judges would read the policy's reasoning. Move it to message.reasoning instead."""
        resp = await self.server_client.post(
            server_name=self.config.model_server.name, url_path="/v1/chat/completions", json=await request.json()
        )
        await raise_for_status(resp)
        out = await get_response_json(resp)
        for choice in out.get("choices") or []:
            msg = choice.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and (m := THINK.match(content)):
                msg["reasoning"] = m.group(0).strip()[len("<think>") : -len("</think>")]
                msg["content"] = content[m.end() :]
        return out

    async def responses(self, request: Request, body: Any = None):
        raise NotImplementedError("collab_intel_agent only supports /run")

    async def run(self, request: Request, body: CollabIntelRunRequest) -> CollabIntelVerifyResponse:
        seed = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=request.cookies,
        )
        await raise_for_status(seed)
        out = Path(tempfile.mkdtemp(dir=self._tmp))
        result: dict[str, Any] = {"alone_pass": False, "test": None}
        async with self._sem:
            alone = await self._run(self._root / body.gate_path, "solo", body.gate_seed, out / "alone")
            result["alone"], result["alone_pass"] = alone, alone["pass"]
            if alone["pass"]:
                result["test"] = await self._run(
                    self._root / body.scenario_path, "manipulation", body.seed, out / "test"
                )
        response = {
            "id": "collab_intel",
            "object": "response",
            "created_at": 0,
            "model": self.config.policy_model,
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
        verify = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": response, "collab_intel": result},
            cookies=seed.cookies,
        )
        await raise_for_status(verify)
        return CollabIntelVerifyResponse.model_validate(await get_response_json(verify))


if __name__ == "__main__":
    CollabIntelAgent.run_webserver()
