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
import shutil
import signal
import subprocess
import tempfile
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from anyio import CancelScope
from fastapi import FastAPI, Request, Response
from pydantic import ConfigDict

from nemo_gym.agents import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client import get_response_json, get_server_url, raise_for_status, rollout_path_prefix
from nemo_gym.config import ROLLOUT_PATH_PREFIX, TASK_INDEX_KEY_NAME, ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    AGENT_RUN_ERROR,
    AGENT_TIMEOUT,
    JUDGE_FAILED,
    MODEL_CONTEXT_LIMIT,
    current_rollout_id,
)
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse
from nemo_gym.trajectory import AgentInvocation, ModelCallRef, ObservationGap, TrajectoryRecord, TrajectoryTurn
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.collab_intel.task_data import TaskData


PARTNERS = ["ci-partner-1", "ci-partner-2", "ci-partner-3"]
JUDGES = ["ci-judge-1", "ci-judge-2", "ci-judge-3"]
POLICY = "ci-policy"
# a leading reasoning block. An unclosed one (cut off by max tokens) is all reasoning.
THINK = re.compile(r"^\s*<think>(.*?)(?:</think>\s*|\Z)", re.S)
# the only parts of the server's environment the harness gets: no tokens or keys reach model-written code
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "TMPDIR")


KEY_METRICS = (
    "collab_intel/score",
    "collab_intel/scored_rows",
    "collab_intel/alone_pass_rate",
    "collab_intel/unfinished",
)


def split_reasoning(msg: dict[str, Any]) -> None:
    """Move a leading <think> block of msg's content to msg["reasoning"]."""
    content = msg.get("content")
    if isinstance(content, str) and (m := THINK.match(content)):
        msg["reasoning"], msg["content"] = m.group(1).strip(), content[m.end() :]


def _failed(kind: str, reason: str) -> dict[str, Any]:
    return {"pass": False, "failure_kind": kind, "failure_reason": reason}


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class CollabIntelAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    resources_server: ResourcesServerRef
    data_root: str = "benchmarks/collab_intel/data/repo"
    judge_base_url: str
    judge_model: str
    judge_extra: dict[str, Any] = {"chat_template_kwargs": {"thinking": False}}
    policy_model: str = "policy_model"
    policy_extra: dict[str, Any] = {}
    max_concurrent: int = 16
    episode_timeout_s: int = 7200


class CollabIntelRunRequest(BaseRunRequest, TaskData):
    pass


class CollabIntelVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class CollabIntelAgent(SimpleResponsesAPIAgent):
    config: CollabIntelAgentConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        root = Path(self.config.data_root)
        self._root = (root if root.is_absolute() else Path(__file__).resolve().parents[2] / root).resolve()
        if not (self._root / "pyproject.toml").exists():
            raise FileNotFoundError(
                f"{self._root} has no pyproject.toml: run benchmarks/collab_intel/prepare.py first"
            )
        # the upstream venv lives in the release dir: drop UV_PROJECT_ENVIRONMENT, which would point uv at Gym's own venv
        env = {k: v for k, v in os.environ.items() if k != "UV_PROJECT_ENVIRONMENT"}
        sync = subprocess.run(["uv", "sync", "--frozen", "--project", str(self._root)], capture_output=True, env=env)
        if sync.returncode:
            raise RuntimeError(f"uv sync of {self._root} failed: {sync.stderr.decode(errors='replace')[-4000:]}")
        self._python = str(self._root / ".venv" / "bin" / "python")
        self._tmp = Path(tempfile.mkdtemp(prefix="collab_intel_"))
        self._sem = asyncio.Semaphore(self.config.max_concurrent)
        # response ids the policy route saw, per running phase (the key is in the phase's policy URL)
        self._response_ids: dict[str, list[str]] = {}

    def _env(self, home: Path, phase_key: str) -> dict[str, str]:
        seat = {
            "base_url": self.config.judge_base_url,
            "api_key_env": "CI_GYM_KEY",
            "model": self.config.judge_model,
            "extra": self.config.judge_extra,
            "price": [0, 0],
        }
        # through this agent's /ci_policy route, which strips the reasoning Gym's chat completions put in content
        policy = {
            "base_url": get_server_url(self.config.name).rstrip("/")
            + rollout_path_prefix(current_rollout_id())
            + f"/ci_policy/{phase_key}/v1",
            "api_key_env": "CI_GYM_KEY",
            "model": self.config.policy_model,
            "extra": self.config.policy_extra,
            "price": [0, 0],
        }
        compat = {POLICY: policy, **{name: seat for name in PARTNERS + JUDGES}}
        judge, second, tiebreak = JUDGES
        return {k: os.environ[k] for k in ENV_ALLOWLIST if k in os.environ} | {
            "HOME": str(home),
            "CI_GYM_KEY": "dummy",
            "HARNESS_OPENAI_COMPAT": json.dumps(compat),
            "HARNESS_JUDGE_MODEL": judge,
            "HARNESS_SECOND_RATER": second,
            "HARNESS_TIEBREAK_RATER": tiebreak,
            "HARNESS_BUDGET_USD": "1000000",
            "HARNESS_PRICES": json.dumps({n: [0, 0] for n in compat}),
            "HARNESS_USAGE_LOG": str(self._tmp / "usage.jsonl"),
        }

    async def _run(self, path: Path, version: str, seed: int, out: Path, phase_key: str = "") -> dict[str, Any]:
        """One upstream episode (collab_intel.harness.cli): returns its pass and outcome from the saved transcript, or
        pass false with a failure_kind and failure_reason when it did not finish."""
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
        # its own process group, so a timeout or cancel also stops the pytest runs it starts in coding scenarios
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=self._env(out.parent, phase_key),
            cwd=str(self._root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            log, _ = await asyncio.wait_for(proc.communicate(), timeout=self.config.episode_timeout_s)
        except TimeoutError:
            # the policy's own deadline (Gym's agent_timeout, a model outcome): a measured fail
            _kill_group(proc)
            await proc.wait()
            return _failed(AGENT_TIMEOUT, f"episode timeout after {self.config.episode_timeout_s} s")
        except asyncio.CancelledError:
            # stop the harness before the semaphore slot is released
            with CancelScope(shield=True):
                _kill_group(proc)
                await proc.communicate()
            raise
        runs = list(out.glob(f"*/{version}_seed{seed}.json"))
        if proc.returncode or not runs:
            tail = log.decode(errors="replace")[-2000:]
            reason = tail or f"harness exited with code {proc.returncode} and no transcript"
            # the Gym model server's code for a policy prompt over its context window (policy_chat passes it through):
            # a model outcome. Any other crash (a partner or judge outage, a harness bug) is the evaluation's.
            return _failed(MODEL_CONTEXT_LIMIT if "context_length_exceeded" in tail else AGENT_RUN_ERROR, reason)
        t = json.loads(runs[0].read_text())
        score = t.get("score")
        # a judge failure (upstream judge_failed record) or a missing verdict is unmeasured, not a policy failure
        if t.get("judge_failed") or not isinstance(score, dict) or score.get("pass") is None:
            failed = _failed(JUDGE_FAILED, f"harness transcript has no pass verdict: {t.get('judge_failed') or score}")
            return failed | {"messages": t.get("messages") or []}
        return {
            "pass": bool(score["pass"]),
            "outcome": score.get("outcome"),
            "ended": t.get("ended"),
            "rounds": t.get("rounds_run"),
            # the team chat, kept in the rollout so a score can be audited after the temp dir is gone
            "messages": t.get("messages") or [],
            "final_answers": t.get("final_answers"),
        }

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/ci_policy/{phase_key}/v1/chat/completions")(self.policy_chat)
        app.post(f"/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/ci_policy/{{phase_key}}/v1/chat/completions")(
            self.policy_chat
        )
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application: FastAPI):
            async with parent_lifespan(application) as state:
                try:
                    yield state
                finally:
                    shutil.rmtree(self._tmp, ignore_errors=True)

        app.router.lifespan_context = lifespan
        return app

    async def policy_chat(self, request: Request) -> Any:
        """The policy seat's calls: the Gym model server's /v1/chat/completions returns reasoning wrapped in <think> tags
        at the start of content (vllm_model with uses_reasoning_parser). The harness would post that into the team chat,
        so the partners and judges would read the policy's reasoning. Move it to message.reasoning instead.
        A 4xx (e.g. context_length_exceeded) goes back to the harness as is: it fails at once instead of retrying a 500,
        and _run reads the code from its log."""
        resp = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_request("/v1/chat/completions", request),
            json=await request.json(),
        )
        if 400 <= resp.status < 500:
            return Response(await resp.read(), status_code=resp.status, media_type="application/json")
        await raise_for_status(resp)
        out = await get_response_json(resp)
        for choice in out.get("choices") or []:
            split_reasoning(choice.get("message") or {})
        ids = self._response_ids.get(request.path_params.get("phase_key"))
        if ids is not None and isinstance(out.get("id"), str):
            ids.append(out["id"])
        return out

    async def _phase(self, path: Path, version: str, seed: int, out: Path) -> dict[str, Any]:
        """_run plus what the trajectory needs: the start time and the policy response ids of this phase."""
        key, started = uuid4().hex, time.time()
        self._response_ids[key] = []
        try:
            result = await self._run(path, version, seed, out, key)
        finally:
            ids = self._response_ids.pop(key)
        return {"version": version, "started": started, "response_ids": ids, "result": result}

    def _trajectory(self, body: CollabIntelRunRequest, phases: list[dict[str, Any]]) -> TrajectoryRecord | None:
        """The canonical trajectory: one invocation per phase (solo, then manipulation) with the team chat as the
        policy saw it, and one turn per policy message. Partner, scripted and event messages are user items, never
        turns. A phase cut off before its transcript (timeout, crash) has no messages, so it reports no turn."""
        rollout_id = self.rollout_id_from_run(body)
        if rollout_id is None:
            return None
        extra = body.model_extra or {}
        task_id = next(
            (
                str(extra[k])
                for k in ("task_id", "problem_id", "instance_id", TASK_INDEX_KEY_NAME)
                if extra.get(k) is not None
            ),
            "unknown",
        )
        trajectory = TrajectoryRecord(task_id=task_id, rollout_id=rollout_id)
        for phase in phases:
            version, result = phase["version"], phase["result"]
            messages = result.get("messages") or []
            refs = [ModelCallRef(model_ref=self.config.model_server, response_id=i) for i in phase["response_ids"]]
            if not refs:
                trajectory.gaps.append(ObservationGap(code="model_call_reference_unavailable", invocation_id=version))
            conversation: list[dict[str, Any]] = []
            question: list[dict[str, Any]] = []
            for m in messages:
                text = m.get("text") or ""
                if m.get("kind") != "evaluated":
                    item = {"role": "user", "content": f"{m.get('seat')}: {text}"}
                    question.append(item)
                    conversation.append(item)
                    continue
                conversation.append({"role": "assistant", "content": text})
                turn_no = sum(t.invocation_id == version for t in trajectory.turns) + 1
                trajectory.turns.append(
                    TrajectoryTurn(
                        invocation_id=version,
                        task_id=task_id,
                        rollout_id=rollout_id,
                        turn_no=turn_no,
                        timestamp=phase["started"],
                        question=question,
                        answer=text,
                        step_count=turn_no,
                    )
                )
                question = []
            trajectory.invocations.append(
                AgentInvocation(
                    invocation_id=version,
                    status="completed" if not result.get("failure_kind") else "incomplete" if messages else "failed",
                    error_type=result.get("failure_kind"),
                    model_calls=refs,
                    conversation=conversation,
                )
            )
        return trajectory

    @staticmethod
    def _score(tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Upstream score: test pass rate per dimension over completed rows that passed alone, then the mean over
        dimensions, each weighted equally. A dimension with no such row is left out of the mean. Also the gate pass
        rate over finished rows and the count of unfinished rows, which the score leaves out."""
        by_dimension = defaultdict(list)
        rollouts = [r for task in tasks for r in task]
        for rollout in rollouts:
            if rollout.get("alone_pass") and not rollout.get("mask_sample") and rollout.get("test_pass") is not None:
                by_dimension[rollout["dimension"]].append(float(rollout["test_pass"]))
        rates = {d: sum(v) / len(v) for d, v in sorted(by_dimension.items())}
        metrics = {f"collab_intel/{d}": 100 * rate for d, rate in rates.items()}
        if rates:
            metrics["collab_intel/score"] = 100 * sum(rates.values()) / len(rates)
        metrics["collab_intel/scored_rows"] = sum(len(v) for v in by_dimension.values())
        finished = [bool(r.get("alone_pass")) for r in rollouts if not r.get("unfinished")]
        if finished:
            metrics["collab_intel/alone_pass_rate"] = 100 * sum(finished) / len(finished)
        metrics["collab_intel/unfinished"] = len(rollouts) - len(finished)
        return metrics

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return self._score(tasks)

    def compute_repeat_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return self._score(tasks)

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in agent_metrics.items() if k in KEY_METRICS}

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
        phases: list[dict[str, Any]] = []
        try:
            async with self._sem:
                phases.append(await self._phase(self._root / body.gate_path, "solo", body.gate_seed, out / "alone"))
                alone = phases[0]["result"]
                result["alone"], result["alone_pass"] = alone, alone["pass"]
                if alone["pass"]:
                    phases.append(
                        await self._phase(self._root / body.scenario_path, "manipulation", body.seed, out / "test")
                    )
                    result["test"] = phases[1]["result"]
            trajectory = self._trajectory(body, phases)
        finally:
            # the rollout keeps the messages, final answers and trajectory, so the transcripts and coding files go
            shutil.rmtree(out, ignore_errors=True)
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
        verified = await get_response_json(verify)
        if trajectory is not None:
            verified["ng_trajectory"] = trajectory.model_dump(mode="json")
        return CollabIntelVerifyResponse.model_validate(verified)


if __name__ == "__main__":
    CollabIntelAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = CollabIntelAgent.run_webserver()
