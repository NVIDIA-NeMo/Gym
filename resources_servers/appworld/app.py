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
"""AppWorld resources server (native gym multi-turn).

AppWorld (StonyBrookNLP/appworld, ACL'24 Best Resource Paper) is a code-as-action
benchmark: an agent solves day-to-day tasks for a simulated supervisor by writing
Python against 457 APIs spanning 9 apps, and is graded by database-state tests
rather than by its code.

Episode lifecycle, following the aviary/toolsandbox env pattern:

* ``seed_session(task_id)`` leases a worker process, initializes the task on it,
  and returns the agent-facing conversation head (system prompt + the
  supervisor/instruction turn) plus the single code-execution tool.
* ``/step(code)`` forwards the code to the worker's IPython shell and returns
  whatever it printed. ``done`` flips once the agent has called
  ``apis.supervisor.complete_task()`` or the interaction budget is spent.
* ``/close`` runs AppWorld's evaluation against the final database state, caches
  the reward, and returns the worker to the pool.
* ``/verify`` is pure: it returns that cached reward.

Concurrency lives in ``worker_pool.py`` — one live AppWorld environment per OS
process is a hard upstream constraint, so episodes run in a pool of
``appworld serve environment`` subprocesses. See that module for the details.

Task text is read from the locally-downloaded AppWorld corpus at seed time and
never copied into gym datasets: AppWorld's tasks/APIs are released under Apache
2.0 with the additional requirement that redistribution stay encrypted. Dataset
rows therefore carry only a task id.
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from openai.types.responses import FunctionToolParam
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.openai_utils import NeMoGymEasyInputMessage
from resources_servers.appworld.prompts import (
    EXECUTE_TOOL_DESCRIPTION,
    EXECUTE_TOOL_NAME,
    INSTRUCTION_TEMPLATE,
    SYSTEM_PROMPT,
)
from resources_servers.appworld.schemas import (
    AppWorldCloseRequest,
    AppWorldCloseResponse,
    AppWorldResourcesServerConfig,
    AppWorldScoring,
    AppWorldSeedSessionRequest,
    AppWorldSeedSessionResponse,
    AppWorldStepRequest,
    AppWorldStepResponse,
    AppWorldVerifyRequest,
    AppWorldVerifyResponse,
)
from resources_servers.appworld.setup_appworld import AppWorldInstall, ensure_appworld
from resources_servers.appworld.worker_pool import AppWorldWorker, AppWorldWorkerPool


logger = logging.getLogger(__name__)

# Reclaim a worker whose episode never called /close (agent crash, killed
# rollout). Without this a leaked episode would permanently shrink the pool.
EPISODE_TIMEOUT_SECS = 3600.0


def execute_tool_param() -> FunctionToolParam:
    """The single function tool an AppWorld agent gets: run Python, see output."""
    return FunctionToolParam(
        type="function",
        name=EXECUTE_TOOL_NAME,
        description=EXECUTE_TOOL_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Print anything you need to see.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        strict=True,
    )


class AppWorldResourcesServer(SimpleResourcesServer):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: AppWorldResourcesServerConfig

    # env_id -> live episode state (worker lease, task id, counters).
    envs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # env_id -> scoring computed at /close, consumed by /verify.
    scoring: Dict[str, AppWorldScoring] = Field(default_factory=dict)

    _pool: Optional[AppWorldWorkerPool] = PrivateAttr(default=None)
    _install: Optional[AppWorldInstall] = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        # Builds AppWorld's isolated venv, unpacks its encrypted bundles and
        # downloads the corpus if needed; all no-ops once the artifacts exist.
        self._install = ensure_appworld(self.config.appworld_root, self.config.appworld_venv)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/step")(self.step)
        app.post("/close")(self.close)
        return app

    # ------------------------------------------------------------------
    # worker pool
    # ------------------------------------------------------------------

    @property
    def install(self) -> AppWorldInstall:
        assert self._install is not None, "model_post_init did not run"
        return self._install

    @property
    def pool(self) -> AppWorldWorkerPool:
        """The worker pool, constructed on first use (never at import time)."""
        if self._pool is None:
            self._pool = AppWorldWorkerPool(
                num_workers=self.config.num_env_workers,
                port_start=self.config.worker_port_start,
                root=self.install.root,
                executable=self.install.executable,
                startup_timeout_secs=self.config.worker_startup_timeout_secs,
                request_timeout_secs=self.config.worker_request_timeout_secs,
            )
        return self._pool

    # ------------------------------------------------------------------
    # seed_session
    # ------------------------------------------------------------------

    async def seed_session(self, body: AppWorldSeedSessionRequest) -> AppWorldSeedSessionResponse:
        await self._reap_expired_episodes()

        env_id = uuid.uuid4().hex
        experiment_name = f"{self.config.experiment_name_prefix}_{env_id}"
        worker = await self.pool.acquire()
        try:
            task = await worker.call(
                "/initialize",
                {
                    "task_id": body.task_id,
                    "experiment_name": experiment_name,
                    "max_interactions": self.config.max_interactions,
                    "timeout_seconds": self.config.execution_timeout_secs,
                    "raise_on_unsafe_syntax": self.config.raise_on_unsafe_syntax,
                },
            )
        except Exception:
            await self.pool.release(worker)
            logger.exception("AppWorld seed failed for task_id=%s", body.task_id)
            raise

        self.envs[env_id] = {
            "worker": worker,
            "task_id": body.task_id,
            "experiment_name": experiment_name,
            "num_interactions": 0,
            "done": False,
            "broken": False,
            "started_at": time.monotonic(),
        }
        logger.info(
            "AppWorld seed env=%s task=%s worker=%d",
            env_id,
            body.task_id,
            worker.index,
        )
        return AppWorldSeedSessionResponse(
            env_id=env_id,
            task_id=body.task_id,
            obs=self._initial_obs(task),
            tools=[execute_tool_param()],
        )

    def _initial_obs(self, task: Dict[str, Any]) -> List[NeMoGymEasyInputMessage]:
        """System prompt + the supervisor/instruction user turn for this task."""
        supervisor = task.get("supervisor") or {}
        template = self.config.instruction_template or INSTRUCTION_TEMPLATE
        instruction = template.format(
            supervisor_first_name=supervisor.get("first_name", ""),
            supervisor_last_name=supervisor.get("last_name", ""),
            supervisor_email=supervisor.get("email", ""),
            supervisor_phone_number=supervisor.get("phone_number", ""),
            instruction=task.get("instruction", ""),
        )
        return [
            NeMoGymEasyInputMessage(role="system", content=self.config.system_prompt or SYSTEM_PROMPT),
            NeMoGymEasyInputMessage(role="user", content=instruction),
        ]

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    async def step(self, body: AppWorldStepRequest) -> AppWorldStepResponse:
        state = self.envs.get(body.env_id)
        if state is None:
            raise KeyError(f"Unknown env_id {body.env_id!r}")
        if state["done"]:
            return AppWorldStepResponse(
                output="This episode has already ended.",
                done=True,
                num_interactions=state["num_interactions"],
            )

        worker: AppWorldWorker = state["worker"]
        task_id = state["task_id"]
        try:
            output = await worker.call("/execute", {"task_id": task_id, "code": body.code})
            state["num_interactions"] += 1
            completed = bool(await worker.call("/task_completed", {"task_id": task_id}))
        except Exception as exc:  # noqa: BLE001 — a dead worker ends the episode, it must not 500
            logger.warning("AppWorld step failed env=%s task=%s: %r", body.env_id, task_id, exc)
            state["done"] = True
            state["broken"] = True
            return AppWorldStepResponse(
                output=f"Execution environment error: {type(exc).__name__}: {exc}",
                done=True,
                num_interactions=state["num_interactions"],
            )

        budget_spent = state["num_interactions"] >= self.config.max_interactions
        state["done"] = completed or budget_spent
        return AppWorldStepResponse(
            output=str(output),
            done=state["done"],
            num_interactions=state["num_interactions"],
        )

    # ------------------------------------------------------------------
    # close / verify
    # ------------------------------------------------------------------

    async def close(self, body: AppWorldCloseRequest) -> AppWorldCloseResponse:
        state = self.envs.pop(body.env_id, None)
        if state is None:
            # Already closed (or reaped): /close is idempotent by design, since
            # the agent calls it from a `finally`.
            return AppWorldCloseResponse(message="Unknown or already-closed env_id", success=False)
        scoring = await self._finish_episode(body.env_id, state)
        self.scoring[body.env_id] = scoring
        return AppWorldCloseResponse(message="Success", success=True)

    async def _finish_episode(self, env_id: str, state: Dict[str, Any]) -> AppWorldScoring:
        """Score the episode, close the task on its worker, and free the worker."""
        worker: AppWorldWorker = state["worker"]
        scoring = await self._score(state)
        try:
            await worker.call("/close", {"task_id": state["task_id"]})
        except Exception as exc:  # noqa: BLE001 — best effort; the next init resets the worker
            logger.warning("AppWorld worker close failed env=%s: %r", env_id, exc)
        self._cleanup_outputs(state["experiment_name"])
        await self.pool.release(worker)
        logger.info(
            "AppWorld close env=%s task=%s reward=%.3f success=%s tests=%d/%d interactions=%d",
            env_id,
            state["task_id"],
            scoring.reward,
            scoring.success,
            scoring.num_passed,
            scoring.num_tests,
            scoring.num_interactions,
        )
        return scoring

    async def _score(self, state: Dict[str, Any]) -> AppWorldScoring:
        scoring = AppWorldScoring(num_interactions=state["num_interactions"])
        if state["broken"]:
            scoring.evaluation_error = "worker_failure"
            return scoring
        try:
            report = await state["worker"].call(
                "/evaluate",
                {"task_id": state["task_id"], "suppress_errors": True},
            )
        except Exception as exc:  # noqa: BLE001 — an unscorable episode is a 0, not a 500
            logger.warning("AppWorld evaluate failed task=%s: %r", state["task_id"], exc)
            scoring.evaluation_error = f"{type(exc).__name__}: {exc}"
            return scoring
        return self._scoring_from_report(report, scoring)

    def _scoring_from_report(self, report: Any, scoring: AppWorldScoring) -> AppWorldScoring:
        """Turn AppWorld's evaluation dict into a reward plus reporting fields."""
        if not isinstance(report, dict):
            scoring.evaluation_error = f"unexpected evaluation payload: {type(report).__name__}"
            return scoring
        passes = report.get("passes") or []
        failures = report.get("failures") or []
        scoring.success = bool(report.get("success"))
        scoring.num_tests = int(report.get("num_tests") or (len(passes) + len(failures)))
        scoring.num_passed = len(passes)
        scoring.partial_credit = (scoring.num_passed / scoring.num_tests) if scoring.num_tests else 0.0
        difficulty = report.get("difficulty")
        scoring.difficulty = int(difficulty) if isinstance(difficulty, (int, float)) else None
        scoring.failed_requirements = [str(failure.get("requirement", "")) for failure in failures]
        scoring.reward = scoring.partial_credit if self.config.dense_reward else float(scoring.success)
        return scoring

    def _cleanup_outputs(self, experiment_name: str) -> None:
        if not self.config.cleanup_experiment_outputs:
            return
        output_dir = Path(self.install.root) / "experiments" / "outputs" / experiment_name
        shutil.rmtree(output_dir, ignore_errors=True)

    async def _reap_expired_episodes(self) -> None:
        """Close episodes whose agent died without calling /close."""
        now = time.monotonic()
        expired = [env_id for env_id, state in self.envs.items() if now - state["started_at"] > EPISODE_TIMEOUT_SECS]
        for env_id in expired:
            state = self.envs.pop(env_id, None)
            if state is None:
                continue
            logger.warning("AppWorld reaping expired env=%s task=%s", env_id, state["task_id"])
            state["broken"] = True
            self.scoring[env_id] = await self._finish_episode(env_id, state)

    async def verify(self, body: AppWorldVerifyRequest) -> AppWorldVerifyResponse:
        scoring = self.scoring.pop(body.response.env_id, None)
        if scoring is None:
            logger.warning(
                "AppWorld verify with no cached scoring env=%s task=%s",
                body.response.env_id,
                body.response.task_id,
            )
            scoring = AppWorldScoring(evaluation_error="missing_scoring")
        return AppWorldVerifyResponse(
            **(body.model_dump() | scoring.as_response_fields()),
        )

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Add AppWorld's Scenario Goal Completion to the default metrics.

        TGC (task goal completion) is the per-row `success` that gym already
        averages. SGC additionally requires *every* variant of a scenario to pass:
        AppWorld ids are ``<scenario>_<variant>``, so variants are regrouped here
        by scenario and rollout index (variant k of a scenario is compared with
        variant k of its siblings, keeping repeats independent).
        """
        by_scenario: Dict[str, Dict[int, List[bool]]] = {}
        for task_rollouts in tasks:
            for rollout_index, rollout in enumerate(task_rollouts):
                task_id = rollout.get("task_id") or (rollout.get("response") or {}).get("task_id")
                if not isinstance(task_id, str) or "_" not in task_id:
                    continue
                scenario = task_id.rsplit("_", 1)[0]
                by_scenario.setdefault(scenario, {}).setdefault(rollout_index, []).append(bool(rollout.get("success")))
        groups = [successes for per_rollout in by_scenario.values() for successes in per_rollout.values()]
        if not groups:
            return {}
        return {"mean/scenario_goal_completion": sum(all(g) for g in groups) / len(groups)}


if __name__ == "__main__":
    AppWorldResourcesServer.run_webserver()
