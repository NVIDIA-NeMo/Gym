# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AutomationBench resources server.

Each session holds one task's simulated SaaS world, built by ``/seed_session`` from the task's ``initial_state`` the
way upstream's verifiers environment builds it. The agent calls the api toolset (``api_search``, ``api_fetch``,
``base64_encode``) through ``POST /{tool_name}``; each call reads or changes that world. ``/verify`` scores the final
world with the rubric of ``benchmarks/automationbench``: the Artificial Analysis headline metric (any broken guardrail
zeroes the task) by default, with upstream's ``partial_credit`` and the guardrail and objective counts alongside.

The tools and the rubric are the ones ``verifiers_agent`` runs for this benchmark, imported from
``automationbench_env``; only the tool loop moves to the agent.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Body, FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, PrivateAttr, ValidationError

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import VERIFIER_ERROR, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.metrics import compute_pass_majority_metrics, compute_subset_metrics, highest_k_metrics
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedRejectedError,
    ResourcesSessionLostError,
    SimpleResourcesServer,
    lost_session_verification,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture
from resources_servers.automationbench.task_data import TaskData, TaskInfo


LOG = logging.getLogger(__name__)

# The scores /verify reports, in the order the benchmark's rubric computes them: task_completed_correctly reads the
# partial_credit that partial_credit caches on the state.
SCORE_NAMES = (
    "partial_credit",
    "aa_headline",
    "task_completed_correctly",
    "guardrails_violated",
    "guardrails_total",
    "objectives_passed",
    "objectives_total",
)
# A row without tools is one prepared before this server existed: the model could not touch the world, so its zero
# would be no grade of the model. The seed refuses it before any model call when the seed carries the task's params;
# /verify masks it for a caller whose seed does not.
NO_TOOLS_REASON = (
    "The row's responses_create_params carries no tools, so the model could not act on the task's world. Rows "
    "prepared before the automationbench resources server lack them: re-run `gym eval prepare --benchmark "
    "automationbench`."
)


class AutomationBenchResourcesServerConfig(BaseResourcesServerConfig):
    # Which score is the reward. Both are always reported, as the verifiers recipe reports them.
    reward_fn: Literal["aa_headline", "partial_credit"] = "aa_headline"
    # Adds every assertion's verdict to the verify response (`assertion_results`).
    export_assertion_results: bool = False


class AutomationBenchVerifyRequest(BaseVerifyRequest):
    task_id: str


class AutomationBenchVerifyResponse(AutomationBenchVerifyRequest, BaseVerifyResponse):
    task_name: Optional[str] = None
    # The task name's prefix (finance, hr, marketing, operations, sales, support); groups the metrics.
    domain: Optional[str] = None
    partial_credit: float = 0.0
    aa_headline: float = 0.0
    task_completed_correctly: float = 0.0
    guardrails_violated: float = 0.0
    guardrails_total: float = 0.0
    objectives_passed: float = 0.0
    objectives_total: float = 0.0
    assertion_results: Optional[List[Dict[str, Any]]] = Field(default=None, exclude_if=lambda value: value is None)


@cache
def tool_env() -> Any:
    """Gym's AutomationBench verifiers environment, used only for its tools and how it calls them.

    ``load_environment`` would also build the 600-task dataset; the tools need none of it, so the environment gets a
    one-row placeholder.
    """
    import verifiers as vf
    from automationbench_env import AutomationBenchEnv
    from datasets import Dataset

    placeholder = Dataset.from_list([{"prompt": [{"role": "user", "content": ""}], "answer": "", "info": "{}"}])
    return AutomationBenchEnv(dataset=placeholder, rubric=vf.Rubric(), toolset="api")


def new_world_state(info: Dict[str, Any]) -> Dict[str, Any]:
    """The rubric's state for a fresh episode, built as upstream's ``AutomationBenchEnv.setup_state`` builds it."""
    from automationbench.runner import compute_allowed_services, strip_none_values
    from automationbench.schema.world import WorldState

    info = copy.deepcopy(info)
    initial_state = strip_none_values(info.get("initial_state", {}))
    info["assertions"] = [strip_none_values(assertion) for assertion in info.get("assertions", [])]
    world = WorldState(**initial_state)
    world.meta.allowed_services = compute_allowed_services(
        initial_state, info.get("assertions", []), info.get("zapier_tools", [])
    )
    return {"info": info, "world": world, "initial_state": copy.deepcopy(initial_state)}


def run_tool(tool_name: str, arguments: Any, state: Dict[str, Any]) -> str:
    """One tool call as verifiers' StatefulToolEnv makes it: the tool's text, or the error's text for the model."""
    env = tool_env()
    if not isinstance(arguments, dict):
        return f"Expected tool arguments to be a dict, got {type(arguments).__name__}: {arguments}"
    arguments = env.update_tool_args(tool_name, arguments, [], state)
    try:
        return str(env.tool_map[tool_name](**arguments))
    except Exception as error:
        return f"{error}"


def score(state: Dict[str, Any], *, export_assertion_results: bool) -> Dict[str, Any]:
    """The rubric's scores of the final world, and the per-assertion verdicts when asked for."""
    import automationbench_env
    from automationbench.rubric import partial_credit, task_completed_correctly

    functions = {"partial_credit": partial_credit, "task_completed_correctly": task_completed_correctly}
    scores = {name: float((functions.get(name) or getattr(automationbench_env, name))(state)) for name in SCORE_NAMES}
    if export_assertion_results:
        scores["assertion_results"] = automationbench_env.assertion_results(state)
    return scores


class AutomationBenchSeedRequest(TaskData, BaseSeedSessionRequest):
    """A seed's ``task_data``: one AutomationBench row, with the task's ``responses_create_params`` that a typed seed
    carries beside it."""

    # Read only to refuse a row without tools; the agent, not this server, sends them to the model.
    responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None


@dataclass
class _Episode:
    state: Dict[str, Any]


class AutomationBenchResourcesServer(SimpleResourcesServer):
    config: AutomationBenchResourcesServerConfig
    seed_request_model = AutomationBenchSeedRequest

    _episodes: Dict[str, _Episode] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # Importing AutomationBench takes minutes on a cold filesystem; do it before serving, not in a tool call.
        tool_env()
        # api_search writes its schema index into site-packages on first use, and not atomically. Write it here: the
        # main process sets up the app before uvicorn starts any worker, so workers only read it.
        run_tool("api_search", {"query": "gmail"}, new_world_state({"initial_state": {}, "assertions": []}))
        # Last, so the routes registered above win over a tool of the same name.
        app.post("/{tool_name}")(self.call_tool)
        return app

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._episodes

    async def initialize_session_state(self, session_id: str, body: AutomationBenchSeedRequest) -> None:
        # A seed without the params (a caller that does not send them) cannot be checked, and is served.
        if body.responses_create_params is not None and not body.responses_create_params.tools:
            raise ResourcesSeedRejectedError(NO_TOOLS_REASON)
        try:
            state = new_world_state(body.info.model_dump())
        except ValidationError as error:
            # No retry can build this world: refuse the seed (422) instead of failing it with a 500 the Environment
            # Server would dispatch again.
            raise ResourcesSeedRejectedError(f"The task's initial_state is not a valid AutomationBench world: {error}")
        self._episodes[session_id] = _Episode(state)

    async def release_session_state(self, session_id: str) -> None:
        self._episodes.pop(session_id, None)

    async def call_tool(self, request: Request, tool_name: str, arguments: Any = Body(None)) -> PlainTextResponse:
        """Run one tool on the session's world.

        The model sees the tool's text as it is, and an unknown tool, bad arguments or a failing call as error text,
        as it would under verifiers.
        """
        session_id = request.session[SESSION_ID_KEY]
        async with self.session_lock(session_id):
            episode = self._episodes.get(session_id)
            if episode is None:
                raise ResourcesSessionLostError("No seeded AutomationBench session for this tool call")
            return PlainTextResponse(run_tool(tool_name, arguments, episode.state))

    async def verify(self, request: Request, body: AutomationBenchVerifyRequest) -> AutomationBenchVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        async with self.verifying(session_id) as verification:
            if verification.is_lost:
                lost = lost_session_verification("No seeded AutomationBench session for this verify")
                return AutomationBenchVerifyResponse(**(body.model_dump() | lost))
            if verification.is_live:
                state = self._episodes[session_id].state
                task_name = state["info"]["task_name"]
                if not body.responses_create_params.tools:
                    # Only a caller whose seed carried no params gets here; the seed refuses the rest.
                    LOG.error("AutomationBench task %s: %s", body.task_id, NO_TOOLS_REASON)
                    scores = {"failure_reason": NO_TOOLS_REASON}
                else:
                    try:
                        scores = score(state, export_assertion_results=self.config.export_assertion_results)
                    except Exception as error:
                        # verifiers would score the raising function 0.0; that zero is no grade of the answer.
                        LOG.exception("Scoring AutomationBench task %s failed", body.task_id)
                        scores = {"failure_reason": f"AutomationBench rubric raised: {type(error).__name__}: {error}"}
                verification.retain({"task_name": task_name, **scores})
            result = dict(verification.retained)
        task_name = result.pop("task_name")
        fields = {"task_name": task_name, "domain": task_name.split(".", 1)[0]}
        if "failure_reason" in result:
            return AutomationBenchVerifyResponse(
                **body.model_dump(),
                **fields,
                reward=0.0,
                mask_sample=True,
                failure_kind=VERIFIER_ERROR,
                failure_reason=result["failure_reason"],
            )
        return AutomationBenchVerifyResponse(
            **body.model_dump(), **fields, **result, reward=result[self.config.reward_fn]
        )

    @staticmethod
    def _scores(rollout: Dict[str, Any]) -> Dict[str, float]:
        return {
            "aa_headline": rollout.get("aa_headline", 0.0),
            "partial_credit": rollout.get("partial_credit", 0.0),
            "task_completed_correctly": rollout.get("task_completed_correctly", 0.0),
        }

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        metrics, _, _, _ = compute_pass_majority_metrics(tasks, score_fn=self._scores)
        metrics.pop("per_sample_aggregate", None)
        metrics.update(
            compute_subset_metrics([self._with_domain(t) for t in tasks], subset_key="domain", score_fn=self._scores)
        )
        return metrics

    @staticmethod
    def _with_domain(rollouts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Label a task that no /verify reached by its task_id, so its counted zeros stay in its domain.

        A rollout counted as zero carries the row's dataset fields, task_id among them, but not the domain /verify
        computes. A domain any repeat reports wins.
        """
        if any(rollout.get("domain") for rollout in rollouts):
            return rollouts
        return [
            {**rollout, "domain": rollout["task_id"].split(".", 1)[0]}
            if isinstance(rollout.get("task_id"), str) and "." in rollout["task_id"]
            else rollout
            for rollout in rollouts
        ]

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key = {
            name: agent_metrics[name] for name in ("mean/input_tokens", "mean/output_tokens") if name in agent_metrics
        }
        key.update(
            highest_k_metrics(
                agent_metrics,
                "pass@1[avg-of-{k}]",
                score_names=["aa_headline", "partial_credit", "task_completed_correctly"],
            )
        )
        return key


class _FixtureToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class _FixtureRequest(BaseModel):
    info: TaskInfo
    tool_calls: List[_FixtureToolCall] = Field(default_factory=list)


class _FixtureVerifier:
    """Exercise the production tools and rubric on a small world without model or dataset downloads."""

    def verify(self, body: _FixtureRequest) -> float:
        state = new_world_state(body.info.model_dump())
        for call in body.tool_calls:
            run_tool(call.name, call.arguments, state)
        return score(state, export_assertion_results=False)["aa_headline"]


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_FixtureVerifier,
    request_model=_FixtureRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    AutomationBenchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AutomationBenchResourcesServer.run_webserver()  # noqa: F401
