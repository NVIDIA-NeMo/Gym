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
"""Schemas shared by the AppWorld resources server and its agent harness.

AppWorld is a code-as-action environment: the agent writes Python against ~457
APIs across 9 simulated apps, and is scored by database-state tests rather than
by its code. Each episode is one task held by the resources server, following
the aviary/toolsandbox env pattern: ``seed_session -> obs + tools``,
``/step(code) -> output, done``, ``/close`` (which scores), then a pure
``/verify`` that returns the cached reward.
"""

from typing import Any, ClassVar, Dict, List, Optional

from openai.types.responses import FunctionToolParam
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseOutputItem,
)


class AppWorldResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the AppWorld resources server."""

    # The reward comes from database-state tests run against a live episode, so a
    # finished trajectory cannot be rescored on its own.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    # Root holding ``data/`` (the downloaded corpus) and ``experiments/outputs/``
    # (per-rollout scratch DBs). None => ``$APPWORLD_ROOT``, else the gitignored
    # ``resources_servers/appworld/.appworld_root``.
    appworld_root: Optional[str] = None
    # Isolated venv the ``appworld`` package is installed into — it cannot share
    # this server's venv (conflicting python-multipart pins) and does not need
    # to, since episodes run in subprocesses. None => ``$APPWORLD_VENV``, else
    # the gitignored ``resources_servers/appworld/.appworld_venv``.
    appworld_venv: Optional[str] = None

    # Size of the ``appworld serve environment`` worker pool. One worker hosts at
    # most one live episode (see worker_pool.py), so this is the hard ceiling on
    # concurrent rollouts — additional /seed_session calls queue for a free
    # worker. Each worker is ~1 idle process; task switching costs ~0.1-0.4s.
    #
    # Deliberately *not* called ``num_workers``: that name is already gym's
    # uvicorn worker count on BaseServerConfig, and setting it would fork this
    # server into N processes each holding its own pool and episode table.
    num_env_workers: int = 8
    # First port tried for worker 0; each worker falls back to an OS-chosen
    # ephemeral port if its preferred one is taken. Kept above gym's own
    # allocation range (10001-20000) and below the Linux ephemeral range, so
    # worker ports cannot collide with ports gym has reserved for its servers.
    worker_port_start: int = 21000
    worker_startup_timeout_secs: float = 300.0
    worker_request_timeout_secs: float = 900.0

    # Cap on ``execute`` calls per episode. Also passed to AppWorld itself, which
    # enforces it independently.
    max_interactions: int = 40
    # Wall-clock cap AppWorld applies to a single code execution.
    execution_timeout_secs: int = 100
    # AppWorld's static-analysis guard on destructive stdlib usage. Keep on:
    # the agent's Python runs in the worker process, not a container.
    raise_on_unsafe_syntax: bool = True

    # Each episode gets its own AppWorld experiment name (``<prefix>_<env_id>``)
    # because AppWorld rmtree's ``experiments/outputs/<name>/tasks/<task_id>/`` on
    # init — two concurrent rollouts of the same task (num_repeats > 1) sharing a
    # name would wipe each other's databases.
    experiment_name_prefix: str = "nemo_gym"
    # Delete that per-episode output directory on /close (~68 KB per rollout).
    cleanup_experiment_outputs: bool = True

    # False => reward is AppWorld's binary Task Goal Completion (all tests pass).
    # True  => fraction of passing tests, as a shaped signal for RL.
    dense_reward: bool = False

    # Prompt overrides; None => the defaults in prompts.py.
    system_prompt: Optional[str] = None
    instruction_template: Optional[str] = None


class AppWorldSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")

    # AppWorld task id, e.g. "82e2fac_1". Rows carry only this id — task text is
    # fetched from the local AppWorld corpus at seed time, never redistributed
    # through gym datasets (see README § Licensing).
    task_id: str


class AppWorldSeedSessionResponse(BaseSeedSessionResponse):
    env_id: str
    task_id: str
    # System prompt + the supervisor/instruction turn, prepended to the rollout.
    obs: List[NeMoGymEasyInputMessage]
    # The single code-execution tool the agent may call.
    tools: List[FunctionToolParam]


class AppWorldStepRequest(BaseModel):
    env_id: str
    code: str


class AppWorldStepResponse(BaseModel):
    # Whatever the code printed, or the traceback it raised.
    output: str
    # True once the agent called ``apis.supervisor.complete_task()`` or the
    # interaction budget ran out.
    done: bool
    # Per-step reward is always 0.0; scoring happens once, in /close.
    reward: float = 0.0
    num_interactions: int


class AppWorldCloseRequest(BaseModel):
    env_id: str


class AppWorldCloseResponse(BaseModel):
    message: str
    success: bool


class AppWorldNeMoGymResponse(NeMoGymResponse):
    env_id: str
    task_id: str
    output: List[NeMoGymResponseOutputItem]


class AppWorldVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    response: AppWorldNeMoGymResponse


# MRO puts AppWorldVerifyRequest.response ahead of BaseVerifyResponse.response.
class AppWorldVerifyResponse(AppWorldVerifyRequest, BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    # AppWorld's Task Goal Completion: did every database-state test pass?
    success: bool = False
    num_tests: int = 0
    num_passed: int = 0
    # num_passed / num_tests — the shaped variant of `success`, always reported
    # so it can be profiled even when `dense_reward` is off.
    partial_credit: float = 0.0
    # Upstream's per-task difficulty indicator (1-3), from the evaluation report.
    difficulty: Optional[int] = None
    num_interactions: int = 0
    # Set when scoring itself blew up, in which case reward is 0.0.
    evaluation_error: Optional[str] = None
    # Requirement strings of the failing tests, for error analysis.
    failed_requirements: List[str] = Field(default_factory=list)


class AppWorldScoring(BaseModel):
    """Scoring cached at /close time and consumed by /verify."""

    reward: float = 0.0
    success: bool = False
    num_tests: int = 0
    num_passed: int = 0
    partial_credit: float = 0.0
    difficulty: Optional[int] = None
    num_interactions: int = 0
    evaluation_error: Optional[str] = None
    failed_requirements: List[str] = Field(default_factory=list)

    def as_response_fields(self) -> Dict[str, Any]:
        return self.model_dump()
