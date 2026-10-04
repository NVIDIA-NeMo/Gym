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
import asyncio
import json
import logging
import os
import time
from typing import Any

from fastapi import HTTPException, Request
from mimoagent.environments.datasets import DatasetEnvironment
from mimoagent.environments.utils import make_dataset_env
from pydantic import ConfigDict

from nemo_gym import failure_kinds
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.mimo_rl_oss import opensource_code, terminal_bench  # noqa: F401  (register dataset types)
from resources_servers.mimo_rl_oss.general_agent.register import MCP_CONFIG_PATH, resolve_task_dir
from resources_servers.mimo_rl_oss.sandbox_env import GymSandboxEnvironment
from resources_servers.mimo_rl_oss.webdev import environment as webdev  # noqa: F401  (registers the dataset type)


LOG = logging.getLogger(__name__)


class MimoRLOSSConfig(BaseResourcesServerConfig):
    sandbox_provider: str | dict[str, Any]
    sandbox_spec: dict[str, Any] = {}
    # Extra keys of mimoagent's yaml `environment` block (reward_mode, judge_agent, ...).
    environment_kwargs: dict[str, Any] = {}
    exec_timeout: int = 600
    # Vision judge for webdev rows (any OpenAI-compatible endpoint), exported as MiMo's WEBDEV_EVAL_JUDGE_* env vars.
    webdev_judge_base_url: str | None = None
    webdev_judge_api_key: str | None = None
    webdev_judge_model: str | None = None
    # LLM judge for general_agent rubrics, exported as MiMo's GA_JUDGE_* env vars (Responses API).
    general_judge_base_url: str | None = None
    general_judge_api_key: str | None = None
    general_judge_model: str | None = None
    # "responses" for hosted APIs, "chat" for self-hosted vLLM judges.
    general_judge_api: str | None = None


class MimoRLOSSRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    instance: dict[str, Any]


class MimoRLOSSSeedResponse(BaseSeedSessionResponse):
    sandbox_descriptor: dict[str, Any]


class MimoRLOSSVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    instance: dict[str, Any]


class MimoRLOSSVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    instance_id: str
    test_output: str
    reward_extra_info: dict[str, Any]


def _failure(extra: dict[str, Any]) -> tuple[str, str] | None:
    """mimoagent's markers for rewards that are infrastructure failures, not policy failures."""
    if extra.get("transport_error"):
        return failure_kinds.SESSION_LOST, "sandbox transport failed during grading"
    category = extra.get("error_category")
    if not category:
        return None
    kind = failure_kinds.JUDGE_FAILED if category == "webdev_drop" else failure_kinds.VERIFIER_ERROR
    return kind, str(extra.get("reward_error") or extra.get("error") or category)


def _final_text(response: Any) -> str:
    texts = []
    for item in response.output or []:
        if getattr(item, "type", None) == "message":
            texts.extend(getattr(part, "text", "") or "" for part in item.content or [])
    return texts[-1] if texts else ""


class MimoRLOSSResourcesServer(SimpleResourcesServer):
    """Sets up and grades MiMo-V2.6-RL-oss tasks with mimoagent's own dataset environments.

    The task box belongs to this server: seed_session builds it (mimoagent's dataset setup) and
    hands harness_agent a descriptor; verify runs mimoagent's grader in the same box and closes it.
    """

    ray_enabled = False
    config: MimoRLOSSConfig

    def model_post_init(self, context: Any, /) -> None:
        # session id -> (task env, creation time). Sessions whose harness never reaches verify are evicted.
        self._envs: dict[str, tuple[DatasetEnvironment, float]] = {}
        self._evictions: set[asyncio.Task] = set()
        for key, value in (
            ("WEBDEV_EVAL_JUDGE_BASE_URL", self.config.webdev_judge_base_url),
            ("WEBDEV_EVAL_JUDGE_API_KEY", self.config.webdev_judge_api_key),
            ("WEBDEV_EVAL_JUDGE_MODEL", self.config.webdev_judge_model),
            ("GA_JUDGE_URL", self.config.general_judge_base_url),
            ("GA_JUDGE_KEY", self.config.general_judge_api_key),
            ("GA_JUDGE_MODEL", self.config.general_judge_model),
            ("GA_JUDGE_API", self.config.general_judge_api),
        ):
            if value:
                os.environ[key] = value
        if self.config.general_judge_base_url and not self.config.general_judge_api_key:
            LOG.warning("general_judge_api_key is empty, general_agent rubric grading will be masked")

    def _make_env(self, instance: dict[str, Any]) -> DatasetEnvironment:
        global_config = get_global_config_dict()
        spec = dict(self.config.sandbox_spec)
        spec["metadata"] = {
            **resolve_provider_metadata(self.config.sandbox_provider, global_config),
            **spec.get("metadata", {}),
        }
        return make_dataset_env(
            instance,
            environment_class=GymSandboxEnvironment,
            provider=resolve_provider_config(self.config.sandbox_provider, global_config),
            spec=spec,
            timeout=self.config.exec_timeout,
            **self.config.environment_kwargs,
        )

    def _setup(self, instance: dict[str, Any]) -> tuple[DatasetEnvironment, dict[str, Any]]:
        env = self._make_env(resolve_task_dir(instance))
        try:
            env.setup_environment()
            servers = getattr(env.env, "mcp_servers", None)
            if servers:
                mcp = {
                    "servers": servers,
                    "bridge_python": env.env.mcp_bridge_python,
                    "bridge_script": env.env.mcp_bridge_script,
                }
                env.copy_text_to(json.dumps(mcp), MCP_CONFIG_PATH)
            return env, env.env.descriptor()
        except BaseException:
            env.cleanup()
            raise

    async def seed_session(self, request: Request, body: MimoRLOSSRequest) -> MimoRLOSSSeedResponse:
        env, descriptor = await asyncio.to_thread(self._setup, body.instance)
        now = time.monotonic()
        ttl = float(self.config.sandbox_spec.get("ttl_s") or 14400)
        for key, (stale, created) in list(self._envs.items()):
            if now - created > ttl:
                del self._envs[key]
                task = asyncio.create_task(asyncio.to_thread(stale.cleanup))
                self._evictions.add(task)
                task.add_done_callback(self._evictions.discard)
        self._envs[str(request.session[SESSION_ID_KEY])] = (env, now)
        return MimoRLOSSSeedResponse(sandbox_descriptor={**descriptor, "workdir": env.repo_path})

    async def verify(self, request: Request, body: MimoRLOSSVerifyRequest) -> MimoRLOSSVerifyResponse:
        entry = self._envs.pop(str(request.session[SESSION_ID_KEY]), None)
        if entry is None:
            raise HTTPException(status_code=400, detail="mimo_rl_oss session is not active")
        env = entry[0]
        try:
            env.attach_rollout(task=body.instance.get("problem_statement", ""), result=_final_text(body.response))
            reward, test_output, extra = await asyncio.to_thread(env.calculate_reward)
        finally:
            await asyncio.to_thread(env.cleanup)
        failure = _failure(extra)
        return MimoRLOSSVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            mask_sample=failure is not None,
            failure_kind=failure[0] if failure else None,
            failure_reason=failure[1] if failure else None,
            instance_id=env.instance_id,
            test_output=test_output[-5000:],
            reward_extra_info={k: v for k, v in extra.items() if k != "last_poc_b64"},
        )


if __name__ == "__main__":
    MimoRLOSSResourcesServer.run_webserver()
