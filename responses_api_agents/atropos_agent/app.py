# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)
from responses_api_agents.atropos_agent.gym_server_bridge import GymServerManager, GymTokenInfo


logger = logging.getLogger(__name__)


class AtroposAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    model_name: str = Field(default="", description="Model name served by Gym's vLLM")

    atropos_env_module: str = Field(default="", description="e.g. 'environments.gsm8k_server'")
    atropos_env_class: str = Field(default="", description="e.g. 'GSM8kEnv'")
    atropos_env_config: Dict[str, Any] = Field(default_factory=dict, description="Override BaseEnvConfig fields")
    atropos_group_size: int = Field(default=8, description="Group size for envs that need cohort buffering")

    reward_function: str = Field(default="", description="Import path (module:Class) for standalone reward fallback")
    reward_function_kwargs: Dict[str, Any] = Field(default_factory=dict)

    max_tokens: int = Field(default=8192)
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)
    max_concurrent: int = Field(default=-1)


class AtroposAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_idx: int = 0
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )
    verifier_metadata: Dict[str, Any] = Field(default_factory=dict)


class AtroposNeMoGymResponse(NeMoGymResponse):
    model_config = ConfigDict(extra="allow")
    output: List[Dict[str, Any]] = Field(default_factory=list)
    reward: float = 0.0
    parallel_tool_calls: bool = False
    tool_choice: str = "none"
    tools: list = Field(default_factory=list)


class AtroposVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    response: AtroposNeMoGymResponse
    reward: float


class AtroposAgent(SimpleResponsesAPIAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: AtroposAgentConfig

    _env: Optional[Any] = None
    _env_ready: bool = False
    _env_has_collect_trajectory: Optional[bool] = None
    _server_manager: Optional[GymServerManager] = None
    _sem: Optional[asyncio.Semaphore] = None
    _reward_fn: Optional[Any] = None

    _cohort_lock: asyncio.Lock = None
    _cohort_buffers: Dict[str, List[Tuple[Any, asyncio.Future]]] = None

    def model_post_init(self, __context: Any) -> None:
        if self.config.max_concurrent > 0:
            self._sem = asyncio.Semaphore(self.config.max_concurrent)
        if self.config.reward_function:
            self._reward_fn = self._load_import(self.config.reward_function, self.config.reward_function_kwargs)
        self._cohort_lock = asyncio.Lock()
        self._cohort_buffers = {}

    @staticmethod
    def _load_import(path: str, kwargs: dict = None) -> Any:
        sep = ":" if ":" in path else "."
        mod_path, cls_name = path.rsplit(sep, 1)
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        return cls(**(kwargs or {}))

    def _get_server_manager(self) -> GymServerManager:
        if self._server_manager is None:
            cfg = get_first_server_config_dict(self.server_client.global_config_dict, self.config.model_server.name)
            url = f"http://{cfg.host}:{cfg.port}/v1"
            self._server_manager = GymServerManager(url, self.config.model_name)
        return self._server_manager

    async def _get_env(self) -> Any:
        if self._env is not None:
            return self._env
        if not self.config.atropos_env_module:
            return None

        mod = importlib.import_module(self.config.atropos_env_module)
        env_cls = getattr(mod, self.config.atropos_env_class)

        env_config_cls = getattr(env_cls, "env_config_cls", None)
        if env_config_cls is None:
            from atroposlib.envs.base import BaseEnvConfig

            env_config_cls = BaseEnvConfig

        config_kwargs = {
            "group_size": self.config.atropos_group_size,
            "use_wandb": False,
            "ensure_scores_are_not_same": False,
            "tokenizer_name": self.config.model_name,
            **self.config.atropos_env_config,
        }
        env_config = env_config_cls(**config_kwargs)

        from atroposlib.envs.server_handling.server_manager import APIServerConfig

        dummy_config = APIServerConfig(
            model_name=self.config.model_name, base_url="http://placeholder", api_key="gym", num_requests_for_eval=1
        )
        env = env_cls(config=env_config, server_configs=[dummy_config], slurm=False, testing=True)
        env.server = self._get_server_manager()

        if not self._env_ready:
            await env.setup()
            self._env_ready = True

        try:
            method = getattr(env.__class__, "collect_trajectory", None)
            from atroposlib.envs.base import BaseEnv

            self._env_has_collect_trajectory = method is not None and method is not BaseEnv.collect_trajectory
        except Exception:
            self._env_has_collect_trajectory = False

        self._env = env
        return env

    @staticmethod
    def _prompt_key(body: AtroposAgentRunRequest) -> str:
        input_msgs = body.responses_create_params.input or []
        key_data = json.dumps(
            [
                {"role": m.get("role", ""), "content": m.get("content", "")} if isinstance(m, dict) else str(m)
                for m in input_msgs
            ],
            sort_keys=True,
        )
        return hashlib.md5(key_data.encode()).hexdigest()

    def _build_item(self, body: AtroposAgentRunRequest) -> Dict[str, Any]:
        vm = body.verifier_metadata
        if "atropos_item" in vm:
            return vm["atropos_item"]
        question = ""
        for msg in body.responses_create_params.input or []:
            r = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
            c = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if r == "user":
                question = c
        item: Dict[str, Any] = {"question": question, "answer": vm.get("answer", vm.get("expected_answer", ""))}
        for k, v in vm.items():
            if k not in ("answer", "expected_answer", "atropos_item"):
                item[k] = v
        return item

    def _build_output(
        self, content: str, token_info: Optional[GymTokenInfo], input_messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for msg in input_messages:
            output.append(
                NeMoGymEasyInputMessage(role=msg.get("role", "user"), content=msg.get("content", "")).model_dump()
            )
        if token_info is not None:
            output.append(
                NeMoGymResponseOutputMessageForTraining(
                    id="msg_atropos_0",
                    content=[NeMoGymResponseOutputText(text=content, annotations=[])],
                    prompt_token_ids=token_info.prompt_token_ids,
                    generation_token_ids=token_info.generation_token_ids,
                    generation_log_probs=token_info.generation_log_probs,
                ).model_dump()
            )
        else:
            output.append(
                NeMoGymResponseOutputMessage(
                    id="msg_atropos_0",
                    content=[NeMoGymResponseOutputText(text=content, annotations=[])],
                ).model_dump()
            )
        return output

    def _build_output_from_scored_item(
        self, scored_item: Dict[str, Any], server_manager: GymServerManager
    ) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        messages = scored_item.get("messages") or []
        server = server_manager.get_server()
        token_infos = server.get_token_infos()
        nodes = server.current_nodes

        all_prompt_ids: List[int] = []
        all_gen_ids: List[int] = []
        all_gen_logprobs: List[float] = []
        for ti in token_infos:
            all_prompt_ids.extend(ti.prompt_token_ids)
            all_gen_ids.extend(ti.generation_token_ids)
            all_gen_logprobs.extend(ti.generation_log_probs)

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("system", "user"):
                output.append(NeMoGymEasyInputMessage(role=role, content=content).model_dump())
            elif role == "assistant":
                if all_gen_ids:
                    output.append(
                        NeMoGymResponseOutputMessageForTraining(
                            id=f"msg_atropos_{len(output)}",
                            content=[NeMoGymResponseOutputText(text=content, annotations=[])],
                            prompt_token_ids=all_prompt_ids,
                            generation_token_ids=all_gen_ids,
                            generation_log_probs=all_gen_logprobs,
                        ).model_dump()
                    )
                    all_prompt_ids, all_gen_ids, all_gen_logprobs = [], [], []
                else:
                    output.append(
                        NeMoGymResponseOutputMessage(
                            id=f"msg_atropos_{len(output)}",
                            content=[NeMoGymResponseOutputText(text=content, annotations=[])],
                        ).model_dump()
                    )

        if not output and all_gen_ids:
            node_text = nodes[0].full_text if nodes else ""
            output.append(
                NeMoGymResponseOutputMessageForTraining(
                    id="msg_atropos_0",
                    content=[NeMoGymResponseOutputText(text=node_text, annotations=[])],
                    prompt_token_ids=all_prompt_ids,
                    generation_token_ids=all_gen_ids,
                    generation_log_probs=all_gen_logprobs,
                ).model_dump()
            )
        return output

    async def _run_single(self, env: Any, body: AtroposAgentRunRequest) -> AtroposNeMoGymResponse:
        item = self._build_item(body)
        server_manager = self._get_server_manager()
        server_manager.get_server().clear_token_infos()

        try:
            scored_item, _ = await env.collect_trajectory(item)
        except Exception as e:
            logger.warning(f"collect_trajectory failed with provided item ({e}), retrying with env.get_next_item()")
            fresh_item = await env.get_next_item()
            scored_item, _ = await env.collect_trajectory(fresh_item)

        if scored_item is not None:
            reward = float(scored_item.get("scores", 0.0))
            output = self._build_output_from_scored_item(scored_item, server_manager)
        else:
            reward = 0.0
            server = server_manager.get_server()
            token_infos = server.get_token_infos()
            nodes = server.current_nodes
            if token_infos:
                ti = token_infos[0]
                content = nodes[0].full_text if nodes else ""
                output = self._build_output(content, ti, [])
            else:
                output = []

        return AtroposNeMoGymResponse(
            id=f"atropos-{body.task_idx}",
            created_at=0,
            model=self.config.model_name,
            object="response",
            output=output,
            reward=reward,
        )

    def _cohort_result(
        self, body: AtroposAgentRunRequest, score: float, token_info: Optional[GymTokenInfo], content: str = ""
    ) -> Tuple[float, List[Dict[str, Any]]]:
        reward = float(score) if score == score else 0.0  # handle NaN
        chat_msgs = [m if isinstance(m, dict) else m.model_dump() for m in (body.responses_create_params.input or [])]
        return reward, self._build_output(content, token_info, chat_msgs)

    async def _run_cohort(self, env: Any, body: AtroposAgentRunRequest) -> AtroposNeMoGymResponse:
        prompt_key = self._prompt_key(body)
        future: asyncio.Future[Tuple[float, List[Dict[str, Any]]]] = asyncio.get_running_loop().create_future()
        group_size = self.config.atropos_group_size

        async with self._cohort_lock:
            if prompt_key not in self._cohort_buffers:
                self._cohort_buffers[prompt_key] = []
            self._cohort_buffers[prompt_key].append((body, future))
            buf = self._cohort_buffers[prompt_key]

            if len(buf) >= group_size:
                item = self._build_item(buf[0][0])
                server = self._get_server_manager().get_server()
                server.clear_token_infos()

                scored_data = None
                try:
                    scored_data, _ = await env.collect_trajectories(item)
                except Exception as e:
                    logger.warning(
                        f"Cohort collect_trajectories failed with provided item ({e}), retrying with env.get_next_item()"
                    )
                    try:
                        fresh_item = await env.get_next_item()
                        scored_data, _ = await env.collect_trajectories(fresh_item)
                    except Exception as e2:
                        logger.error(f"Cohort collect_trajectories failed: {e2}")

                token_infos = server.get_token_infos()
                nodes = server.current_nodes
                scores = (scored_data or {}).get("scores") or []
                messages = (scored_data or {}).get("messages") or []

                for i, (b, f) in enumerate(buf):
                    if f.done():
                        continue
                    ti = token_infos[i] if i < len(token_infos) else None
                    content = ""
                    if i < len(messages):
                        msg_list = messages[i]
                        for m in msg_list if isinstance(msg_list, (list, tuple)) else [msg_list]:
                            if isinstance(m, dict) and m.get("role") == "assistant":
                                content = m.get("content", "")
                                break
                    if not content and i < len(nodes):
                        content = nodes[i].full_text
                    score = scores[i] if i < len(scores) else 0.0
                    if not scores and self._reward_fn:
                        try:
                            r = self._reward_fn.compute([content], solution=b.verifier_metadata.get("answer", ""))
                            score = r[0] if r else 0.0
                        except Exception:
                            pass
                    f.set_result(self._cohort_result(b, score, ti, content))

                del self._cohort_buffers[prompt_key]

        try:
            reward, output = await asyncio.wait_for(future, timeout=300)
        except asyncio.TimeoutError:
            logger.error(f"Cohort timed out. Ensure num_repeats >= atropos_group_size ({group_size}).")
            reward, output = 0.0, []
        return AtroposNeMoGymResponse(
            id=f"atropos-{body.task_idx}",
            created_at=0,
            model=self.config.model_name,
            object="response",
            output=output,
            reward=reward,
        )

    async def _run_no_env(self, body: AtroposAgentRunRequest) -> AtroposNeMoGymResponse:
        server_manager = self._get_server_manager()
        chat_messages = [
            msg if isinstance(msg, dict) else msg.model_dump() for msg in (body.responses_create_params.input or [])
        ]
        temperature = getattr(body.responses_create_params, "temperature", None) or self.config.temperature
        top_p = getattr(body.responses_create_params, "top_p", None) or self.config.top_p

        async with server_manager.managed_server() as managed:
            completion = await managed.chat_completion(
                messages=chat_messages,
                n=1,
                max_tokens=self.config.max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        completion_text = completion.choices[0].message.content or ""
        vm = body.verifier_metadata
        answer = vm.get("answer", vm.get("expected_answer", ""))

        reward = 0.0
        if self._reward_fn:
            try:
                scores = self._reward_fn.compute([completion_text], solution=answer)
                reward = float(scores[0]) if scores else 0.0
            except Exception as e:
                logger.warning(f"Reward function failed: {e}")

        token_infos = server_manager.get_server().get_token_infos()
        token_info = token_infos[0] if token_infos else None
        return AtroposNeMoGymResponse(
            id=f"atropos-{body.task_idx}",
            created_at=0,
            model=self.config.model_name,
            object="response",
            output=self._build_output(completion_text, token_info, chat_messages),
            reward=reward,
        )

    async def responses(
        self,
        request: Request,
        response: Response,
        body: AtroposAgentRunRequest = Body(),
    ) -> AtroposNeMoGymResponse:
        try:
            env = await self._get_env()
            if env is None:
                return await self._run_no_env(body)
            elif self._env_has_collect_trajectory:
                return await self._run_single(env, body)
            else:
                return await self._run_cohort(env, body)
        except Exception as e:
            logger.error(f"Exception in responses(): {type(e).__name__}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

    async def run(
        self,
        request: Request,
        response: Response,
        body: AtroposAgentRunRequest = Body(),
    ) -> AtroposVerifyResponse:
        async def _inner() -> AtroposVerifyResponse:
            resp = await self.responses(request, response, body)
            return AtroposVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=resp,
                reward=resp.reward,
            )

        if self._sem is not None:
            async with self._sem:
                return await _inner()
        return await _inner()


if __name__ == "__main__":
    AtroposAgent.run_webserver()
