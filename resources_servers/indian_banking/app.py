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
"""Indian retail-banking customer-support environment (multi-turn, tool-using).

A GymnasiumServer for ``gymnasium_agent``: ``/reset`` seeds a per-episode copy of the
synthetic customer DB; ``/step`` executes tool calls in-process or, for a text turn, asks
the LLM user simulator for the customer's next message; terminal steps are scored by
``core.reward`` (+ the NL-assertion judge).
"""

import json
import logging
from typing import Any, Optional

import aiohttp
from fastapi import FastAPI
from pydantic import Field

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.gymnasium import GymnasiumServer, extract_text
from resources_servers.indian_banking.core import engine, judge, reward, user_sim
from resources_servers.indian_banking.core.reward import TASK_KEY, WORLD_KEY


logger = logging.getLogger(__name__)

# top_k=1 keeps tool responses short; enforced at dispatch, not by the policy.
_FORCED_TOOL_ARGS: dict[str, dict[str, Any]] = {
    "search_knowledge_base": {"top_k": 1},
}

# Never add or drop a key here, or downstream aggregation ends up with a ragged column.
_REWARD_KEYS: tuple[str, ...] = (
    "score",
    "strict",
    "action_frac",
    "action",
    "db",
    "communicate",
    "communicate_frac",
    "dense",
    "judge",
    "write_purity",
    "name_frac",
    "bad_writes",
    "seq_frac",
    "repeat_penalty",
    "longest_call_streak",
)


def _empty_reward_info() -> dict[str, float]:
    return {k: 0.0 for k in _REWARD_KEYS}


def _coerce_args(args: dict) -> dict:
    """Decode JSON-encoded strings in tool arguments (e.g. '["SB1"]' -> ["SB1"])."""
    out = {}
    for key, value in args.items():
        if isinstance(value, str):
            stripped = value.strip()
            if stripped[:1] in "[{":
                try:
                    parsed = json.loads(stripped)
                except (TypeError, ValueError):
                    parsed = None
                if isinstance(parsed, (list, dict)):
                    value = parsed
        out[key] = value
    return out


def _reward_info(scores: dict[str, Any]) -> dict[str, float]:
    def _num(key: str) -> float:
        v = scores.get(key)
        return float(v) if isinstance(v, (int, float)) else 0.0

    return {k: _num(k) for k in _REWARD_KEYS}


class IndianBankingResourcesServerConfig(BaseResourcesServerConfig):
    # Point both at the same non-policy model server so the policy never judges itself.
    user_sim_model_server: ModelServerRef
    judge_model_server: ModelServerRef
    # Sampling params for the two auxiliary models; ``input`` is filled in per call.
    user_sim_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    should_use_judge: bool = True
    # Mixed text+tool-call turns end the episode when true. Needed by on-policy trainers
    # whose chat template cannot re-render such a turn byte-identically.
    strict_turn_protocol: bool = False

    max_user_turns: int = Field(8, ge=1)
    max_tool_rounds: int = Field(12, ge=1)
    user_sim_retries: int = Field(3, ge=1)
    judge_retries: int = Field(2, ge=0)
    judge_timeout_seconds: float = Field(60.0, gt=0)

    # Synthetic customer DB and knowledge base; default to data/ next to app.py.
    db_fpath: Optional[str] = None
    kb_fpath: Optional[str] = None


class IndianBankingResourcesServer(GymnasiumServer):
    config: IndianBankingResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        engine.configure(db_path=self.config.db_fpath, kb_path=self.config.kb_fpath)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        customer = metadata.get("customer")
        world = engine.seed_world(customer)

        initial_state = metadata.get("initial_state") or {}
        for action in initial_state.get("initialization_actions") or []:
            name = action.get("name")
            if not name:
                continue
            engine.apply_tool(world, name, dict(action.get("arguments") or {}))

        scenario = metadata.get("user_scenario") or {}
        task = {
            "task_id": metadata.get("task_id"),
            "customer": customer,
            "evaluation_criteria": metadata.get("evaluation_criteria") or {},
            "user_scenario": scenario,
        }

        opening = metadata.get("opening_message")
        if not opening:
            opening = user_sim.derive_opening_message(scenario)

        nl_history: list[dict[str, str]] = []
        if opening:
            nl_history.append({"role": "user", "content": opening})

        self.session_state[session_id] = {
            WORLD_KEY: world,
            TASK_KEY: task,
            "nl_history": nl_history,
            "dialog_user_turns": 0,
            "tool_rounds": 0,
        }
        return opening, {}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        state = self.session_state.get(session_id)
        if state is None:
            # reset() should always run first; this is just a defensive fallback.
            logger.warning("step() called with unknown session_id=%r", session_id)
            return None, 0.0, True, False, _empty_reward_info()

        function_calls = [item for item in action.output if getattr(item, "type", None) == "function_call"]

        if function_calls and extract_text(action).strip() and self.config.strict_turn_protocol:
            # Strict mode: a mixed turn ends the episode. Lenient mode runs the tools and drops the text.
            scores = await self._finish(state)
            return None, scores["score"], False, True, _reward_info(scores)

        if function_calls:
            world = state[WORLD_KEY]
            tool_outputs = []
            for call in function_calls:
                try:
                    args = json.loads(call.arguments) if call.arguments else {}
                except (TypeError, ValueError):
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                args = _coerce_args(args)
                args.update(_FORCED_TOOL_ARGS.get(call.name, {}))
                result = engine.apply_tool(world, call.name, args)
                tool_outputs.append(GymnasiumServer.tool_output(call, result))

            state["tool_rounds"] = state.get("tool_rounds", 0) + 1
            if state["tool_rounds"] >= self.config.max_tool_rounds:
                scores = await self._finish(state)
                return None, scores["score"], False, True, _reward_info(scores)

            return None, 0.0, False, False, {"tool_outputs": tool_outputs}

        assistant_text = extract_text(action)
        nl_history: list[dict[str, str]] = state.setdefault("nl_history", [])

        if not assistant_text.strip():
            # An empty turn (no text, no tool calls) ends the episode.
            scores = await self._finish(state)
            return None, scores["score"], False, True, _reward_info(scores)

        nl_history.append({"role": "assistant", "content": assistant_text})

        if state.get("dialog_user_turns", 0) >= self.config.max_user_turns:
            scores = await self._finish(state)
            return None, scores["score"], False, True, _reward_info(scores)

        user_reply = await self._user_sim_reply(state)
        if user_reply is None:
            # Unreachable after all retries: end the episode instead of hanging.
            scores = await self._finish(state)
            return None, scores["score"], False, True, _reward_info(scores)

        nl_history.append({"role": "user", "content": user_reply})

        if any(tok in user_reply for tok in user_sim.STOP_TOKENS):
            scores = await self._finish(state)
            return None, scores["score"], True, False, _reward_info(scores)

        state["dialog_user_turns"] = state.get("dialog_user_turns", 0) + 1
        return user_reply, 0.0, False, False, {}

    @staticmethod
    def _response_text(response: NeMoGymResponse) -> str:
        return extract_text(response).strip()

    async def _user_sim_reply(self, state: dict) -> Optional[str]:
        task = state.get(TASK_KEY) or {}
        scenario = task.get("user_scenario") or {}
        sim_system = user_sim.user_sim_system_prompt(scenario)

        nl_history = state.get("nl_history") or []
        # Role-swap: to the sim, the policy's turns are "user", its own are "assistant".
        sim_input = [NeMoGymEasyInputMessage(role="system", content=sim_system)]
        for m in nl_history:
            sim_input.append(
                NeMoGymEasyInputMessage(role="user" if m["role"] == "assistant" else "assistant", content=m["content"])
            )
        create_params = self.config.user_sim_responses_create_params.model_copy(deep=True)
        create_params.input = sim_input

        for attempt in range(self.config.user_sim_retries):
            try:
                resp = await self.server_client.post(
                    server_name=self.config.user_sim_model_server.name,
                    url_path="/v1/responses",
                    json=create_params,
                )
                await raise_for_status(resp)
                content = self._response_text(NeMoGymResponse.model_validate(await get_response_json(resp)))
                if content:
                    return content
            except Exception as exc:  # noqa: BLE001 - transient server hiccup, retry
                logger.warning("user-sim call failed (attempt %d): %s", attempt + 1, exc)
        return None

    async def _maybe_judge(self, state: dict) -> Optional[float]:
        if not self.config.should_use_judge:
            return None

        task = state.get(TASK_KEY) or {}
        request, direct_score = judge.build_judge_request(task, state.get("nl_history") or [])
        if request is None:
            return direct_score  # None (gated out) or 0.0 (empty transcript)

        if request.cache_key in judge._cache:
            return judge._cache[request.cache_key]

        create_params = self.config.judge_responses_create_params.model_copy(deep=True)
        create_params.input = [
            NeMoGymEasyInputMessage(role="system", content=request.system_prompt),
            NeMoGymEasyInputMessage(role="user", content=request.user_prompt),
        ]

        for _ in range(self.config.judge_retries + 1):
            try:
                resp = await self.server_client.post(
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=create_params,
                    # Explicit timeout: a wedged judge endpoint must fail over, not hang the rollout.
                    timeout=aiohttp.ClientTimeout(total=self.config.judge_timeout_seconds),
                )
                await raise_for_status(resp)
                content = self._response_text(NeMoGymResponse.model_validate(await get_response_json(resp)))
                score = judge.parse_verdict(content, request.num_assertions)
                judge._cache[request.cache_key] = score
                return score
            except Exception as exc:  # noqa: BLE001 - judge must never break scoring
                logger.warning("judge call failed: %s", exc)
        return None  # fail-open: leave the judge out rather than fake a 0

    async def _finish(self, state: dict) -> dict:
        judge_score = await self._maybe_judge(state)
        return reward.score_trajectory(state, judge_score=judge_score)


if __name__ == "__main__":
    IndianBankingResourcesServer.run_webserver()
