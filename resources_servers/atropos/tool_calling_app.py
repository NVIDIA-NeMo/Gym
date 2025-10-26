# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

atropos_path = Path(__file__).parent.parent.parent / "atropos"
if str(atropos_path) not in sys.path:
    sys.path.insert(0, str(atropos_path))


from environments.tool_use_multiturn_server import (
    MultiTurnToolCallingEnv,
    system_prompt,
    _validate_reply_and_extract,
    _json_objects_match,
    _parse_expected_call,
)

from nemo_gym.integrations.atropos import AtroposStepRequest, AtroposStepResponse
from nemo_gym.openai_utils import NeMoGymEasyInputMessage

from resources_servers.atropos.app import AtroposResourcesServer

from fastapi import Request


class ToolCallingAtroposServer(AtroposResourcesServer[MultiTurnToolCallingEnv]):
    system_prompt: Optional[str] = system_prompt
    _shared_env: Optional[MultiTurnToolCallingEnv] = None

    async def env_factory(self, task_idx: int) -> MultiTurnToolCallingEnv:
        if self._shared_env is None:
            env_config, server_configs = MultiTurnToolCallingEnv.config_init()
            env_config.scenario_category = "multiturn"
            env_config.max_tool_call_turns_cap = 5
            env_config.wrong_call_penalty = -0.2

            self._shared_env = MultiTurnToolCallingEnv(
                config=env_config,
                server_configs=server_configs,
                slurm=False,
                testing=True,
            )
            await self._shared_env.setup()
        return self._shared_env

    async def get_initial_item(self, env: MultiTurnToolCallingEnv, task_idx: int) -> Dict[str, Any]:
        messages_tuple, expected_calls_by_turn, inter_turns = env.train_items[
            task_idx % len(env.train_items)
        ]

        messages = [dict(m) for m in messages_tuple]
        tools_definition = ""
        if messages and messages[0].get("role") == "system":
            tools_definition = messages[0]["content"]

        return {
            "messages": messages,
            "expected_calls": expected_calls_by_turn,
            "inter_turns": inter_turns,
            "current_turn": 0,
            "predicted_calls": [],
            "tools": tools_definition,
        }

    async def format_item_as_message(self, item: Dict[str, Any]) -> str:
        messages = item["messages"]
        for msg in messages:
            if msg.get("role") in ("user", "human") or msg.get("from") == "human":
                content = msg.get("content") or msg.get("value", "")
                if item.get("tools"):
                    return f"{item['tools']}\n\n{content}"
                return content
        return "No user message found"

    async def score_response(
        self,
        env: MultiTurnToolCallingEnv,
        item: Dict[str, Any],
        response: str,
        messages: list,
    ) -> Tuple[float, bool, Optional[Dict]]:
        # Fix missing opening <think> tag if response has </think> but not <think>
        if "</think>" in response.lower() and "<think>" not in response.lower():
            response = "<think>\n" + response

        tool_calls = _validate_reply_and_extract(response)

        # Debug logging
        print(f"\n=== SCORING DEBUG ===")
        print(f"Response length: {len(response)}")
        print(f"Has <think>: {'<think>' in response.lower()}")
        print(f"Has </think>: {'</think>' in response.lower()}")
        print(f"Extracted tool calls: {tool_calls}")
        print(f"Tool calls is None: {tool_calls is None}")

        if tool_calls is None:
            tool_calls = []
        item["predicted_calls"].extend(tool_calls)

        current_turn = item["current_turn"]
        expected_calls = item["expected_calls"]
        inter_turns = item["inter_turns"]
        is_final_turn = current_turn + 1 >= len(expected_calls)

        if not is_final_turn:
            tool_responses = self._simulate_tool_responses(tool_calls)
            obs_text = "\n\n".join(tool_responses)

            if current_turn < len(inter_turns):
                inter_parts = [msg.get("value") or msg.get("content", "") for msg in inter_turns[current_turn]]
                if inter_parts:
                    obs_text += "\n\n" + "\n\n".join(inter_parts)

            item["current_turn"] += 1
            return 0.0, False, {"observation": obs_text}

        all_predicted = item["predicted_calls"]
        all_expected = [call for turn in expected_calls for call in turn]

        # Use atropos scoring
        exp_jsons = [_parse_expected_call(c) for c in all_expected]
        correct = sum(1 for p, e in zip(all_predicted, exp_jsons) if _json_objects_match(p, e))

        # Debug final scoring
        print(f"\n=== FINAL SCORING ===")
        print(f"Total predicted calls: {len(all_predicted)}")
        print(f"Total expected calls: {len(all_expected)}")
        print(f"Predicted calls: {all_predicted}")
        print(f"Expected jsons: {exp_jsons}")
        print(f"Correct matches: {correct}")

        dense = correct / max(1, len(exp_jsons))
        bonus = 0.5 if correct == len(exp_jsons) else 0.0
        penalty = -0.2 if all_predicted and correct < len(exp_jsons) else 0.0
        reward = dense + bonus + penalty

        print(f"Reward: {reward} (dense={dense}, bonus={bonus}, penalty={penalty})")
        print(f"===================\n")

        return reward, True, {
            "correct": correct == len(all_expected),
            "num_correct_calls": correct,
            "total_expected_calls": len(all_expected),
            "predicted_calls": [self._call_to_str(c) for c in all_predicted],
            "expected_calls": [self._call_to_str(c) for c in all_expected],
            "reward_breakdown": {
                "dense": dense,
                "bonus": bonus,
                "penalty": penalty,
            },
        }

    async def step(
        self,
        request: Request,
        body: AtroposStepRequest,
    ) -> AtroposStepResponse:
        state = self.env_id_to_state[body.env_id]
        state.messages.append({"role": "assistant", "content": body.action})

        reward, done, info = await self.score_response(
            env=state.env,
            item=state.item,
            response=body.action,
            messages=state.messages,
        )

        state.total_reward += reward
        state.done = done
        state.info = info or {}
        self.env_id_to_total_reward[body.env_id] = state.total_reward

        obs = []
        if not done and "observation" in info:
            obs.append(NeMoGymEasyInputMessage(role="user", content=info["observation"]))

        return AtroposStepResponse(obs=obs, reward=reward, done=done, info=info)

    def _simulate_tool_responses(self, tool_calls: List[Dict[str, Any]]) -> List[str]:
        responses = []
        for call in tool_calls:
            response = {
                "status": "success",
                "function": call.get("name", "unknown"),
                "result": "Operation completed successfully",
            }
            responses.append(f"<tool_response>\n{json.dumps(response, indent=2)}\n</tool_response>")
        return responses

    def _call_to_str(self, call: Any) -> str:
        if isinstance(call, str):
            try:
                call = json.loads(call)
            except:
                return call
        if not isinstance(call, dict):
            return str(call)
        return f"{call.get('name', 'unknown')}({json.dumps(call.get('arguments', {}))})"


if __name__ == "__main__":
    ToolCallingAtroposServer.run_webserver()
