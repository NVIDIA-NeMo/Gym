# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
DWRL Pointwise GenRM Resources Server.

Rewards each rollout by asking a GenRM model to judge that one response on its own. The judge first writes an
evaluation of the response, then a 1-token call reads the probability that the GenRM gives to `score_token` ("yes").
That probability is the BT score, and the mean over the judge passes is the reward.

Input (/verify, once per rollout):
- responses_create_params: The conversation that the rollout answers
- response: The rollout's Response API object
- principle: Optional evaluation plan

Output:
- The rollout's reward, returned as soon as its judgment is done
"""
from __future__ import annotations

import json
import re
import asyncio
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionSystemMessageParam,
    NeMoGymChatCompletionUserMessageParam,
    NeMoGymChatCompletionAssistantMessageParam,
)
from resources_servers.dwrl_genrm_compare_pointwise.utils import (
    GenRMOutputParseError,
    compute_reward,
    extract_output_text,
    parse_genrm_output,
)


logger = logging.getLogger(__name__)

def flatten_to_single_turn(message_log):
    ret = ""
    for idx, message in enumerate(message_log):
        if message.role == "system":
            ret += "System: " + message.content.strip() + "\n\n"
        elif message.role == "user":
            ret += "User: " + message.content.strip() + "\n\n"
        elif message.role == "assistant":
            resp_no_thinking1 = re.sub(r"(?i)(?s)(\<think\>)(.*?)(\<\/think\>)", "", message.content).strip()
            resp_no_thinking2 = re.sub(r"(?i)(?s)(.*?)(\<\/think\>)", "", resp_no_thinking1).strip()
            ret += "Assistant: " + resp_no_thinking2 + "\n\n"
    
    return ret.strip()

def convert_to_chat_completion_format(message_log):
    ret = []
    for idx, message in enumerate(message_log):
        if message.role == "system":
            ret.append( NeMoGymChatCompletionSystemMessageParam(role="system", content=message.content) )
        elif message.role == "user":
            ret.append( NeMoGymChatCompletionUserMessageParam(role="user", content=message.content) )
        elif message.role == "assistant":
            ret.append( NeMoGymChatCompletionAssistantMessageParam(role="assistant", content=message.content) )
    
    return ret

class GenRMCompareConfig(BaseResourcesServerConfig):
    """Configuration for the GenRM compare server.

    Attributes:
        genrm_model_server: Target GenRM model server (default: genrm_model from config)
        genrm_responses_create_params: Base create params for GenRM calls
        num_judges_per_comparison: Number of judge passes per pair (majority voting)
        aggregator_method: Method for aggregating scores
        reasoning_bonus: Bonus for shortest reasoning content among top performers
        answer_bonus: Bonus for shortest answer among top performers
        top_percentile: Percentile threshold for applying bonuses
        group_reasoning_length_penalty_coeff: Coefficient for reasoning length penalty
        group_answer_length_penalty_coeff: Coefficient for answer length penalty
        default_score: Default neutral score when parsing fails
        default_ranking: BT score (the reward) of a judge pass whose GenRM call failed
        debug_logging: Enable verbose logging for debugging
        genrm_parse_retries: How many times a judge pass is repeated (both GenRM calls) when its evaluation cannot be
            parsed. 0 (the default): not repeated, an unparseable evaluation only gets default_score for its 1-5 score
        genrm_parse_retry_sleep_s: Sleep duration between parse retries
        use_principle: Enable principle-based comparison
        default_principle: Default principle when none provided in request
    """

    name: str = "genrm_compare"
    genrm_model_server: ModelServerRef  # Default: genrm_model (see config)
    #genrm_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    genrm_responses_create_params: NeMoGymChatCompletionCreateParamsNonStreaming

    # Comparison strategy
    #comparison_strategy: str = "circular"  # "all_pairs" or "circular"
    num_judges_per_comparison: int = Field(default=1, ge=1)

    # Principle-based GenRM settings
    use_principle: bool = False
    default_principle: str = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
        "to the user prompt. Begin your evaluation by generating your own answer to the prompt. You must provide "
        "your answer before judging any answers. When evaluating the assistants' answers, compare both assistants' "
        "answers with your answer. You must identify and correct any mistakes or inaccurate information. Then "
        "consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly "
        "responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than "
        "one interpretation, it is more helpful and appropriate to ask for clarifications or more information from "
        "the user than providing an answer based on assumptions. Relevant means all parts of the response closely "
        "connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or "
        "excessive. Then consider the creativity and novelty of the assistant's answers when needed. Finally, "
        "identify any missing important information in the assistants' answers that would be beneficial to include "
        "when responding to the user prompt."
    )

    # Aggregator settings (only "simple_tiebreaker" is currently implemented)
    aggregator_method: str = "simple_tiebreaker"

    # Length bonus config (only for simple_tiebreaker)
    reasoning_bonus: float = 0.0
    answer_bonus: float = 0.0
    top_percentile: float = 0.2
    group_reasoning_length_penalty_coeff: float = 0.0
    group_answer_length_penalty_coeff: float = 0.0

    # Defaults when parsing or judging fails. `default_score` stands in for an unparseable 1-5 score. `default_ranking`
    # is the reward of a pass whose GenRM call failed: 0.0, so that a failure is never worth more than a real score.
    default_score: float = 3.0
    default_ranking: float = 0.0

    # Debug logging
    debug_logging: bool = False

    # Retry config for parse failures. Parsing only raises, and so only triggers a retry, if genrm_parse_retries > 0.
    # The 1-5 score does not influence the reward, so by default (0) a parse failure only costs a warning.
    genrm_parse_retries: int = 0
    genrm_parse_retry_sleep_s: float = 0.2
    
    # DWRL specific
    score_token: str
    bt_prompt: str
    dwrl_prompt: str


class GenRMCompareVerifyRequest(BaseVerifyRequest):
    """Verify request with an optional principle for principle-based GenRM judging."""

    model_config = ConfigDict(extra="allow")

    principle: Optional[str] = None  # Principle for principle-based GenRM; forwarded by agent when provided


def _input_to_conversation_history(input_messages: Any) -> List[Dict[str, str]]:
    """Convert Response API input messages to conversation_history list of {role, content}."""
    out: List[Dict[str, str]] = []
    items = list(input_messages) if input_messages else []
    for m in items:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
        else:
            role = getattr(m, "role", "user")
            content = getattr(m, "content", "") or ""
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "output_text"
            )
        out.append({"role": str(role), "content": str(content)})
    return out


class GenRMCompareResourcesServer(SimpleResourcesServer):
    """Resources server that rewards each rollout with a DWRL pointwise GenRM judgment.

    - /verify: every rollout is judged on its own, and the reward is returned as soon as its judgment is done.
      Nothing is shared between rollouts, so a rollout that never arrives, arrives twice or arrives late cannot
      affect (or block) any other rollout, and the server can run any number of worker processes.
    """

    config: GenRMCompareConfig

    async def verify(self, body: GenRMCompareVerifyRequest) -> BaseVerifyResponse:
        """Judge one rollout with the GenRM and return its reward.

        The reward is the mean, over the `num_judges_per_comparison` judge passes, of the Bradley-Terry score
        (the probability that the GenRM gives to `score_token` for this rollout). It depends on this rollout only.
        """
        input_messages = getattr(body.responses_create_params, "input", None) or []
        conversation_history = _input_to_conversation_history(input_messages)
        response_obj = body.response.model_dump() if hasattr(body.response, "model_dump") else body.response

        judge_results = await self._judge_rollout(conversation_history, response_obj, body.principle)

        return BaseVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=compute_reward(judge_results),
        )

    async def _judge_rollout(
        self,
        conversation_history: List[Dict[str, str]],
        response_obj: Dict[str, Any],
        principle: Optional[str] = None,
    ) -> List[Tuple[float, float]]:
        """Judge one rollout `num_judges_per_comparison` times, concurrently. Results are in judge order."""
        results = await asyncio.gather(
            *[
                self._run_single_comparison(conversation_history, response_obj, principle=principle)
                for _ in range(self.config.num_judges_per_comparison)
            ]
        )
        return list(results)

    async def _run_single_comparison(
        self,
        conversation_history: List[Dict[str, str]],
        response_obj_1: Dict[str, Any],
        #response_obj_2: Dict[str, Any],
        #pair_idx: Tuple[int, int] = (0, 0),
        principle: Optional[str] = None,
    ) -> Tuple[float, float, float]:
        """Run a single pairwise comparison via GenRM.

        Args:
            conversation_history: The conversation context
            response_obj_1: First Response API object
            response_obj_2: Second Response API object
            pair_idx: Tuple of (i, j) for logging
            principle: Optional principle for principle-based comparison

        Returns:
            Tuple of (score_1, score_2, ranking)
        """
        cfg = self.config

        # Extract final answer from Response API objects (GenRM only takes the final answer, not reasoning)
        response_1 = extract_output_text(response_obj_1)
        #response_2 = extract_output_text(response_obj_2)

        # input carries only the conversation history (standard OpenAI roles).
        # The comparison payload is passed via metadata so the request schema stays
        # generic and GenRMModelMixin._preprocess_chat_completion_create_params can
        # inject the GenRM-specific roles (response_1, response_2, principle) server-side.
        messages: List[NeMoGymEasyInputMessage] = [
            NeMoGymEasyInputMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                type="message",
            )
            for msg in conversation_history
        ]

        principle_val = principle if principle else cfg.default_principle.strip()
        #print("### PRINCIPLE_INSIDE_COMPARE: ", principle_val, flush=True)
        prompt_template = cfg.dwrl_prompt.strip().format(context=flatten_to_single_turn(messages), response=response_1, principle=principle_val)
        init_msg = [NeMoGymEasyInputMessage(role="user", content=prompt_template, type="message")]

        # Build the request params
        responses_create_params = cfg.genrm_responses_create_params.model_copy(deep=True)
        #responses_create_params.input = init_msg
        #responses_create_params = NeMoGymChatCompletionCreateParamsNonStreaming(messages=convert_to_chat_completion_format(init_msg))
        responses_create_params.messages = convert_to_chat_completion_format(init_msg)
        #responses_create_params.metadata = {"chat_template_kwargs": json.dumps({"enable_thinking": True}), "extra_body": json.dumps({"chat_template_kwargs": {"enable_thinking": True}})}
        #responses_create_params.model = "nvidia/nvidia/Nemotron-3-Nano-30B-A3B"
        #responses_create_params.top_p = 0.95
        #responses_create_params.temperature = 1.0
        #responses_create_params.max_tokens = 16384

        try:
            # Retry logic for parse failures (not connection errors, which are handled elsewhere)
            max_attempts = max(1, int(cfg.genrm_parse_retries) + 1)

            for attempt_idx in range(max_attempts):
                # Call the GenRM model via /v1/responses endpoint (server name from config, e.g. genrm_model)
                response_thought = await self.server_client.post(
                    server_name=cfg.genrm_model_server.name,
                    url_path="/v1/chat/completions",
                    json=responses_create_params,
                )
                raw_response_thought = await response_thought.json()
                #print("### RAW_RESPONSE_THOUGHT: ", raw_response_thought, flush=True)

                # Extract raw string response without any post processing
                #thought_raw_str = raw_response_thought['output'][0]['content'][0]['text']
                thought_raw_str = raw_response_thought['choices'][0]['message']['content']
				
                # The second turn continues this attempt's evaluation. It is built here, not appended to init_msg,
                # so that a later attempt does not see the evaluations of the earlier ones.
                second_turn_msg = init_msg + [
                    NeMoGymEasyInputMessage(role="assistant", content=thought_raw_str, type="message"),
                    NeMoGymEasyInputMessage(role="user", content=cfg.bt_prompt, type="message"),
                ]
				
                responses_second_turn = cfg.genrm_responses_create_params.model_copy(deep=True)
                #responses_second_turn = NeMoGymChatCompletionCreateParamsNonStreaming(messages=convert_to_chat_completion_format(init_msg))
                #responses_second_turn.model = "model"
                #responses_second_turn.input = init_msg
                responses_second_turn.messages = convert_to_chat_completion_format(second_turn_msg)
                responses_second_turn.logprobs = True
                responses_second_turn.metadata = {"chat_template_kwargs": json.dumps({"enable_thinking": False}), "extra_body": json.dumps({"chat_template_kwargs": {"enable_thinking": False}})}
                responses_second_turn.top_p = 1.0
                responses_second_turn.temperature = 1.0
                responses_second_turn.max_tokens = 1
                responses_second_turn.top_logprobs = 20
				
                response_score = await self.server_client.post(
                    server_name=cfg.genrm_model_server.name,
                    url_path="/v1/chat/completions",
                    json=responses_second_turn,
                )
                raw_response_score = await response_score.json()
                #print("### RAW_RESPONSE_SCORE: ", raw_response_score, flush=True)
                
                top_logprobs = raw_response_score['choices'][0]['logprobs']['content'][-1]['top_logprobs']
                score = None
                for possible_tokens in top_logprobs:
                    if possible_tokens['token'] == cfg.score_token:
                        score = math.exp(possible_tokens['logprob'])
                        break
                if score is None:
                    for possible_tokens in top_logprobs:
                        if possible_tokens['token'].lower().strip() == cfg.score_token:
                            score = math.exp(possible_tokens['logprob'])
                            break
                if score is None:
                    # A token outside the top-k has a probability of at most 1/k, so 0 is the right order of magnitude.
                    # This is also what happens, for every rollout, if `score_token` does not match the model's tokens.
                    logger.warning(
                        f"[GenRM] Score token {cfg.score_token!r} not found in the top {len(top_logprobs)} logprobs "
                        f"{[t['token'] for t in top_logprobs]}; using a BT score of 0.0"
                    )
                    score = 0.0
                print("### BT_SCORE: ", score, flush=True)

                try:
                    resp_no_thinking1 = re.sub(r"(?i)(?s)(\<think\>)(.*?)(\<\/think\>)", "", thought_raw_str).strip()
                    resp_no_thinking2 = re.sub(r"(?i)(?s)(.*?)(\<\/think\>)", "", resp_no_thinking1).strip()
                    
                    score_1 = parse_genrm_output(
                        resp_no_thinking2,
                        cfg.default_score,
                        raise_on_fail=cfg.genrm_parse_retries > 0,
                    )
                    print(f"Score_1: {score_1}", flush=True)
                    return score_1, score

                except GenRMOutputParseError as e:
                    if attempt_idx < max_attempts - 1:
                        logger.warning(
                            f"[GenRM] Parse failed (attempt {attempt_idx + 1} of {max_attempts}); retrying. {e}"
                        )
                        await asyncio.sleep(float(cfg.genrm_parse_retry_sleep_s))
                        continue

                    # Give up: fall back to defaults
                    logger.warning(
                        f"[GenRM] Parse failed after {max_attempts} attempts; "
                        f"falling back to defaults. {e}"
                    )
                    return cfg.default_score, score

            return cfg.default_score, cfg.default_ranking

        except Exception as e:
            logger.error(f"[GenRM] Error in comparison: {e}")
            return cfg.default_score, cfg.default_ranking


if __name__ == "__main__":
    GenRMCompareResourcesServer.run_webserver()
