# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-turn TIR judge resource server.
#
# Provides multi-turn awareness for TIR tasks (summarize + retry flow)
# with math_with_judge as the final verifier.
#
# Supports two response processors:
#   strip_thinking  — removes <think>…</think>, stripped text as previous_attempt
#   summary_model   — the agent handles summary turns; this server builds
#                     summarize/retry prompts and delegates final verification
import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ConfigDict as PydanticConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import get_response_json

LOG = logging.getLogger(__name__)

PROMPT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"


def _load_prompt_template(filename: str) -> str:
    with open(PROMPT_TEMPLATES_DIR / filename) as f:
        return yaml.safe_load(f)["user"]


SUMMARIZE_REASONING_TEMPLATE = _load_prompt_template("summarize_reasoning.yaml")
RETRY_WITH_SUMMARY_TEMPLATE = _load_prompt_template("retry_with_summary.yaml")


def strip_thinking(text: str) -> str:
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in result:
        result = result.split("<think>")[0].strip()
    elif "</think>" in result:
        result = result.split("</think>", 1)[1].strip()
    elif "</think>" not in text and "<think>" in text:
        result = ""
    return result


def extract_thinking(text: str) -> str:
    if "<think>" not in text:
        return text.strip()
    result = text.replace("<think>", "", 1)
    if "</think>" in result:
        result = result.split("</think>", 1)[0]
    return result.strip()


# ---------------------------------------------------------------------------
#  Config and request/response models
# ---------------------------------------------------------------------------

class MathTIRMultiturnConfig(BaseResourcesServerConfig):
    verifier_server_name: str = "math_with_judge"
    max_turns: int = 2
    reward_strategy: str = "final_only"
    response_processor: str = "summary_model"
    zero_truncated_turn_reward: bool = False
    truncated_turn_reward_share: float = 1.0


class MathTIRMultiturnVerifyRequest(BaseVerifyRequest):
    model_config = PydanticConfigDict(extra="allow")
    problem: str = ""
    question: str = ""
    expected_answer: str = ""
    turn_index: int = 0
    was_truncated: bool = False
    is_summary_turn: bool = False
    existing_summary: str = "None"


class MathTIRMultiturnVerifyResponse(BaseVerifyResponse):
    model_config = PydanticConfigDict(extra="allow")
    needs_correction: bool = False
    correction_prompt: Optional[str] = None
    is_summary_prompt: bool = False
    turn_info: dict = {}


class MathTIRMultiturnSeedRequest(BaseSeedSessionRequest):
    model_config = PydanticConfigDict(extra="allow")


class MathTIRMultiturnSeedResponse(BaseSeedSessionResponse):
    pass


# ---------------------------------------------------------------------------
#  Server
# ---------------------------------------------------------------------------

class MathTIRMultiturnServer(SimpleResourcesServer):
    config: MathTIRMultiturnConfig

    async def seed_session(self, body: MathTIRMultiturnSeedRequest) -> MathTIRMultiturnSeedResponse:
        return MathTIRMultiturnSeedResponse()

    def _effective_truncated_share(self) -> float:
        share = self.config.truncated_turn_reward_share
        if share >= 1.0 and self.config.zero_truncated_turn_reward:
            return 0.0
        return share

    async def verify(self, body: MathTIRMultiturnVerifyRequest) -> MathTIRMultiturnVerifyResponse:
        problem = body.problem or body.question or ""
        full_response = self._extract_assistant_text(body.response)
        turn_index = body.turn_index
        was_truncated = body.was_truncated
        is_summary_turn = body.is_summary_turn

        use_summary_model = self.config.response_processor == "summary_model"
        reasoning_turn = self._count_reasoning_turns(turn_index, use_summary_model, is_summary_turn)
        is_final_reasoning = reasoning_turn >= self.config.max_turns
        truncated_share = self._effective_truncated_share()

        if is_summary_turn:
            clean_summary = strip_thinking(full_response)
            retry_prompt = RETRY_WITH_SUMMARY_TEMPLATE.format(
                problem=problem,
                summary=clean_summary,
            )
            return MathTIRMultiturnVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                needs_correction=True,
                correction_prompt=retry_prompt,
                is_summary_prompt=False,
                turn_info={
                    "turn_index": turn_index,
                    "reasoning_turn": reasoning_turn,
                    "is_summary_turn": True,
                    "_existing_summary": clean_summary,
                    "truncated_turn_reward_share": truncated_share,
                },
            )

        if is_final_reasoning:
            reward = await self._delegate_verify(body)
            return MathTIRMultiturnVerifyResponse(
                **body.model_dump(),
                reward=reward,
                needs_correction=False,
                turn_info={
                    "turn_index": turn_index,
                    "reasoning_turn": reasoning_turn,
                    "is_final": True,
                    "truncated_turn_reward_share": truncated_share,
                },
            )
        else:
            processed = strip_thinking(full_response)

            if use_summary_model:
                reasoning_trace = self._extract_reasoning_trace(body.response)
                if not reasoning_trace:
                    reasoning_trace = strip_thinking(full_response) or full_response
                summary_prompt = SUMMARIZE_REASONING_TEMPLATE.format(
                    problem=problem,
                    reasoning_process=reasoning_trace,
                )
                return MathTIRMultiturnVerifyResponse(
                    **body.model_dump(),
                    reward=0.0,
                    needs_correction=True,
                    correction_prompt=summary_prompt,
                    is_summary_prompt=True,
                    turn_info={
                        "turn_index": turn_index,
                        "reasoning_turn": reasoning_turn,
                        "_phase": "summarizing",
                        "_stripped_assistant_content": processed,
                        "truncated_turn_reward_share": truncated_share,
                    },
                )
            else:
                retry_prompt = RETRY_WITH_SUMMARY_TEMPLATE.format(
                    problem=problem,
                    summary=processed,
                )
                return MathTIRMultiturnVerifyResponse(
                    **body.model_dump(),
                    reward=0.0,
                    needs_correction=True,
                    correction_prompt=retry_prompt,
                    is_summary_prompt=False,
                    turn_info={
                        "turn_index": turn_index,
                        "reasoning_turn": reasoning_turn,
                        "_stripped_assistant_content": processed,
                        "truncated_turn_reward_share": truncated_share,
                    },
                )

    def _count_reasoning_turns(self, turn_index: int, use_summary_model: bool, is_summary_turn: bool) -> int:
        if not use_summary_model:
            return turn_index + 1
        if is_summary_turn:
            return (turn_index + 1) // 2
        return (turn_index // 2) + 1

    async def _delegate_verify(self, body: MathTIRMultiturnVerifyRequest) -> float:
        """Delegate final verification to math_with_judge."""
        verify_data = body.model_dump()
        resp = await self.server_client.post(
            server_name=self.config.verifier_server_name,
            url_path="/verify",
            json=verify_data,
        )
        result = await get_response_json(resp)
        return float(result.get("reward", 0.0))

    @staticmethod
    def _get_outputs(response: Any) -> list:
        if not response:
            return []
        if isinstance(response, dict):
            return response.get("output", []) or []
        return getattr(response, "output", []) or []

    def _extract_assistant_text(self, response: Any) -> str:
        outputs = self._get_outputs(response)
        if not outputs:
            return ""
        reasoning_parts = []
        content_parts = []
        for out in outputs:
            out_type = out.get("type") if isinstance(out, dict) else getattr(out, "type", None)
            out_role = out.get("role") if isinstance(out, dict) else getattr(out, "role", None)
            if out_type == "reasoning":
                summaries = out.get("summary", []) if isinstance(out, dict) else getattr(out, "summary", [])
                for s in summaries or []:
                    s_text = s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")
                    if s_text:
                        reasoning_parts.append(s_text)
            elif out_type == "message" and out_role == "assistant":
                content_list = out.get("content", []) if isinstance(out, dict) else getattr(out, "content", [])
                for c in content_list or []:
                    c_type = c.get("type") if isinstance(c, dict) else getattr(c, "type", None)
                    c_text = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
                    if c_type == "output_text":
                        content_parts.append(c_text or "")
        result = ""
        if reasoning_parts:
            result = "<think>" + "\n".join(reasoning_parts) + "</think>"
        result += "".join(content_parts)
        return result

    def _extract_reasoning_trace(self, response: Any) -> str:
        """Extract full reasoning trace including thinking, tool calls, and results.

        Produces a readable trace for the summary model that includes all
        output types in order: reasoning (thinking), assistant text,
        tool invocations, and tool execution results.
        """
        outputs = self._get_outputs(response)
        if not outputs:
            return ""
        parts = []
        for out in outputs:
            out_type = out.get("type") if isinstance(out, dict) else getattr(out, "type", None)
            if out_type == "reasoning":
                summaries = out.get("summary", []) if isinstance(out, dict) else getattr(out, "summary", [])
                for s in summaries or []:
                    s_text = s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")
                    if s_text:
                        parts.append(s_text.strip())
            elif out_type == "message":
                out_role = out.get("role") if isinstance(out, dict) else getattr(out, "role", None)
                if out_role == "assistant":
                    content_list = out.get("content", []) if isinstance(out, dict) else getattr(out, "content", [])
                    for c in content_list or []:
                        c_text = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
                        if c_text and c_text.strip():
                            parts.append(c_text.strip())
            elif out_type == "function_call":
                name = out.get("name", "tool") if isinstance(out, dict) else getattr(out, "name", "tool")
                args = out.get("arguments", "") if isinstance(out, dict) else getattr(out, "arguments", "")
                if args:
                    parts.append(f"[Executed {name}]\n{args}")
            elif out_type == "function_call_output":
                output_text = out.get("output", "") if isinstance(out, dict) else getattr(out, "output", "")
                if output_text:
                    parts.append(f"[Execution result]\n{output_text}")
        return "\n\n".join(parts)


if __name__ == "__main__":
    MathTIRMultiturnServer.run_webserver()
