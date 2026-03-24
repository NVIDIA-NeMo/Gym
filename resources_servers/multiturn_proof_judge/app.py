# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-turn proof judge resource server.
#
# Extends proof_judge with multi-turn awareness:
#   - Non-final turns: strip thinking, build reprompt, return needs_correction=True
#   - Final turn: judge proof via verifier + meta-verifier, return reward
#
# Supports two response processors:
#   strip_thinking  — removes <think>…</think>, visible text as previous_attempt
#   summary_model   — the agent handles summary turns; this server just provides
#                     reprompt building when called with is_summary_turn=True
#
# Judge modes (same as proof_judge):
#   A) Gym-internal: judge_model_server points to a Gym-managed vllm_model.
#   B) External: reads JUDGE_SERVER_ARGS env var → AsyncOpenAI /v1/chat/completions
import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import PrivateAttr

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
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json

LOG = logging.getLogger(__name__)

LOG_JSONL_PATH = os.environ.get("PROOF_JUDGE_LOG_JSONL_PATH", None)

PROMPT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"


def _load_prompt_template(filename: str) -> str:
    with open(PROMPT_TEMPLATES_DIR / filename) as f:
        return yaml.safe_load(f)["user"]


PROVER_PROMPT_TEMPLATE = _load_prompt_template("prover.yaml")
VERIFIER_PROMPT_TEMPLATE = _load_prompt_template("verifier.yaml")
META_VERIFIER_PROMPT_TEMPLATE = _load_prompt_template("meta-verifier.yaml")
PROVER_REPROMPT_TEMPLATE = _load_prompt_template("prover_reprompt.yaml")
PROVER_REPROMPT_TRUNCATED_TEMPLATE = _load_prompt_template("prover_reprompt_truncated.yaml")
PROVER_SUMMARY_TEMPLATE = _load_prompt_template("prover_summary.yaml")

SOLUTION_HEADER = "## Solution"
SELF_EVAL_HEADER = "## Self Evaluation"

_FALLBACK_EMPTY_PREVIOUS = "(Your previous response was empty or could not be parsed.)"


# ---------------------------------------------------------------------------
#  External judge helpers
# ---------------------------------------------------------------------------

def _get_judge_client_config() -> Optional[tuple[str, str, int, list[str]]]:
    raw = os.environ.get("JUDGE_SERVER_ARGS")
    if not raw:
        return None
    cfg = json.loads(raw)
    model = cfg["model"]
    server_type = cfg["server_type"]
    port = cfg["port"]

    host = cfg.get("host")
    if host:
        master_nodes = [host]
    else:
        n_servers = cfg.get("n_servers", 1)
        judge_het_group = cfg.get("judge_het_group", 0)
        master_nodes = []
        for i in range(n_servers):
            env_var = f"SLURM_MASTER_NODE_HET_GROUP_{judge_het_group + i}"
            node = os.environ.get(env_var)
            if node:
                master_nodes.append(node)
        if not master_nodes:
            master_nodes = ["localhost"]
            LOG.info("[multiturn_proof_judge] No SLURM het group vars found, falling back to localhost")

    LOG.info(
        "[multiturn_proof_judge] External judge: model=%s port=%s n_servers=%s",
        model, port, len(master_nodes),
    )
    return model, server_type, port, master_nodes


# ---------------------------------------------------------------------------
#  Text processing
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks from model responses."""
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in result:
        result = result.split("<think>")[0].strip()
    elif "</think>" in result:
        result = result.split("</think>", 1)[1].strip()
    elif "</think>" not in text and "<think>" in text:
        result = ""
    return result


def extract_thinking(text: str) -> str:
    """Extract the content inside <think>…</think> tags."""
    if "<think>" not in text:
        return text.strip()
    result = text.replace("<think>", "", 1)
    if "</think>" in result:
        result = result.split("</think>", 1)[0]
    return result.strip()


def extract_boxed_score(text: str) -> Optional[float]:
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    content_start = start + len("\\boxed{")
    end = text.find("}", content_start)
    if end == -1:
        return None
    try:
        score = float(text[content_start:end].strip())
        return score if score in (0, 0.5, 1) else None
    except ValueError:
        return None


def parse_response(
    response: str, assert_think_end: bool = False
) -> tuple[Optional[tuple[str, str, float]], Optional[str]]:
    if assert_think_end and "</think>" not in response:
        return None, "missing_think_end"
    response = response.split("</think>")[-1].strip()
    if SOLUTION_HEADER not in response:
        return None, "missing_solution_header"
    after_solution = response.split(SOLUTION_HEADER, 1)[1]
    if SELF_EVAL_HEADER not in after_solution:
        return None, "missing_self_eval_header"
    proof, self_eval = after_solution.split(SELF_EVAL_HEADER, 1)
    return (proof.strip(), self_eval.strip(), extract_boxed_score(self_eval) or 0.0), None


def build_reprompt(
    processed_content: str,
    *,
    was_truncated: bool = False,
    problem: str = "",
) -> str:
    previous_attempt = processed_content if processed_content else _FALLBACK_EMPTY_PREVIOUS
    template = PROVER_REPROMPT_TRUNCATED_TEMPLATE if was_truncated else PROVER_REPROMPT_TEMPLATE
    return template.format(problem=problem, previous_attempt=previous_attempt)


def build_summary_prompt(
    problem: str, reasoning: str, existing_summary: str = "None"
) -> str:
    return PROVER_SUMMARY_TEMPLATE.format(
        problem=problem, reasoning=reasoning, existing_summary=existing_summary,
    )


# ---------------------------------------------------------------------------
#  Config and request/response models
# ---------------------------------------------------------------------------

class MultiturnProofJudgeConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_model_name: str = ""
    alpha: float = 1.0
    beta: float = 0.0
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 100000
    assert_think_end: bool = False
    max_turns: int = 2
    reward_strategy: str = "final_only"
    response_processor: str = "strip_thinking"
    zero_truncated_turn_reward: bool = False
    truncated_turn_reward_share: float = 1.0


class MultiturnProofVerifyRequest(BaseVerifyRequest):
    problem: str = ""
    turn_index: int = 0
    was_truncated: bool = False
    is_summary_turn: bool = False
    existing_summary: str = "None"


class MultiturnProofVerifyResponse(BaseVerifyResponse):
    needs_correction: bool = False
    correction_prompt: Optional[str] = None
    is_summary_prompt: bool = False
    turn_info: dict = {}


class MultiturnProofSeedRequest(BaseSeedSessionRequest):
    model_config = PydanticConfigDict(extra="allow")
    problem: str = ""


class MultiturnProofSeedResponse(BaseSeedSessionResponse):
    initial_prompt: Optional[str] = None


# ---------------------------------------------------------------------------
#  Server
# ---------------------------------------------------------------------------

class MultiturnProofJudgeServer(SimpleResourcesServer):
    config: MultiturnProofJudgeConfig

    _ext_clients: Optional[list] = PrivateAttr(default=None)
    _ext_model: Optional[str] = PrivateAttr(default=None)
    _ext_rr_counter: int = PrivateAttr(default=0)
    _log_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)

    async def seed_session(self, body: MultiturnProofSeedRequest) -> MultiturnProofSeedResponse:
        if body.problem:
            initial_prompt = PROVER_PROMPT_TEMPLATE.format(problem=body.problem)
            return MultiturnProofSeedResponse(initial_prompt=initial_prompt)
        return MultiturnProofSeedResponse()

    def _effective_truncated_share(self) -> float:
        """Resolve effective truncated_turn_reward_share from config.

        New ``truncated_turn_reward_share`` (float, default 1.0) takes
        precedence.  Falls back to legacy ``zero_truncated_turn_reward``
        (bool) when the float has not been explicitly lowered from 1.0.
        """
        share = self.config.truncated_turn_reward_share
        if share >= 1.0 and self.config.zero_truncated_turn_reward:
            return 0.0
        return share

    async def verify(self, body: MultiturnProofVerifyRequest) -> MultiturnProofVerifyResponse:
        problem = body.problem or ""
        full_response = self._extract_assistant_text(body.response)
        turn_index = body.turn_index
        was_truncated = body.was_truncated
        is_summary_turn = body.is_summary_turn
        existing_summary = body.existing_summary

        use_summary_model = self.config.response_processor == "summary_model"
        reasoning_turn = self._count_reasoning_turns(turn_index, use_summary_model, is_summary_turn)
        is_final_reasoning = reasoning_turn >= self.config.max_turns
        truncated_share = self._effective_truncated_share()

        if is_summary_turn:
            # Summary turn: model just generated a summary. Use it in reprompt.
            clean_summary = strip_thinking(full_response)
            reprompt = build_reprompt(
                clean_summary, was_truncated=was_truncated, problem=problem,
            )
            return MultiturnProofVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                needs_correction=True,
                correction_prompt=reprompt,
                is_summary_prompt=False,
                turn_info={
                    "turn_index": turn_index,
                    "reasoning_turn": reasoning_turn,
                    "is_summary_turn": True,
                    "_existing_summary": clean_summary,
                    "_stripped_assistant_content": clean_summary,
                    "truncated_turn_reward_share": truncated_share,
                },
            )

        if is_final_reasoning:
            # Final reasoning turn → judge the proof
            reward, details = await self._judge_single(problem, full_response)
            if LOG_JSONL_PATH:
                await self._append_log_jsonl(
                    event="judge",
                    turn_index=turn_index,
                    reasoning_turn=reasoning_turn,
                    reward=reward,
                    problem=problem[:200],
                    response_len=len(full_response),
                    judge_details=details,
                )
            return MultiturnProofVerifyResponse(
                **body.model_dump(),
                reward=reward,
                needs_correction=False,
                turn_info={
                    "turn_index": turn_index,
                    "reasoning_turn": reasoning_turn,
                    "is_final": True,
                    "_judge_info": details,
                    "truncated_turn_reward_share": truncated_share,
                },
            )
        else:
            # Non-final reasoning turn → build next turn input
            processed = strip_thinking(full_response)

            if use_summary_model:
                # Need a summary turn next: extract thinking chain for summary
                thinking_content = extract_thinking(full_response)
                summary_prompt = build_summary_prompt(
                    problem=problem,
                    reasoning=thinking_content,
                    existing_summary=existing_summary,
                )
                return MultiturnProofVerifyResponse(
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
                # strip_thinking mode: build reprompt directly
                reprompt = build_reprompt(
                    processed, was_truncated=was_truncated, problem=problem,
                )
                return MultiturnProofVerifyResponse(
                    **body.model_dump(),
                    reward=0.0,
                    needs_correction=True,
                    correction_prompt=reprompt,
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
        # With summary model: reasoning turns are 0, 2, 4, ... and summary turns are 1, 3, 5, ...
        # reasoning_turn = (turn_index + 1) // 2 if not is_summary_turn else turn_index // 2
        if is_summary_turn:
            return (turn_index + 1) // 2
        return (turn_index // 2) + 1

    def _extract_assistant_text(self, response: Any) -> str:
        """Extract full assistant text from response, including reasoning.

        When vLLM uses a reasoning_parser, thinking content is split into a
        separate ``type="reasoning"`` output item.  We reconstruct the full
        ``<think>…</think>`` + answer string so that downstream helpers like
        ``extract_thinking`` and ``strip_thinking`` work correctly.
        """
        if not response:
            return ""
        if isinstance(response, dict):
            outputs = response.get("output", []) or []
        else:
            outputs = getattr(response, "output", []) or []
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

    # ------------------------------------------------------------------
    #  External judge
    # ------------------------------------------------------------------

    def _init_external_clients(self) -> None:
        from openai import AsyncOpenAI

        cfg = _get_judge_client_config()
        if cfg is None:
            raise RuntimeError("_init_external_clients called but JUDGE_SERVER_ARGS is not set")
        model, _server_type, port, master_nodes = cfg
        self._ext_model = model
        self._ext_clients = []
        for node in master_nodes:
            base_url = f"http://{node}:{port}/v1"
            client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=60 * 60 * 4)
            self._ext_clients.append(client)
        LOG.info("[multiturn_proof_judge] Initialized %d external judge clients", len(self._ext_clients))

    def _next_ext_client(self):
        client = self._ext_clients[self._ext_rr_counter % len(self._ext_clients)]
        self._ext_rr_counter += 1
        return client

    async def _call_judge_external(self, user_content: str) -> str:
        if self._ext_clients is None:
            self._init_external_clients()
        client = self._next_ext_client()
        response = await client.chat.completions.create(
            model=self._ext_model,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    async def _call_judge_internal(self, user_content: str) -> str:
        from nemo_gym.server_utils import raise_for_status

        server_name = self.config.judge_model_server.name
        model = self.config.judge_model_name or server_name
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content=user_content)],
            model=model,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_output_tokens=self.config.max_tokens,
        )
        resp = await self.server_client.post(
            server_name=server_name,
            url_path="/v1/responses",
            json=params.model_dump(),
        )
        if resp.status >= 400:
            LOG.warning("[multiturn_proof_judge] Judge HTTP %s", resp.status)
        await raise_for_status(resp)
        data = await get_response_json(resp)
        judge_resp = NeMoGymResponse.model_validate(data)
        return self._extract_assistant_text(judge_resp)

    async def _call_judge(self, user_content: str) -> str:
        if os.environ.get("JUDGE_SERVER_ARGS"):
            return await self._call_judge_external(user_content)
        return await self._call_judge_internal(user_content)

    async def _judge_single(
        self, problem: str, full_response: str
    ) -> tuple[float, dict[str, Any]]:
        alpha = self.config.alpha
        beta = self.config.beta

        parsed, reason = parse_response(
            full_response, assert_think_end=self.config.assert_think_end
        )
        if parsed is None:
            return 0.0, {"r_format": 0.0, "reason": reason}

        proof, self_analysis, s_prime = parsed

        verifier_prompt = VERIFIER_PROMPT_TEMPLATE.format(problem=problem, proof=proof)
        verifier_response = await self._call_judge(verifier_prompt)
        r_y = extract_boxed_score(verifier_response) or 0.0

        if beta == 0:
            return alpha * r_y, {
                "r_y": r_y,
                "s_prime": s_prime,
                "verifier_response": verifier_response,
            }

        meta_prompt = META_VERIFIER_PROMPT_TEMPLATE.format(
            problem=problem, proof=proof, proof_analysis=self_analysis
        )
        meta_response = await self._call_judge(meta_prompt)
        r_meta = extract_boxed_score(meta_response) or 0.0
        r_z = (1.0 - abs(s_prime - r_y)) * r_meta
        return alpha * r_y + beta * r_z, {
            "r_y": r_y,
            "r_meta": r_meta,
            "s_prime": s_prime,
            "verifier_response": verifier_response,
            "meta_response": meta_response,
        }

    async def _append_log_jsonl(self, **kwargs) -> None:
        if self._log_lock is None:
            self._log_lock = asyncio.Lock()
        try:
            record = {"ts": datetime.now(timezone.utc).isoformat(), **kwargs}
            async with self._log_lock:
                with open(LOG_JSONL_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            LOG.warning("[multiturn_proof_judge] Failed to append log: %s", e)


if __name__ == "__main__":
    MultiturnProofJudgeServer.run_webserver()
