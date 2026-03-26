# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Proof-with-judge resource server: verifier + meta-verifier reward for theorem proving.
# Combined env that handles both data formatting (via prepare_data.py) and verification.
# Reuses the same judge logic as proof_judge.
#
# Supports two judge paths:
# 1. Gym-internal: calls judge_model_server via /v1/responses (judge managed by Gym/Ray)
# 2. External: reads JUDGE_SERVER_ARGS env var → AsyncOpenAI /v1/chat/completions
# (judge servers managed by nemo-skill pipeline as SLURM het groups)
import asyncio
import json
import logging
import os
import socket
import time
from asyncio import Event
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
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
MAX_PROOF_CHARS = 40_000
MAX_VERIFICATION_CHARS = 20_000


def _load_prompt_template(filename: str) -> str:
    """Load the 'user' field from a prompt YAML file in prompt_templates/."""
    with open(PROMPT_TEMPLATES_DIR / filename) as f:
        return yaml.safe_load(f)["user"]


VERIFIER_PROMPT_TEMPLATE = _load_prompt_template("verifier.yaml")
META_VERIFIER_PROMPT_TEMPLATE = _load_prompt_template("meta-verifier.yaml")


# ---------------------------------------------------------------------------
# External judge helpers (from math-with-judge-het pattern)
# ---------------------------------------------------------------------------


def _get_judge_client_config() -> tuple[str, str, int, list[str]]:
    """Get base_url and model for the judge servers from environment.

    Returns:
        model: The model name
        server_type: The server type
        port: The port (same for all servers)
        master_nodes: List of master nodes for each server
    """
    server_args_str = os.environ["JUDGE_SERVER_ARGS"]
    server_config = json.loads(server_args_str)
    server_type, model = server_config["server_type"], server_config["model"]
    n_servers = server_config.get("n_servers", 1)
    port = server_config["port"]

    # Get master nodes for each server (Het 0 is the ray server, so servers start at het group 1)
    master_nodes = []
    for i in range(n_servers):
        het_group = i + 1  # Het 0 is the ray server
        env_var = f"SLURM_MASTER_NODE_HET_GROUP_{het_group}"
        master_node = os.environ[env_var]
        master_nodes.append(master_node)

    LOG.info("[proof_judge] JUDGE_SERVER_ARGS: %s", server_args_str)

    return model, server_type, port, master_nodes


def _wait_for_server(server_address: str, *, interval_seconds: float = 3.0) -> None:
    """Wait until the external judge server starts accepting TCP connections."""
    host, port_str = server_address.rsplit(":", 1)
    port = int(port_str)
    LOG.info("Waiting for external judge server at %s", server_address)
    while True:
        try:
            with socket.create_connection((host, port), timeout=5):
                pass
            break
        except OSError:
            time.sleep(interval_seconds)
    LOG.info("External judge server at %s is ready", server_address)


SOLUTION_HEADER = "## Solution"
SELF_EVAL_HEADER = "## Self Evaluation"


def extract_boxed_score(text: str) -> Optional[float]:
    """Extract the last \\boxed{...} score. Returns 0, 0.5, or 1, or None."""
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


def validate_text_length(text: str, *, name: str, max_chars: int) -> Optional[dict[str, Any]]:
    text_length = len(text)
    if text_length <= max_chars:
        return None
    return {
        "r_format": 0.0,
        "reason": f"{name}_too_long",
        f"{name}_length": text_length,
        f"{name}_max_length": max_chars,
    }


def extract_generated_token_count(response_or_data: Any) -> int:
    usage = getattr(response_or_data, "usage", None)
    if usage is None and isinstance(response_or_data, dict) and "usage" in response_or_data:
        usage = response_or_data["usage"]
    if usage is None:
        return 0

    if isinstance(usage, dict):
        if "output_tokens" in usage and usage["output_tokens"] is not None:
            return int(usage["output_tokens"])
        if "completion_tokens" in usage and usage["completion_tokens"] is not None:
            return int(usage["completion_tokens"])
        return 0

    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is not None:
        return int(output_tokens)

    completion_tokens = getattr(usage, "completion_tokens", None)
    if completion_tokens is not None:
        return int(completion_tokens)

    return 0


def parse_response(
    response: str, assert_think_end: bool = False
) -> tuple[Optional[tuple[str, str, float]], Optional[str]]:
    """Parse policy response into (proof, self_analysis, s_prime). Returns (None, reason) on failure."""
    if assert_think_end and "</think>" not in response:
        return None, "missing_think_end"
    response = response.split("</think>")[-1].strip()
    if SOLUTION_HEADER not in response:
        return None, "missing_solution_header"
    after_solution = response.split(SOLUTION_HEADER, 1)[1]
    if SELF_EVAL_HEADER not in after_solution:
        return None, "missing_self_eval_header"
    proof, self_eval = after_solution.split(SELF_EVAL_HEADER, 1)
    proof = proof.strip()
    self_eval = self_eval.strip()
    s_prime = extract_boxed_score(self_eval)
    if s_prime is None:
        return None, "invalid_boxed_score"
    return (proof, self_eval, s_prime), None


class ProofWithJudgeResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_model_name: str = ""
    alpha: float = 1.0
    beta: float = 0.0
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 100000
    zero_reward_incorrect_groups: bool = False
    expected_group_size: int = -1
    assert_think_end: bool = False

    def model_post_init(self, __context: Any) -> None:
        if self.zero_reward_incorrect_groups and self.expected_group_size <= 0:
            raise ValueError("expected_group_size must be > 0 when zero_reward_incorrect_groups is enabled")


class ProofWithJudgeVerifyRequest(BaseVerifyRequest):
    problem: str = ""


class IncorrectGroupCoordinator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    verifier_rewards: list[float] = Field(default_factory=list)
    zero_out_group_reward: Optional[bool] = None
    event: Event = Field(default_factory=Event)


class ProofWithJudgeResourcesServer(SimpleResourcesServer):
    config: ProofWithJudgeResourcesServerConfig

    _ext_clients: Optional[list] = PrivateAttr(default=None)
    _ext_model: Optional[str] = PrivateAttr(default=None)
    _ext_rr_counter: int = PrivateAttr(default=0)
    _ext_init_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _log_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _incorrect_group_coordinators: dict[str, IncorrectGroupCoordinator] = PrivateAttr(
        default_factory=lambda: defaultdict(IncorrectGroupCoordinator)
    )

    async def verify(self, body: ProofWithJudgeVerifyRequest) -> BaseVerifyResponse:
        problem = getattr(body, "problem", "") or (body.model_dump().get("problem") or "")
        full_response = self._extract_assistant_text(body.response)
        if not full_response:
            reward, details = 0.0, {"r_format": 0.0, "reason": "empty_response", "judge_generated_tokens": 0}
        else:
            reward, details = await self._judge_single(problem, full_response)
        reward, details = await self._maybe_zero_incorrect_group_reward(
            problem=problem, reward=reward, details=details
        )
        if LOG_JSONL_PATH:
            await self._append_log_jsonl(
                log_path=LOG_JSONL_PATH,
                problem=problem,
                generated_sequence=full_response,
                reward=reward,
                details=details,
            )
        return BaseVerifyResponse(**body.model_dump(), reward=reward)

    async def _append_log_jsonl(
        self,
        *,
        log_path: str,
        problem: str,
        generated_sequence: str,
        reward: float,
        details: dict[str, Any],
    ) -> None:
        if self._log_lock is None:
            self._log_lock = asyncio.Lock()
        try:
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "problem": problem,
                "generated_sequence": generated_sequence,
                "reward": reward,
                **details,
            }
            async with self._log_lock:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            LOG.warning("[proof_judge] Failed to append log_jsonl %s: %s", log_path, e)

    async def _maybe_zero_incorrect_group_reward(
        self, *, problem: str, reward: float, details: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        if not self.config.zero_reward_incorrect_groups:
            return reward, details
        if not problem:
            raise ValueError("problem must be set when zero_reward_incorrect_groups is enabled")

        verifier_reward = float(details["r_y"]) if "r_y" in details else 0.0
        coordinator = self._incorrect_group_coordinators[problem]
        coordinator.verifier_rewards.append(verifier_reward)

        if len(coordinator.verifier_rewards) == self.config.expected_group_size:
            coordinator.zero_out_group_reward = all(r_y == 0.0 for r_y in coordinator.verifier_rewards)
            self._incorrect_group_coordinators.pop(problem)
            coordinator.event.set()
        else:
            await coordinator.event.wait()

        assert coordinator.zero_out_group_reward is not None
        if not coordinator.zero_out_group_reward:
            return reward, {**details, "group_all_r_y_zero": False}

        return 0.0, {
            **details,
            "original_reward": reward,
            "group_all_r_y_zero": True,
        }

    def _extract_assistant_text(self, response: Any) -> str:
        if not response or not getattr(response, "output", None):
            return ""
        parts = []
        for out in response.output:
            if getattr(out, "type", None) != "message":
                continue
            if getattr(out, "role", None) != "assistant":
                continue
            for c in getattr(out, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    parts.append(getattr(c, "text", "") or "")
        return "".join(parts)

    # ------------------------------------------------------------------
    # External judge via JUDGE_SERVER_ARGS (AsyncOpenAI, round-robin)
    # ------------------------------------------------------------------

    async def _init_external_clients(self) -> None:
        """Lazily create AsyncOpenAI clients for external judge servers."""
        from openai import AsyncOpenAI

        cfg = _get_judge_client_config()
        if cfg is None:
            raise RuntimeError("_init_external_clients called but JUDGE_SERVER_ARGS is not set")
        model, _server_type, port, master_nodes = cfg
        clients = []
        for node in master_nodes:
            _wait_for_server(f"{node}:{port}")
            base_url = f"http://{node}:{port}/v1"
            client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=60 * 60 * 4)
            clients.append(client)
        if not clients:
            raise RuntimeError("No external judge clients were initialized")
        self._ext_model = model
        self._ext_clients = clients
        LOG.info("[proof_judge] Initialized %d external judge clients", len(self._ext_clients))

    async def _ensure_external_clients(self) -> None:
        if self._ext_clients is not None:
            return
        if self._ext_init_lock is None:
            self._ext_init_lock = asyncio.Lock()
        async with self._ext_init_lock:
            if self._ext_clients is None:
                await self._init_external_clients()

    def _next_ext_client(self):
        if not self._ext_clients:
            raise RuntimeError("External judge clients are not initialized")
        client = self._ext_clients[self._ext_rr_counter % len(self._ext_clients)]
        self._ext_rr_counter += 1
        return client

    async def _call_judge_external(self, user_content: str) -> tuple[str, int]:
        """Call external judge via OpenAI-compatible /v1/chat/completions."""
        await self._ensure_external_clients()

        client = self._next_ext_client()
        response = await client.chat.completions.create(
            model=self._ext_model,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        content = response.choices[0].message.content
        return (content.strip() if content else "", extract_generated_token_count(response))

    # ------------------------------------------------------------------
    # Gym-internal judge via /v1/responses
    # ------------------------------------------------------------------

    async def _call_judge_internal(self, user_content: str) -> tuple[str, int]:
        """Call judge through Gym's server_client (judge model managed by Gym/Ray)."""
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
            LOG.warning("[proof_judge] Judge HTTP %s (server_name=%s)", resp.status, server_name)
        await raise_for_status(resp)
        data = await get_response_json(resp)
        generated_tokens = extract_generated_token_count(data)
        judge_resp = NeMoGymResponse.model_validate(data)
        return self._extract_assistant_text(judge_resp), generated_tokens

    # ------------------------------------------------------------------
    # Unified dispatcher
    # ------------------------------------------------------------------

    async def _call_judge(self, user_content: str) -> tuple[str, int]:
        """Route to external (JUDGE_SERVER_ARGS) or internal (Gym /v1/responses) judge."""
        if os.environ.get("JUDGE_SERVER_ARGS"):
            return await self._call_judge_external(user_content)
        return await self._call_judge_internal(user_content)

    async def _call_verifier_and_meta_verifier(
        self,
        *,
        problem: str,
        proof: str,
        self_analysis: str,
        beta: float,
    ) -> tuple[tuple[str, int], Optional[tuple[str, int]]]:
        verifier_prompt = VERIFIER_PROMPT_TEMPLATE.format(problem=problem, proof=proof)
        if beta == 0:
            verifier_result = await self._call_judge(verifier_prompt)
            return verifier_result, None

        meta_prompt = META_VERIFIER_PROMPT_TEMPLATE.format(problem=problem, proof=proof, proof_analysis=self_analysis)
        verifier_result, meta_result = await asyncio.gather(
            self._call_judge(verifier_prompt),
            self._call_judge(meta_prompt),
        )
        return verifier_result, meta_result

    async def _judge_single(self, problem: str, full_response: str) -> tuple[float, dict[str, Any]]:
        alpha = self.config.alpha
        beta = self.config.beta

        parsed, reason = parse_response(full_response, assert_think_end=self.config.assert_think_end)
        if parsed is None:
            return 0.0, {"r_format": 0.0, "reason": reason, "judge_generated_tokens": 0}

        proof, self_analysis, s_prime = parsed
        proof_length_error = validate_text_length(proof, name="proof", max_chars=MAX_PROOF_CHARS)
        if proof_length_error is not None:
            proof_length_error["judge_generated_tokens"] = 0
            return 0.0, proof_length_error
        self_verification_length_error = validate_text_length(
            self_analysis, name="self_verification", max_chars=MAX_VERIFICATION_CHARS
        )
        if self_verification_length_error is not None:
            self_verification_length_error["judge_generated_tokens"] = 0
            return 0.0, self_verification_length_error

        verifier_result, meta_result = await self._call_verifier_and_meta_verifier(
            problem=problem,
            proof=proof,
            self_analysis=self_analysis,
            beta=beta,
        )
        verifier_response, verifier_generated_tokens = verifier_result
        r_y = extract_boxed_score(verifier_response) or 0.0

        if beta == 0:
            return alpha * r_y, {
                "r_y": r_y,
                "s_prime": s_prime,
                "judge_generated_tokens": verifier_generated_tokens,
                "verifier_generated_tokens": verifier_generated_tokens,
                "verifier_response": verifier_response,
            }

        assert meta_result is not None
        meta_response, meta_generated_tokens = meta_result
        r_meta = extract_boxed_score(meta_response) or 0.0
        r_z = (1.0 - abs(s_prime - r_y)) * r_meta
        return alpha * r_y + beta * r_z, {
            "judge_generated_tokens": verifier_generated_tokens + meta_generated_tokens,
            "r_y": r_y,
            "r_meta": r_meta,
            "s_prime": s_prime,
            "verifier_generated_tokens": verifier_generated_tokens,
            "meta_generated_tokens": meta_generated_tokens,
            "verifier_response": verifier_response,
            "meta_response": meta_response,
        }


if __name__ == "__main__":
    ProofWithJudgeResourcesServer.run_webserver()
