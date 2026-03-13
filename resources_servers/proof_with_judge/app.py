# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Proof-with-judge resource server: verifier + meta-verifier reward for theorem proving.
# Combined env that handles both data formatting (via prepare_data.py) and verification.
# Reuses the same judge logic as proof_judge.
#
# Supports two judge paths:
#   1. Gym-internal: calls judge_model_server via /v1/responses (judge managed by Gym/Ray)
#   2. External: reads JUDGE_SERVER_ARGS env var → AsyncOpenAI /v1/chat/completions
#      (judge servers managed by nemo-skill pipeline as SLURM het groups)
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import get_response_json

LOG = logging.getLogger(__name__)

LOG_JSONL_PATH = os.environ.get("PROOF_JUDGE_LOG_JSONL_PATH", None)

PROMPT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"


def _load_prompt_template(filename: str) -> str:
    """Load the 'user' field from a prompt YAML file in prompt_templates/."""
    with open(PROMPT_TEMPLATES_DIR / filename) as f:
        return yaml.safe_load(f)["user"]


VERIFIER_PROMPT_TEMPLATE = _load_prompt_template("verifier.yaml")
META_VERIFIER_PROMPT_TEMPLATE = _load_prompt_template("meta-verifier.yaml")


# ---------------------------------------------------------------------------
#  External judge helpers (from math-with-judge-het pattern)
# ---------------------------------------------------------------------------

def _get_judge_client_config() -> Optional[tuple[str, str, int, list[str]]]:
    """Read judge server addresses from JUDGE_SERVER_ARGS env var.

    Set by nemo-skill pipeline when launching heterogeneous SLURM jobs,
    or manually for local testing.

    Returns (model, server_type, port, master_nodes) or None when not set.

    Supports two addressing modes:
      - "host" field: direct address (e.g. local testing without SLURM)
      - SLURM_MASTER_NODE_HET_GROUP_* env vars: cluster het jobs
    """
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
            LOG.info("[proof_with_judge] No SLURM het group vars found, falling back to localhost")

    LOG.info(
        "[proof_with_judge] External judge: model=%s port=%s n_servers=%s master_nodes[0]=%s",
        model, port, len(master_nodes), master_nodes[0] if master_nodes else None,
    )
    return model, server_type, port, master_nodes


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


def parse_response(
    response: str, assert_think_end: bool = False
) -> tuple[Optional[tuple[str, str, float]], Optional[str]]:
    """Parse policy response into (proof, self_analysis, s_prime). Returns (None, reason) on failure."""
    if assert_think_end and "</think>" not in response:
        return None, "missing_think_end"
    response = response.split("</think>")[-1].strip()
    if SOLUTION_HEADER not in response:
        return None, f"missing_solution_header"
    after_solution = response.split(SOLUTION_HEADER, 1)[1]
    if SELF_EVAL_HEADER not in after_solution:
        return None, f"missing_self_eval_header"
    proof, self_eval = after_solution.split(SELF_EVAL_HEADER, 1)
    proof = proof.strip()
    self_eval = self_eval.strip()
    s_prime = extract_boxed_score(self_eval)
    if s_prime is None:
        return None, f"invalid_boxed_score"
    return (proof, self_eval, s_prime), None


class ProofWithJudgeResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_model_name: str = ""
    alpha: float = 1.0
    beta: float = 0.0
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 100000
    assert_think_end: bool = False


class ProofWithJudgeVerifyRequest(BaseVerifyRequest):
    problem: str = ""


class ProofWithJudgeResourcesServer(SimpleResourcesServer):
    config: ProofWithJudgeResourcesServerConfig

    _ext_clients: Optional[list] = PrivateAttr(default=None)
    _ext_model: Optional[str] = PrivateAttr(default=None)
    _ext_rr_counter: int = PrivateAttr(default=0)
    _log_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)

    async def verify(self, body: ProofWithJudgeVerifyRequest) -> BaseVerifyResponse:
        problem = getattr(body, "problem", "") or (body.model_dump().get("problem") or "")
        full_response = self._extract_assistant_text(body.response)
        if not full_response:
            return BaseVerifyResponse(**body.model_dump(), reward=0.0)

        reward, details = await self._judge_single(problem, full_response)
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
            LOG.warning("[proof_with_judge] Failed to append log_jsonl %s: %s", log_path, e)

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
    #  External judge via JUDGE_SERVER_ARGS (AsyncOpenAI, round-robin)
    # ------------------------------------------------------------------

    def _init_external_clients(self) -> None:
        """Lazily create AsyncOpenAI clients for external judge servers."""
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
        LOG.info("[proof_with_judge] Initialized %d external judge clients", len(self._ext_clients))

    def _next_ext_client(self):
        client = self._ext_clients[self._ext_rr_counter % len(self._ext_clients)]
        self._ext_rr_counter += 1
        return client

    async def _call_judge_external(self, user_content: str) -> str:
        """Call external judge via OpenAI-compatible /v1/chat/completions."""
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

    # ------------------------------------------------------------------
    #  Gym-internal judge via /v1/responses
    # ------------------------------------------------------------------

    async def _call_judge_internal(self, user_content: str) -> str:
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
            LOG.warning("[proof_with_judge] Judge HTTP %s (server_name=%s)", resp.status, server_name)
        await raise_for_status(resp)
        data = await get_response_json(resp)
        judge_resp = NeMoGymResponse.model_validate(data)
        return self._extract_assistant_text(judge_resp)

    # ------------------------------------------------------------------
    #  Unified dispatcher
    # ------------------------------------------------------------------

    async def _call_judge(self, user_content: str) -> str:
        """Route to external (JUDGE_SERVER_ARGS) or internal (Gym /v1/responses) judge."""
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


if __name__ == "__main__":
    ProofWithJudgeResourcesServer.run_webserver()
