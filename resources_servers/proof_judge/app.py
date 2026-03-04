# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Proof judge resource server: verifier + meta-verifier reward for theorem proving.
# Ported from nemo_skills JudgeEnvironment (DeepSeek Math style reward).
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
from typing import Any, Optional

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
            LOG.info("[proof_judge] No SLURM het group vars found, falling back to localhost")

    LOG.info(
        "[proof_judge] External judge: model=%s port=%s n_servers=%s master_nodes[0]=%s",
        model, port, len(master_nodes), master_nodes[0] if master_nodes else None,
    )
    return model, server_type, port, master_nodes

SOLUTION_HEADER = "## Solution"
SELF_EVAL_HEADER = "## Self Evaluation"

# Verifier prompt template (from openprover verifier.yaml)
VERIFIER_PROMPT_TEMPLATE = """## Instruction

Your task is to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0
- Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1

Please carefully reason out and analyze the quality of the solution below, and in your final response present a detailed evaluation of the solution's quality followed by your score. Therefore, your response should be in the following format:

Here is my evaluation of the solution:
... // Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution.

Based on my evaluation, the final overall score should be:
\\boxed{{...}} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the above criteria

---

Here is your task input:

## Problem
{problem}

## Solution
{proof}"""

# Meta-verifier prompt template (from openprover meta-verifier.yaml)
META_VERIFIER_PROMPT_TEMPLATE = """You are given a "problem", "solution", and "solution evaluation", and you need to assess the whether this "solution evaluation" is reasonable.

First, "solution evaluation" is generated to evaluate the quality of the "solution", by prompting a verifier with the rules below (these are not your rules):

```
Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0

Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1
```

Next, I will introduce the rules for you to analyze the quality of the "solution evaluation":

1. Your task is to analyze the "solution evaluation". You do not need to solve the "problem", nor do you need to strictly assess whether the "solution" is accurate. Your only task is to strictly follow the rules below to evaluate whether the "solution evaluation" is reasonable.

2. You need to analyze the content of the "solution evaluation" from three aspects:

Step Restatement: In the "solution evaluation", certain behaviors of the "solution" may be restated. You need to return to the original text of the "solution" and check whether the "solution" actually has these behaviors mentioned in the "solution evaluation".

Defect Analysis: "solution evaluation" may point out errors or defects in the "solution". You need to carefully analyze whether the mentioned errors and defects are indeed valid.

Expression Analysis: Whether the "solution evaluation"'s expressions are accurate.

Score Analysis: Whether the final score given by the "solution evaluation" matches the defects it found. You need to analyze according to the scoring rules given above.

3. The most important part is **defect analysis**: In this part, your core task is to check whether the errors or defects of the "solution" pointed out in the "solution evaluation" are reasonable. In other words, any positive components about the "solution" in the "solution evaluation", regardless of whether they are reasonable, are not within your evaluation scope.

- For example: If the "solution evaluation" says that a certain conclusion in the "solution" is correct, but actually this conclusion is incorrect, then you do not need to care about this point. All parts that the "solution evaluation" considers correct do not belong to your evaluation scope.
- Specifically: If the "solution evaluation" believes that the "solution" is completely accurate and has not found any errors or defects, then regardless of whether the "solution" itself is actually accurate, even if there are obvious errors, you should still consider its analysis of errors to be reasonable.

**Importantly**, for defects found by the "solution evaluation", you need to analyze two points simultaneously:

- whether this defect actually exists
- whether the "solution evaluation"'s analysis of this defect is accurate

These two aspects constitute the analysis of defects.

4. About **expression analysis**, if there are certain expression errors in the "solution evaluation", even minor errors in details, you need to identify them. However, please note that identifying incorrect steps in the "solution" as correct steps does not constitute an **expression error**.

In practice, expression errors include but are not limited to:

- If the "solution evaluation" identifies some reasoning step(s) in the "solution" as incorrect, then it cannot further indicate that subsequent conclusion(s) depending on those reasoning step(s) are wrong, but can only indicate that subsequent conclusion(s) are "not rigorously demonstrated."
- Typos and calculation errors made by "solution evaluation"
- Inaccurate restatement of content from "solution"

5. Finally, you need to present your analysis of the "solution evaluation" in your output and also rate its quality based on the rules below:

First, if there is at least one unreasonable defect among the defects found by the "solution evaluation", then you only need to do **defect analysis**:

- If all defects found by the "solution evaluation" are unreasonable, then you should rate it with \\(0\\)
- If some defects found by the "solution evaluation" are reasonable and some are unreasonable, then your rating should be \\(0.5\\)

Next, if the "solution evaluation" points out no errors or defects, or all defects found by the evaluation are reasonable, then you should do the following things:

- Analyze whether "expression errors" exist in the "solution evaluation" (**expression analysis**) or whether "solution evaluation" gives a wrong score according to the rules for "solution evaluation" (**score analysis**). If yes, you should rate the "solution evaluation" with \\(0.5\\); if no, your rating should be \\(1\\)

Your output should follow the format below:

Here is my analysis of the "solution evaluation":
... // Your analysis here.

Based on my analysis, I will rate the "solution evaluation" as:
\\boxed{{...}} // where ... should be a numerical rating of the "solution evaluation" (0, 0.5, or 1, and nothing else) based on the criteria above.

---

Here is your task input:

## Problem
{problem}

## Solution
{proof}

## Solution Evaluation
{proof_analysis}"""


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


class ProofJudgeResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_model_name: str = ""  # model name/path for /v1/responses (e.g. same as judge server model)
    alpha: float = 1.0
    beta: float = 0.0
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 100000
    assert_think_end: bool = False


class ProofJudgeVerifyRequest(BaseVerifyRequest):
    problem: str = ""


class ProofJudgeResourcesServer(SimpleResourcesServer):
    config: ProofJudgeResourcesServerConfig

    _ext_clients: Optional[list] = PrivateAttr(default=None)
    _ext_model: Optional[str] = PrivateAttr(default=None)
    _ext_rr_counter: int = PrivateAttr(default=0)
    _log_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)

    async def verify(self, body: ProofJudgeVerifyRequest) -> BaseVerifyResponse:
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
            LOG.warning("[proof_judge] Failed to append log_jsonl %s: %s", log_path, e)

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
        LOG.info("[proof_judge] Initialized %d external judge clients", len(self._ext_clients))

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
            LOG.warning("[proof_judge] Judge HTTP %s (server_name=%s)", resp.status, server_name)
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
    ProofJudgeResourcesServer.run_webserver()
