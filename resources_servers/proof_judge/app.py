# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Proof judge resource server: verifier + meta-verifier reward for theorem proving.
# Ported from nemo_skills JudgeEnvironment (DeepSeek Math style reward).
import logging
from typing import Any, Optional

from pydantic import BaseModel

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
... // Your evaluation here.

Based on my evaluation, the final overall score should be:
\\boxed{{...}} // where ... should be the final overall score (0, 0.5, or 1, and nothing else)

---

Here is your task input:

## Problem
{problem}

## Solution
{proof}"""

# Meta-verifier prompt template (from openprover meta-verifier.yaml)
META_VERIFIER_PROMPT_TEMPLATE = """You are given a "problem", "solution", and "solution evaluation", and you need to assess the whether this "solution evaluation" is reasonable.

[... scoring rules ...]

Your output should follow the format below:

Here is my analysis of the "solution evaluation":
... // Your analysis here.

Based on my analysis, I will rate the "solution evaluation" as:
\\boxed{{...}} // where ... should be a numerical rating (0, 0.5, or 1, and nothing else).

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

    async def verify(self, body: ProofJudgeVerifyRequest) -> BaseVerifyResponse:
        problem = getattr(body, "problem", "") or (body.model_dump().get("problem") or "")
        full_response = self._extract_assistant_text(body.response)
        if not full_response:
            return BaseVerifyResponse(**body.model_dump(), reward=0.0)

        reward, _ = await self._judge_single(problem, full_response)
        return BaseVerifyResponse(**body.model_dump(), reward=reward)

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

    async def _call_judge(self, user_content: str) -> str:
        from nemo_gym.server_utils import raise_for_status

        model = self.config.judge_model_name or self.config.judge_model_server.name
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content=user_content)],
            model=model,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_output_tokens=self.config.max_tokens,
        )
        resp = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=params.model_dump(),
        )
        await raise_for_status(resp)
        data = await get_response_json(resp)
        judge_resp = NeMoGymResponse.model_validate(data)
        return self._extract_assistant_text(judge_resp)

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
            return alpha * r_y, {"r_y": r_y, "s_prime": s_prime}

        meta_prompt = META_VERIFIER_PROMPT_TEMPLATE.format(
            problem=problem, proof=proof, proof_analysis=self_analysis
        )
        meta_response = await self._call_judge(meta_prompt)
        r_meta = extract_boxed_score(meta_response) or 0.0
        r_z = (1.0 - abs(s_prime - r_y)) * r_meta
        return alpha * r_y + beta * r_z, {"r_y": r_y, "r_meta": r_meta, "s_prime": s_prime}


if __name__ == "__main__":
    ProofJudgeResourcesServer.run_webserver()
