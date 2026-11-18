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

import asyncio
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from resources_servers.ether0.setup_ether0 import ensure_ether0


ensure_ether0()

from ether0.model_prompts import extract_answer_loose  # noqa: E402
from ether0.models import RewardFunctionInfo  # noqa: E402
from pydantic import Field, PrivateAttr

from nemo_gym.core import VERIFIER_ERROR
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture
from resources_servers.ether0 import local_oracles  # noqa: E402
from resources_servers.ether0.graders import EVAL_FUNCTIONS  # noqa: E402
from resources_servers.ether0.verifier_fixture import create_server  # noqa: E402


logger = logging.getLogger(__name__)

# No oracle was configured, so nothing failed; the masked zero is not a measurement.
ORACLE_NOT_CONFIGURED = "ether0:oracle_not_configured"


class Ether0ResourcesServerConfig(BaseResourcesServerConfig):
    # Grade the oracle-backed rewards (sol_eval, rxn_forward) in this process; see local_oracles.py.
    # Ignored when ETHER0_REMOTES_API_BASE_URL points at an ether0-serve.
    local_oracles: bool = True
    moltrans_model_path: Optional[str] = None
    zinc20_bloom_path: Optional[str] = None
    oracle_max_concurrency: int = Field(
        default=8, ge=1, description="Maximum simultaneous oracle-backed grades per worker."
    )


class Ether0RunRequest(BaseRunRequest):
    verifier_metadata: Optional[dict[str, Any]] = None


class Ether0VerifyRequest(Ether0RunRequest, BaseVerifyRequest):
    pass


class Ether0VerifyResponse(BaseVerifyResponse):
    extracted_answer: Optional[str] = None
    eval_function: Optional[str] = None
    problem_type: Optional[str] = None


class Ether0ResourcesServer(SimpleResourcesServer):
    # Each verification runs one ether0 reward function on the answer: there is no session state to export,
    # and a checkpoint can re-run a verification after a crash instead of waiting for it.
    checkpoint_mode = "stateless"
    checkpoint_verify = "replay"
    config: Ether0ResourcesServerConfig
    _oracle_semaphore: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        self._oracle_semaphore = asyncio.Semaphore(self.config.oracle_max_concurrency)
        if self.config.local_oracles and not os.environ.get(local_oracles.REMOTE_ORACLES_ENV_VAR):
            local_oracles.install(self.config.moltrans_model_path, self.config.zinc20_bloom_path)
        super().model_post_init(context)

    async def verify(self, body: Ether0VerifyRequest) -> Ether0VerifyResponse:
        text = _extract_last_assistant_text(body)

        meta = body.verifier_metadata or {}
        solution_str = meta.get("solution", "")
        problem_type = meta.get("problem_type", "")

        try:
            reward_info = RewardFunctionInfo.model_validate(solution_str)
        except Exception:
            logger.warning("Malformed solution string: %r", solution_str)
            return _response(body, 0.0, None, None, problem_type)

        eval_fn_name = reward_info.fxn_name
        answer_info = reward_info.answer_info

        text = text.replace("<|answer_start|>", "<answer>").replace("<|answer_end|>", "</answer>")
        answer = _extract_answer_multi_format(text)
        if answer is None:
            return _response(body, 0.0, None, eval_fn_name, problem_type)

        choices = meta.get("choices", {})
        if choices and len(answer) == 1 and answer.isalpha():
            answer = choices.get(answer.upper(), answer)

        eval_fn = EVAL_FUNCTIONS.get(eval_fn_name)
        if eval_fn is None:
            logger.warning("Unknown eval function %r", eval_fn_name)
            return _response(body, 0.0, answer, eval_fn_name, problem_type)

        if eval_fn_name in local_oracles.ORACLE_EVAL_FUNCTIONS and not (
            self.config.local_oracles or os.environ.get(local_oracles.REMOTE_ORACLES_ENV_VAR)
        ):
            return _response(
                body,
                0.0,
                answer,
                eval_fn_name,
                problem_type,
                mask_sample=True,
                failure_kind=ORACLE_NOT_CONFIGURED,
                failure_reason=f"Set local_oracles or {local_oracles.REMOTE_ORACLES_ENV_VAR} to score {eval_fn_name}",
            )

        try:
            if eval_fn_name in local_oracles.ORACLE_EVAL_FUNCTIONS:
                reward = await self._grade_oracle(eval_fn, answer, answer_info)
            else:
                reward = eval_fn(answer, answer_info)
        except local_oracles.OracleError as error:
            logger.exception("Oracle failed in %r for problem_type=%r", eval_fn_name, problem_type)
            return _response(
                body,
                0.0,
                answer,
                eval_fn_name,
                problem_type,
                mask_sample=True,
                failure_kind=VERIFIER_ERROR,
                failure_reason=str(error),
            )
        except Exception:
            logger.exception("Error in %r for problem_type=%r", eval_fn_name, problem_type)
            reward = 0.0

        return _response(body, reward, answer, eval_fn_name, problem_type)

    async def _grade_oracle(self, eval_fn: Callable[[str, str], float], answer: str, answer_info: str) -> float:
        async with self._oracle_semaphore:
            # A cancelled HTTP request cannot stop a running thread. Keep its slot until the
            # grade finishes, including across repeated cancellation, so the limit remains real.
            task = asyncio.create_task(asyncio.to_thread(eval_fn, answer, answer_info))
            cancelled = False
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
                except Exception:
                    if not cancelled:
                        raise
            if cancelled:
                if not task.cancelled():
                    task.exception()
                raise asyncio.CancelledError
            return task.result()


def _response(
    body: Ether0VerifyRequest,
    reward: float,
    extracted_answer: Optional[str],
    eval_function: Optional[str],
    problem_type: str,
    **failure: Any,
) -> Ether0VerifyResponse:
    return Ether0VerifyResponse(
        **body.model_dump(exclude={"extracted_answer", "eval_function", "problem_type"}),
        reward=reward,
        extracted_answer=extracted_answer,
        eval_function=eval_function,
        problem_type=problem_type,
        **failure,
    )


_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")
_ANSWER_LETTER_RE = re.compile(r"Answer\s*:\s*([A-Za-z])\s*$", re.MULTILINE)


def _extract_answer_multi_format(text: str) -> str | None:
    # <answer> tags
    ans = extract_answer_loose(text).strip()
    if ans:
        return ans
    # \boxed{}, rightmost
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip() or None
    # Answer: LETTER, rightmost
    matches = _ANSWER_LETTER_RE.findall(text)
    if matches:
        return matches[-1].strip() or None
    return None


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    for o in body.response.output:
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_server,
    request_model=Ether0VerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    Ether0ResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = Ether0ResourcesServer.run_webserver()  # noqa: F401
