# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

LOG = logging.getLogger(__name__)

LOG_JSONL_PATH = os.environ.get("PROOF_GENSELECT_LOG_JSONL_PATH", None)


def extract_best_solution(response: str, assert_think_end: bool = False) -> tuple[Optional[int], Optional[str]]:
    if assert_think_end and "</think>" not in response:
        return None, "missing_think_end"

    response = response.split("</think>")[-1].strip()
    if "<best_solution>" not in response or "</best_solution>" not in response:
        return None, "missing_best_solution_tag"

    try:
        selected_index = response.split("<best_solution>", 1)[1].split("</best_solution>", 1)[0].strip()
        best_solution = int(selected_index)
    except ValueError:
        return None, "invalid_best_solution_value"

    if best_solution not in (1, 2):
        return None, "best_solution_out_of_range"

    return best_solution, None


class ProofGenSelectResourcesServerConfig(BaseResourcesServerConfig):
    assert_think_end: bool = False


class ProofGenSelectVerifyRequest(BaseVerifyRequest):
    problem: str
    proof_1: str
    proof_2: str
    correct_index: int
    score_1: Optional[float] = None
    score_2: Optional[float] = None


class ProofGenSelectResourcesServer(SimpleResourcesServer):
    config: ProofGenSelectResourcesServerConfig

    _log_lock: Optional[Any] = PrivateAttr(default=None)

    async def verify(self, body: ProofGenSelectVerifyRequest) -> BaseVerifyResponse:
        full_response = self._extract_assistant_text(body.response)
        if not full_response:
            return BaseVerifyResponse(**body.model_dump(), reward=0.0)

        selected_index, reason = extract_best_solution(
            full_response, assert_think_end=self.config.assert_think_end
        )
        reward = 1.0 if selected_index == body.correct_index else 0.0
        details = {
            "selected_index": selected_index,
            "correct_index": body.correct_index,
        }
        if reason is not None:
            details["reason"] = reason

        if LOG_JSONL_PATH:
            await self._append_log_jsonl(
                log_path=LOG_JSONL_PATH,
                problem=body.problem,
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
        import asyncio

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
                with open(log_path, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            LOG.warning("[proof_genselect] Failed to append log_jsonl %s: %s", log_path, exc)

    def _extract_assistant_text(self, response: Any) -> str:
        if not response or not getattr(response, "output", None):
            return ""

        parts = []
        for out in response.output:
            if getattr(out, "type", None) != "message":
                continue
            if getattr(out, "role", None) != "assistant":
                continue
            for content in getattr(out, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    parts.append(getattr(content, "text", "") or "")
        return "".join(parts)


if __name__ == "__main__":
    ProofGenSelectResourcesServer.run_webserver()
