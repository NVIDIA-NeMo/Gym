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
import asyncio
import logging
import re
from typing import Any, Optional

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import raise_for_status, request


LOG = logging.getLogger(__name__)


class CritPtResourcesServerConfig(BaseResourcesServerConfig):
    api_url: str = "https://artificialanalysis.ai/api/v2/critpt/evaluate"
    api_key: str
    # AA API requires submissions for every hosted problem in PUBLIC mode (currently 70).
    # The server buffers verify() calls until batch_size unique problem_ids accumulate,
    # then fires one API call and distributes the aggregate accuracy to all waiters.
    batch_size: int = 70


class CritPtRunRequest(BaseRunRequest):
    problem_id: str


class CritPtVerifyRequest(CritPtRunRequest, BaseVerifyRequest):
    pass


class CritPtVerifyResponse(BaseVerifyResponse):
    problem_id: str
    accuracy: float
    timeout_rate: float


class CritPtResourcesServer(SimpleResourcesServer):
    config: CritPtResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._lock = asyncio.Lock()
        # Active batch: {"future": asyncio.Future, "submissions": dict[problem_id, submission]}.
        # Reset to None once batch_size is reached; new calls start a fresh batch.
        self._current_batch: Optional[dict] = None

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.get("/status")(self.status)
        return app

    async def status(self) -> dict:
        """Return the current batch fill (buffered submissions vs batch_size). Read-only; no lock."""
        batch = self._current_batch
        buffered = len(batch["submissions"]) if batch is not None else 0
        return {"buffered": buffered, "batch_size": self.config.batch_size}

    async def verify(self, body: CritPtVerifyRequest) -> CritPtVerifyResponse:
        code = _extract_code(_extract_output_text(body))
        submission = {
            "problem_id": body.problem_id,
            "generated_code": f"```python\n{code}\n```" if code else "```python\n```",
            "model": "unknown",
            "generation_config": {},
        }

        async with self._lock:
            if self._current_batch is None:
                self._current_batch = {
                    "future": asyncio.get_running_loop().create_future(),
                    "submissions": {},
                }
            batch = self._current_batch
            batch["submissions"][body.problem_id] = submission
            future = batch["future"]
            LOG.warning(
                "CritPt verify: %d/%d submissions buffered (problem_id=%s)",
                len(batch["submissions"]),
                self.config.batch_size,
                body.problem_id,
            )

            ready_to_fire = len(batch["submissions"]) >= self.config.batch_size
            if ready_to_fire:
                submissions_snapshot = list(batch["submissions"].values())
                self._current_batch = None  # next caller starts a fresh batch
            else:
                submissions_snapshot = None

        if ready_to_fire:
            LOG.info("CritPt batch full (%d submissions); firing AA API.", len(submissions_snapshot))
            try:
                result = await _call_api(self.config.api_url, self.config.api_key, submissions_snapshot)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        result = await future
        accuracy = result["accuracy"]
        timeout_rate = result.get("timeout_rate", 0.0)
        # AA API returns only an aggregate accuracy. Following nemo-skills, every rollout in the
        # batch receives the same aggregate as its reward; pass@1 across the dataset then equals
        # the aggregate accuracy.
        return CritPtVerifyResponse(
            **body.model_dump(),
            reward=accuracy,
            accuracy=accuracy,
            timeout_rate=timeout_rate,
        )


def _extract_output_text(body: CritPtVerifyRequest) -> str:
    parts = []
    for output_item in body.response.output:
        if output_item.type != "message":
            continue
        for content_item in output_item.content:
            if content_item.type != "output_text":
                continue
            parts.append(content_item.text)
    return "".join(parts)


def _extract_code(text: str) -> str:
    """Extract Python code from model output. Matches nemo-skills _extract_code_from_generation logic."""
    matches = re.findall(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


async def _call_api(api_url: str, api_key: str, submissions: list[dict]) -> dict:
    payload = {
        "submissions": submissions,
        "batch_metadata": {},
    }
    response = await request(
        method="POST",
        url=api_url,
        json=payload,
        headers={"x-api-key": api_key},
    )
    await raise_for_status(response)
    return await response.json()


if __name__ == "__main__":
    CritPtResourcesServer.run_webserver()
