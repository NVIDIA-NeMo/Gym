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
from typing import Any, Dict, List, Union

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.server_utils import request


LOG = logging.getLogger(__name__)


class CritPtResourcesServerConfig(BaseResourcesServerConfig):
    api_url: str = "https://artificialanalysis.ai/api/v2/critpt/evaluate"
    api_key: str
    # AA API requires submissions for every hosted problem in PUBLIC mode (currently 70).
    # The server buffers verify() calls until batch_size unique problem_ids accumulate,
    # then fires one API call and distributes the aggregate accuracy to all waiters.
    batch_size: int = 70
    # Per-batch AA API call timeout. AA can take ~minutes to evaluate 70 submissions
    # server-side; default generously, override if needed.
    api_timeout_seconds: float = 1800.0
    # Retries for the AA API call on 5xx server errors (transient). 4xx (bad payload)
    # fails immediately — retrying won't help.
    api_max_retries: int = 4
    # Base seconds for exponential backoff between AA retries (2nd try waits this, then 2x, ...).
    api_retry_backoff_seconds: float = 5.0


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
        # Serializes AA API calls: the endpoint 500s when two full-batch submissions are
        # in flight at once (e.g. num_repeats>1, where multiple batches fill ~simultaneously).
        # Holding this lock around _call_api fires them one at a time.
        self._api_lock = asyncio.Lock()
        # Pending batches, oldest first. Each batch is
        # {"future": asyncio.Future, "submissions": dict[problem_id, submission]}.
        # A verify() call joins the first batch that doesn't already contain its problem_id,
        # or creates a new batch if all existing batches already have it. This enables
        # num_repeats > 1: each repeat of a given problem flows into a distinct batch.
        self._batches: list[dict] = []
        # Monotonic counter of verify() calls received since startup. Surfaced in the
        # per-verify log line so users tailing the log can read the running total inline
        # without grepping for line counts.
        self._total_verify_calls: int = 0

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.get("/status")(self.status)
        return app

    async def status(self) -> dict:
        """Return the live buffer fill across all pending batches. Read-only; no lock."""
        return {
            "pending_batches": [len(b["submissions"]) for b in self._batches],
            "batch_size": self.config.batch_size,
        }

    async def verify(self, body: CritPtVerifyRequest) -> CritPtVerifyResponse:
        code = _extract_code(_extract_output_text(body))
        submission = {
            "problem_id": body.problem_id,
            "generated_code": f"```python\n{code}\n```" if code else "```python\n```",
            "model": "unknown",
            "generation_config": {},
        }

        async with self._lock:
            # Find the first pending batch that doesn't already contain this problem_id.
            # If all pending batches have it (or none exist), open a new one.
            target_batch = next(
                (b for b in self._batches if body.problem_id not in b["submissions"]),
                None,
            )
            if target_batch is None:
                target_batch = {
                    "future": asyncio.get_running_loop().create_future(),
                    "submissions": {},
                }
                self._batches.append(target_batch)

            target_batch["submissions"][body.problem_id] = submission
            future = target_batch["future"]
            self._total_verify_calls += 1
            LOG.warning(
                "CritPt verify #%d: batch %d at %d/%d submissions buffered (problem_id=%s)",
                self._total_verify_calls,
                self._batches.index(target_batch),
                len(target_batch["submissions"]),
                self.config.batch_size,
                body.problem_id,
            )

            ready_to_fire = len(target_batch["submissions"]) >= self.config.batch_size
            if ready_to_fire:
                submissions_snapshot = list(target_batch["submissions"].values())
                self._batches.remove(target_batch)
            else:
                submissions_snapshot = None

        if ready_to_fire:
            LOG.warning("CritPt batch full (%d submissions); firing AA API.", len(submissions_snapshot))
            try:
                # Serialize: only one AA submission in flight at a time (concurrent ones 500).
                async with self._api_lock:
                    result = await asyncio.wait_for(
                        _call_api(
                            self.config.api_url,
                            self.config.api_key,
                            submissions_snapshot,
                            max_retries=self.config.api_max_retries,
                            backoff_seconds=self.config.api_retry_backoff_seconds,
                        ),
                        timeout=self.config.api_timeout_seconds,
                    )
                future.set_result(result)
            except Exception as e:
                LOG.exception("CritPt AA API call failed; failing all %d waiters: %s", len(submissions_snapshot), e)
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

    # ──────────────────────────────────────────────────────────
    # Aggregate metrics overrides
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _critpt_score_fn(r: dict) -> Dict[str, Union[float, bool]]:
        return {"accuracy": r["accuracy"]} if "accuracy" in r else {}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute CritPt metrics: pass@k, majority@k, per-sample stats — named `accuracy`."""
        return compute_pass_majority_metrics(
            tasks,
            score_fn=self._critpt_score_fn,
            answer_key="problem_id",
        )[0]

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline metrics for CritPt: pass@1/accuracy and pass@k/accuracy variants."""
        key: Dict[str, Any] = {}

        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}"))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}"))

        return key


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


async def _call_api(
    api_url: str,
    api_key: str,
    submissions: list[dict],
    max_retries: int = 4,
    backoff_seconds: float = 5.0,
) -> dict:
    payload = {
        "submissions": submissions,
        "batch_metadata": {},
    }
    for attempt in range(1, max_retries + 1):
        response = await request(
            method="POST",
            url=api_url,
            json=payload,
            headers={"x-api-key": api_key},
        )
        if response.ok:
            return await response.json()

        body = (await response.text())[:2000]
        # Retry only on 5xx (transient AA server errors). 4xx means a bad request/payload
        # that won't succeed on retry, so fail fast with the response body for debugging.
        if response.status >= 500 and attempt < max_retries:
            wait = backoff_seconds * (2 ** (attempt - 1))
            LOG.warning(
                "CritPt AA API returned %d (attempt %d/%d); retrying in %.0fs: %s",
                response.status,
                attempt,
                max_retries,
                wait,
                body,
            )
            await asyncio.sleep(wait)
            continue
        raise RuntimeError(
            f"CritPt AA API returned {response.status} for {len(submissions)} submissions "
            f"({len(set(s['problem_id'] for s in submissions))} unique problem_ids): {body}"
        )
    raise RuntimeError("CritPt AA API: exhausted retries without a response")  # unreachable


if __name__ == "__main__":
    CritPtResourcesServer.run_webserver()
