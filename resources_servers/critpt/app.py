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
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI
from pydantic import Field, field_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.server_utils import request


# Repo root: resources_servers/critpt/app.py -> nemo-gym/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _cache_dir_from_env() -> Optional[Path]:
    """Read CRITPT_CACHE_DIR at server-construction time, or None if unset."""
    raw = os.environ.get("CRITPT_CACHE_DIR")
    return Path(raw) if raw else None


def _resolve_cache_dir(cache_dir: Path) -> Path:
    """Anchor a relative cache_dir to the repo root; leave absolute paths as-is."""
    return cache_dir if cache_dir.is_absolute() else _REPO_ROOT / cache_dir


def _run_subdir_name() -> str:
    """Unique-per-launch subdirectory name (``<timestamp>-<pid>-<rand>``).

    Each server launch writes into its own subdirectory so independent runs never
    share the append-only cache files. Sharing them would pollute partial_metrics.json
    (which aggregates every line in aa_responses.jsonl) and collide the per-process
    submission_id counter across runs. The random suffix guarantees uniqueness even for
    launches that land in the same second within one process.
    """
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def refresh_partial_metrics(cache_dir: Optional[Path]) -> None:
    """Recompute partial_metrics.json from the on-disk caches.

    Called after every successful AA call (live verify() or replay) so the file
    always reflects current state.
    """
    if cache_dir is None:
        return
    responses_path = cache_dir / "aa_responses.jsonl"
    submissions_path = cache_dir / "submissions.jsonl"
    scored = 0
    accuracy_sum = 0.0
    timeout_rate_sum = 0.0
    n_batches = 0
    if responses_path.exists():
        with responses_path.open("r") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n = len(rec["submission_ids"])
                response = rec["response"]
                scored += n
                accuracy_sum += float(response.get("accuracy", 0.0)) * n
                timeout_rate_sum += float(response.get("timeout_rate", 0.0)) * n
                n_batches += 1
    n_submissions = 0
    if submissions_path.exists():
        with submissions_path.open("r") as fh:
            for raw in fh:
                if raw.strip():
                    n_submissions += 1
    partial = {
        "scored_submissions": scored,
        "total_submissions_seen": n_submissions,
        "pending_submissions": max(0, n_submissions - scored),
        "scored_batches": n_batches,
        "mean_accuracy_over_scored": (accuracy_sum / scored) if scored else None,
        "mean_timeout_rate_over_scored": (timeout_rate_sum / scored) if scored else None,
        "ts": time.time(),
    }
    out_path = cache_dir / "partial_metrics.json"
    with out_path.open("w") as fh:
        json.dump(partial, fh, indent=2)


LOG = logging.getLogger(__name__)

# Canonical PUBLIC problem set; used to pad sub-batch fires in smoke-test mode.
_ALL_PROBLEM_IDS = [f"Challenge_{n}_main" for n in range(1, 71)]


class CritPtRateLimitExceeded(RuntimeError):
    """Raised when the AA scoring API returns HTTP 429 (quota exhausted).

    Carries the structured retry signals AA emits so callers (verify() in this
    file, and the standalone `replay.py` tool) can distinguish "wait and retry"
    from a programmer error and surface a clean, machine-greppable log line
    instead of a generic stack trace.
    """

    def __init__(self, retry_after_seconds: int, reset_unix: int, body: str):
        self.retry_after_seconds = retry_after_seconds
        self.reset_unix = reset_unix
        self.body = body
        super().__init__(
            f"CritPt AA quota exhausted (retry_after={retry_after_seconds}s, reset_unix={reset_unix}); body={body!r}"
        )


class CritPtResourcesServerConfig(BaseResourcesServerConfig):
    api_url: str = "https://artificialanalysis.ai/api/v2/critpt/evaluate"
    # One AA key, or a list of keys for in-process rotation on HTTP 429.
    api_key: Union[str, List[str]]
    # AA PUBLIC mode requires all 70 problems in one call; verify() buffers until full.
    batch_size: int = 70
    # Per-batch AA API call timeout. AA can take ~minutes to evaluate 70 submissions server-side.
    api_timeout_seconds: float = 1800.0
    api_max_retries: int = 4
    api_retry_backoff_seconds: float = 5.0
    # Smoke-test only. When set < batch_size, the buffer fires after this many real
    # submissions arrive and pads up to batch_size with empty padding submissions.
    fire_after: Optional[int] = None
    # Max time a single verify() will wait for its batch to fill (and the AA call to
    # finish) to prevent hang.
    verify_timeout_seconds: float = 9000.0  # (2.5h)
    # When set, the server persists every arriving submission and every successful
    # AA response under this directory. Enables partial-scoring visibility during a
    # live run (writes update incrementally, not only at the end) and judge-only
    # replay via `replay.py` after the AA quota resets. Files written:
    #   submissions.jsonl    one line per submission seen at verify()
    #   aa_responses.jsonl   one line per AA call that returned 2xx
    #   partial_metrics.json aggregate accuracy over scored submissions
    # Leaving this None (or unset) preserves the prior behavior (no on-disk state).
    cache_dir: Optional[Path] = Field(default_factory=_cache_dir_from_env)
    # Each server launch writes into its own ``<timestamp>-<pid>`` subdirectory under
    # cache_dir so independent runs never share append-only cache files (which would
    # otherwise pollute partial_metrics.json and collide submission ids across runs).
    # Set False to write directly into cache_dir, e.g. to replay a specific prior run.
    unique_cache_per_run: bool = True

    @field_validator("api_key")
    @classmethod
    def _validate_api_key(cls, v):
        """Fail fast on empty key configs.

        We accept `str | List[str]` for ergonomic env passthrough, but a bare
        empty string or an empty/all-blank list almost always means a missing
        export — surface that at construction time rather than at the first
        429 (where a confused stack trace would suggest a quota bug instead).
        """
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("CritPt api_key must be a non-empty string")
            return v
        items = list(v)
        if any(not isinstance(k, str) for k in items):
            raise ValueError("CritPt api_key list must contain only strings")
        keys = [k for k in items if k.strip()]
        if not keys:
            raise ValueError("CritPt api_key list must contain at least one non-empty key")
        return keys


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
        # AA 500s on concurrent full-batch submissions; serialize them.
        self._api_lock = asyncio.Lock()
        # Pending batches, oldest first; each verify() joins the first batch lacking its problem_id.
        self._batches: list[dict] = []
        # Monotonic counter surfaced in per-verify log lines for inline progress tracking.
        self._total_verify_calls: int = 0
        # Stable, atomic-w.r.t.-the-asyncio-loop ids assigned to each submission/batch
        # as we persist them. Used by replay.py to detect which submissions already
        # have an AA score and skip re-shipping them.
        self._submission_counter: int = 0
        self._batch_counter: int = 0
        # Normalize api_key into a non-empty list of keys; the field validator
        # has already enforced non-emptiness. The rotation cursor is advanced
        # after each successful batch so successive batches round-robin across
        # keys, and bumped on 429 so the next batch starts past the
        # rate-limited key. All cursor mutations happen under `_api_lock`.
        raw = self.config.api_key
        self._api_keys: List[str] = [raw] if isinstance(raw, str) else list(raw)
        self._key_index: int = 0
        if self.config.cache_dir is not None:
            base = _resolve_cache_dir(self.config.cache_dir)
            if self.config.unique_cache_per_run:
                base = base / _run_subdir_name()
            self.config.cache_dir = base
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            LOG.warning("CritPt cache for this run: %s", self.config.cache_dir)

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
        # Persist before any await so on-disk state captures every submission that
        # reached verify(), even if an asyncio cancellation lands before the batch
        # fires (common when a sibling rollout fails and Gym cancels the cohort).
        submission_id = self._record_submission(submission)

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
                    "submission_ids": {},
                }
                self._batches.append(target_batch)

            target_batch["submissions"][body.problem_id] = submission
            target_batch["submission_ids"][body.problem_id] = submission_id
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

            ready_to_fire = len(target_batch["submissions"]) >= (self.config.fire_after or self.config.batch_size)
            if ready_to_fire:
                submissions_snapshot = list(target_batch["submissions"].values())
                submission_ids_snapshot = list(target_batch["submission_ids"].values())
                self._batches.remove(target_batch)
                # Smoke-mode padding: top up to batch_size with empty padding for missing
                # problem_ids. Padded entries are synthetic so they get no submission_id.
                if len(submissions_snapshot) < self.config.batch_size:
                    existing = {s["problem_id"] for s in submissions_snapshot}
                    for pid in _ALL_PROBLEM_IDS:
                        if len(submissions_snapshot) >= self.config.batch_size:
                            break
                        if pid not in existing:
                            submissions_snapshot.append(
                                {
                                    "problem_id": pid,
                                    "generated_code": "```python\n```",
                                    "model": "unknown",
                                    "generation_config": {},
                                }
                            )
            else:
                submissions_snapshot = None
                submission_ids_snapshot = None

        if ready_to_fire:
            LOG.warning(
                "CritPt batch full (%d submissions); firing AA API (key %d/%d).",
                len(submissions_snapshot),
                self._key_index + 1,
                len(self._api_keys),
            )
            try:
                # Serialize: only one AA submission in flight at a time (concurrent ones 500).
                # Rotation cursor (`self._key_index`) is mutated only under this
                # lock, so successive batches cleanly round-robin across keys.
                async with self._api_lock:
                    result = await asyncio.wait_for(
                        self._call_aa_with_rotation(submissions_snapshot),
                        timeout=self.config.api_timeout_seconds,
                    )
                self._record_aa_response(submission_ids_snapshot, result)
                self._refresh_partial_metrics()
                future.set_result(result)
            except CritPtRateLimitExceeded as e:
                LOG.error(
                    "CritPt AA quota exhausted on all %d configured key(s) "
                    "(last retry_after=%ds, reset_unix=%d); failing %d waiters in "
                    "this batch. Submissions cached at %s; run "
                    "`python -m resources_servers.critpt.replay --cache-dir <that_path>` "
                    "after the quota resets to score the remaining batches without "
                    "rerunning inference.",
                    len(self._api_keys),
                    e.retry_after_seconds,
                    e.reset_unix,
                    len(submissions_snapshot),
                    self.config.cache_dir,
                )
                future.set_exception(e)
            except Exception as e:
                LOG.exception("CritPt AA API call failed; failing all %d waiters: %s", len(submissions_snapshot), e)
                future.set_exception(e)

        try:
            result = await asyncio.wait_for(future, timeout=self.config.verify_timeout_seconds)
        except asyncio.TimeoutError:
            LOG.error(
                "CritPt verify timed out after %ss waiting for batch to fire (problem_id=%s). "
                "Likely a sibling rollout failed before reaching verify().",
                self.config.verify_timeout_seconds,
                body.problem_id,
            )
            raise

        accuracy = result["accuracy"]
        timeout_rate = result.get("timeout_rate", 0.0)
        # AA returns one aggregate accuracy; distribute it as each rollout's reward (matches nemo-skills).
        return CritPtVerifyResponse(
            **body.model_dump(),
            reward=accuracy,
            accuracy=accuracy,
            timeout_rate=timeout_rate,
        )

    # ──────────────────────────────────────────────────────────
    # AA key rotation
    # ──────────────────────────────────────────────────────────

    async def _call_aa_with_rotation(self, submissions: List[Dict]) -> Dict:
        """Ship a batch to AA, rotating across configured keys on HTTP 429.

        Sticky behaviour: every batch starts on `self._key_index` and the
        cursor only advances when a key 429s. On success the cursor stays
        on the key that worked, so subsequent batches keep hitting it
        until AA rate-limits it. On a 429 the cursor moves to the next key
        and retries the same submissions immediately; once every key has
        429'd in a single batch (one full cycle), the last
        `CritPtRateLimitExceeded` is re-raised so the surrounding verify()
        block can fail the cohort with a clean signal.

        With a single configured key this degenerates to a thin wrapper
        around `_call_api` (one attempt, no rotation).

        Callers must hold `self._api_lock` — both because AA itself 500s on
        concurrent full-batch submissions, and because this function mutates
        `self._key_index`.
        """
        n = len(self._api_keys)
        last_exc: Optional[CritPtRateLimitExceeded] = None
        for attempt in range(n):
            current = (self._key_index + attempt) % n
            try:
                result = await _call_api(
                    self.config.api_url,
                    self._api_keys[current],
                    submissions,
                    max_retries=self.config.api_max_retries,
                    backoff_seconds=self.config.api_retry_backoff_seconds,
                )
                self._key_index = current
                return result
            except CritPtRateLimitExceeded as exc:
                last_exc = exc
                remaining = n - (attempt + 1)
                if remaining > 0:
                    next_idx = (current + 1) % n
                    LOG.warning(
                        "CritPt AA rate-limited on key %d/%d (retry_after=%ds, reset_unix=%d); rotating to key %d/%d.",
                        current + 1,
                        n,
                        exc.retry_after_seconds,
                        exc.reset_unix,
                        next_idx + 1,
                        n,
                    )
        if last_exc is None:
            raise RuntimeError("_call_aa_with_rotation invoked with no api_keys configured")
        raise last_exc

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

    # ──────────────────────────────────────────────────────────
    # Persistence helpers (cache_dir-gated; no-op when unset)
    # ──────────────────────────────────────────────────────────

    def _record_submission(self, submission: Dict[str, Any]) -> int:
        """Assign a monotonic submission_id and append the submission to disk.

        Synchronous on purpose: there must be no asyncio await between
        constructing `submission` in verify() and committing it here, so
        cancellation cannot land between "model produced an answer" and
        "answer is on disk".
        """
        sid = self._submission_counter
        self._submission_counter += 1
        if self.config.cache_dir is None:
            return sid
        path = self.config.cache_dir / "submissions.jsonl"
        with path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "submission_id": sid,
                        "submission": submission,
                        "ts": time.time(),
                    }
                )
                + "\n"
            )
        return sid

    def _record_aa_response(self, submission_ids: List[int], response: Dict[str, Any]) -> int:
        """Append a successful AA batch response to aa_responses.jsonl.

        Replay reads this file to skip batches that already have an AA score,
        so a partial-quota run never wastes future quota re-scoring submissions
        that succeeded the first time.
        """
        bid = self._batch_counter
        self._batch_counter += 1
        if self.config.cache_dir is None:
            return bid
        path = self.config.cache_dir / "aa_responses.jsonl"
        with path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "batch_id": bid,
                        "submission_ids": submission_ids,
                        "response": response,
                        "ts": time.time(),
                    }
                )
                + "\n"
            )
        return bid

    def _refresh_partial_metrics(self) -> None:
        refresh_partial_metrics(self.config.cache_dir)


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
        # 429: quota exhausted. AA emits Retry-After (seconds until allowed) and
        # X-Ratelimit-Reset (unix timestamp). Raise a typed exception so verify()
        # and replay.py can surface a clean signal instead of a generic stack
        # trace; do not retry in-pipeline (real Retry-After is ~24h on free tier).
        if response.status == 429:
            try:
                retry_after = int(response.headers.get("Retry-After") or "0")
            except (TypeError, ValueError):
                retry_after = 0
            try:
                reset_unix = int(response.headers.get("X-Ratelimit-Reset") or "0")
            except (TypeError, ValueError):
                reset_unix = 0
            raise CritPtRateLimitExceeded(
                retry_after_seconds=retry_after,
                reset_unix=reset_unix,
                body=body,
            )
        # Retry only on 5xx (transient AA server errors). Non-429 4xx means a bad
        # request/payload that won't succeed on retry, so fail fast with the
        # response body for debugging.
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
