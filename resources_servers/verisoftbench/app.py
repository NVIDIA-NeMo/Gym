# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""VeriSoftBench verifier: Lean 4 kernel verification inside the pinned repository toolchains.

Each rollout is one of upstream's k candidates together with its repair chain. Round 0
verifies the model's ``<lean4_proof>`` (and optional ``<lean4_invented_lemmas>``); when it
fails, the response carries upstream's repair prompt and the state the next round needs.
The agent (``responses_api_agents/verisoftbench_agent``) drives up to ``r`` repair rounds
and reports the rollout as solved if any round verified. The benchmark's estimand — a task
is solved if any of k candidates or their repairs verify — is computed over the k rollouts
of a task in ``compute_metrics``.

Everything that decides whether a proof counts runs in upstream's own code (see
``upstream.py``); this file adds the request/response contract, failure attribution,
concurrency bounds and metrics.
"""

import asyncio
import logging
import re
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ValidationError
from upstream import (
    CONTEXT_MODE,
    Harness,
    UpstreamError,
    resolve_upstream_dir,
)

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.failure_kinds import VERIFIER_ERROR
from nemo_gym.openai_utils import NeMoGymResponse


logger = logging.getLogger(__name__)

# Upstream: 3 repair rounds per failed candidate (``fix_single_theorem(max_fix_attempts=3)``).
DEFAULT_MAX_REPAIR_ROUNDS = 3
# Upstream: ``max_workers: 8`` verification threads in the shipped configs.
DEFAULT_MAX_CONCURRENT_COMPILATIONS = 8
# Upstream's initial path resolves "has already been declared" clashes once, its repair
# path up to ten times (``verify_single_proof`` vs ``fix_single_theorem``).
CONFLICT_ATTEMPTS_INITIAL = 1
CONFLICT_ATTEMPTS_REPAIR = 10

# Everything upstream's evaluator reads from a task row.
REQUIRED_TASK_FIELDS = ("id", "thm_name", "thm_stmt", "lean_root", "rel_path", "imports", "local_ctx")


class Status(str, Enum):
    VERIFIED = "verified"
    LEAN_ERROR = "lean_error"  # compiled, kernel rejected (includes `sorry`/`admit`, timeouts)
    NO_PROOF_TAG = "no_proof_tag"  # response had no <lean4_proof>/<fixed_lean4_thm_proof>; compiled as empty (round 0) or previous (repair)
    EMPTY_OUTPUT = "empty_output"
    BACKEND_ERROR = "backend_error"  # Lean container or docker unusable
    BAD_METADATA = "bad_metadata"
    BAD_REPAIR_STATE = "bad_repair_state"
    ROUNDS_EXHAUSTED = "rounds_exhausted"  # agent asked for a round beyond max_repair_rounds


HARNESS_FAULTS = {
    Status.BACKEND_ERROR: "the Lean container or the docker daemon was unavailable",
    Status.BAD_METADATA: "verifier_metadata is missing a field or has a wrong type",
    Status.BAD_REPAIR_STATE: "repair_round > 0 without a usable repair_state",
    Status.ROUNDS_EXHAUSTED: "repair_round exceeds max_repair_rounds",
}


class VeriSoftBenchResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE = ReverifyMode.STATELESS
    # Pinned utopia-group/VeriSoftBench checkout (scripts/prepare_verisoftbench.py creates it).
    upstream_dir: str
    # Running container of the image built by docker/build_image.sh.
    docker_container: str = "verisoftbench-lean"
    max_repair_rounds: int = DEFAULT_MAX_REPAIR_ROUNDS
    max_concurrent_compilations: int = DEFAULT_MAX_CONCURRENT_COMPILATIONS
    # Round-0 conflict-resolution attempts and repair-round attempts, upstream's values.
    conflict_attempts_initial: int = CONFLICT_ATTEMPTS_INITIAL
    conflict_attempts_repair: int = CONFLICT_ATTEMPTS_REPAIR
    # Refuse to serve if the container is not running at startup.
    require_container_at_startup: bool = True


class RepairState(BaseModel):
    """What upstream carries from one repair round to the next."""

    proof: str
    aux_lemmas: str
    err_msg: str


class VeriSoftBenchRunRequest(BaseRunRequest):
    verifier_metadata: dict[str, Any]


class VeriSoftBenchVerifyRequest(VeriSoftBenchRunRequest, BaseVerifyRequest):
    repair_round: int = 0
    repair_state: Optional[dict[str, Any]] = None


class VeriSoftBenchVerifyResponse(BaseVerifyResponse):
    status: str
    repair_round: int
    verified: bool
    harness_failure: float
    task_id: Optional[int] = None
    lean_root: Optional[str] = None
    category: Optional[str] = None
    proof: str = ""
    aux_lemmas: str = ""
    reasoning: str = ""
    err_msg: Optional[str] = None
    conflict_resolution_attempts: int = 0
    lean_time_s: float = 0.0
    needs_repair: bool = False
    repair_prompt: Optional[str] = None
    repair_state: Optional[dict[str, Any]] = None
    # Set when upstream's repair-prompt builder raised; the chain ends as it does upstream.
    repair_prompt_error: Optional[str] = None


def _last_assistant_text(response: NeMoGymResponse) -> str:
    """Upstream reads the concatenated text blocks of the reply (``_call_claude``)."""
    for item in reversed(response.output):
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            return "".join(
                getattr(part, "text", "") for part in item.content if getattr(part, "type", None) == "output_text"
            )
    return ""


_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _sanitize(value: Any) -> Any:
    """Lone surrogates survive the runner's JSON round trip and fail on the way out."""
    if isinstance(value, str):
        return _SURROGATE_RE.sub("�", value)
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def check_task_fields(metadata: dict[str, Any]) -> Optional[str]:
    """Return a reason if upstream's evaluator would fail on this row, else None."""
    for field in REQUIRED_TASK_FIELDS:
        if field not in metadata:
            return f"missing verifier_metadata.{field}"
    for field in ("thm_name", "thm_stmt", "lean_root", "rel_path", "local_ctx"):
        if not isinstance(metadata[field], str):
            return f"verifier_metadata.{field} is not a string"
    if not isinstance(metadata["imports"], list) or not all(isinstance(i, str) for i in metadata["imports"]):
        return "verifier_metadata.imports is not a list of strings"
    if not isinstance(metadata["id"], int) or isinstance(metadata["id"], bool):
        return "verifier_metadata.id is not an integer"
    gt = metadata.get("ground_truth_proof", "")
    if gt is not None and not isinstance(gt, str):
        return "verifier_metadata.ground_truth_proof is not a string"
    return None


class VeriSoftBenchResourcesServer(SimpleResourcesServer):
    config: VeriSoftBenchResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._harness: Optional[Harness] = None
        self._harness_error: Optional[str] = None
        self._entries: dict[int, dict[str, Any]] = {}
        self._entry_locks: dict[int, asyncio.Lock] = {}
        self._compile_slots = asyncio.Semaphore(self.config.max_concurrent_compilations)
        try:
            self._harness = Harness(resolve_upstream_dir(self.config.upstream_dir), self.config.docker_container)
        except UpstreamError as e:
            # Kept as a status rather than a crash so that a misconfigured server reports
            # every rollout as a harness fault instead of taking the run down with a 500.
            self._harness_error = str(e)
            logger.error("Upstream harness unavailable: %s", e)
        if (
            self.config.require_container_at_startup
            and self._harness is not None
            and not self._harness.container_running()
        ):
            raise RuntimeError(
                f"Lean container {self.config.docker_container!r} is not running; start it with "
                f"`docker run -d --name {self.config.docker_container} <image>` (see README) or set "
                "require_container_at_startup=false to serve anyway and report backend_error"
            )

    # --- metrics ---------------------------------------------------------------

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Upstream's estimand over the rollouts of each task.

        ``any_of_k/solved``: fraction of tasks with at least one rollout whose candidate
        or a repair of it verified — Table 3's "pass@k with r repair rounds".
        ``any_of_k/solved_without_repair``: the same over round-0 verdicts only
        (upstream's ``proof_success_rate_wo_fix``). Both in percent, like upstream's tables.
        """
        out: dict[str, Any] = {}
        if not tasks:
            return out
        solved = [any(r.get("reward", 0.0) >= 1.0 for r in rollouts) for rollouts in tasks]
        solved_wo = [any(r.get("solved_without_repair", 0.0) >= 1.0 for r in rollouts) for rollouts in tasks]
        out["any_of_k/solved"] = 100.0 * sum(solved) / len(tasks)
        out["any_of_k/solved_without_repair"] = 100.0 * sum(solved_wo) / len(tasks)
        out["count/tasks"] = len(tasks)
        out["min/rollouts_per_task"] = min(len(rollouts) for rollouts in tasks)
        out["max/rollouts_per_task"] = max(len(rollouts) for rollouts in tasks)
        by_category: dict[str, list[bool]] = {}
        for rollouts, is_solved in zip(tasks, solved):
            category = next((r.get("category") for r in rollouts if isinstance(r.get("category"), str)), None)
            if category is not None:
                by_category.setdefault(category, []).append(is_solved)
        for category, values in sorted(by_category.items()):
            key = category.lower().replace(" ", "_").replace(".", "")
            out[f"any_of_k/solved/{key}"] = 100.0 * sum(values) / len(values)
            out[f"count/tasks/{key}"] = len(values)
        return out

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        keep = {
            "any_of_k/solved",
            "any_of_k/solved_without_repair",
            "count/tasks",
            "min/rollouts_per_task",
            "mean/reward",
            "mean/solved_without_repair",
            "mean/repair_rounds_used",
            "mean/harness_failure",
            "mean/input_tokens",
            "mean/output_tokens",
            "mean/total_tokens",
        }
        return {k: v for k, v in agent_metrics.items() if k in keep}

    # --- verify ----------------------------------------------------------------

    async def verify(self, body: VeriSoftBenchVerifyRequest) -> VeriSoftBenchVerifyResponse:
        metadata = body.verifier_metadata
        round_index = body.repair_round
        reason = check_task_fields(metadata) if isinstance(metadata, dict) else "verifier_metadata is not an object"
        if reason is not None:
            return self._respond(body, Status.BAD_METADATA, failure_reason=reason)
        if self._harness is None:
            return self._respond(body, Status.BACKEND_ERROR, failure_reason=self._harness_error)
        if round_index < 0 or round_index > self.config.max_repair_rounds:
            return self._respond(body, Status.ROUNDS_EXHAUSTED)
        repair_state: Optional[RepairState] = None
        if round_index > 0:
            try:
                repair_state = RepairState.model_validate(body.repair_state)
            except ValidationError as e:
                return self._respond(
                    body, Status.BAD_REPAIR_STATE, failure_reason=f"repair_state: {e.errors()[0]['msg']}"
                )

        text = _sanitize(_last_assistant_text(body.response))
        harness = self._harness
        try:
            entry = await self._entry_for(metadata)
        except Exception as e:  # docker exec failed while reading the source file
            logger.warning("Could not prepare task %s: %s", metadata.get("id"), e)
            return self._respond(body, Status.BACKEND_ERROR, failure_reason=f"prepare_entry: {e}")

        reasoning = ""
        if round_index == 0:
            proof, lemmas, reasoning = harness.parse_initial(text)
            had_tag = "<lean4_proof>" in text
            attempts = self.config.conflict_attempts_initial
        else:
            proof, lemmas = harness.parse_repair(text, repair_state.proof, repair_state.aux_lemmas)
            had_tag = "<fixed_lean4_thm_proof>" in text
            attempts = self.config.conflict_attempts_repair

        proof_id = f"{round_index}_{uuid.uuid4().hex[:10]}"
        loop = asyncio.get_running_loop()
        async with self._compile_slots:
            if not harness.container_running():
                return self._respond(body, Status.BACKEND_ERROR, failure_reason="Lean container is not running")
            t0 = loop.time()
            try:
                outcome = await asyncio.to_thread(harness.verify_candidate, entry, proof, lemmas, proof_id, attempts)
            except Exception as e:
                logger.exception("verify_candidate raised for task %s", metadata.get("id"))
                return self._respond(
                    body, Status.BACKEND_ERROR, failure_reason=f"verify_candidate: {type(e).__name__}: {e}"
                )
            lean_time = loop.time() - t0

        if not outcome.verified and outcome.backend_fault:
            return self._respond(body, Status.BACKEND_ERROR, failure_reason=outcome.error_msg, lean_time_s=lean_time)

        if outcome.verified:
            status = Status.VERIFIED
        elif not text.strip():
            status = Status.EMPTY_OUTPUT
        elif not had_tag:
            status = Status.NO_PROOF_TAG
        else:
            status = Status.LEAN_ERROR

        needs_repair = not outcome.verified and round_index < self.config.max_repair_rounds
        repair_prompt = None
        next_state = None
        repair_prompt_error = None
        if needs_repair:
            next_state = RepairState(
                proof=outcome.proof, aux_lemmas=outcome.aux_lemmas, err_msg=outcome.error_msg or ""
            )
            try:
                repair_prompt = await asyncio.to_thread(
                    harness.repair_prompt, entry, next_state.proof, next_state.aux_lemmas, next_state.err_msg
                )
            except Exception as e:
                # Upstream's prompt builder can raise on some compiler-error shapes
                # (`_format_no_goals` indexes a list that may be shorter than two). In
                # upstream's evaluator that exception ends the candidate's repair chain and
                # the round's verdict stands; the same happens here, and it is flagged.
                logger.warning("repair_prompt raised for task %s: %s: %s", metadata.get("id"), type(e).__name__, e)
                needs_repair = False
                next_state = None
                repair_prompt_error = f"{type(e).__name__}: {e}"
        return self._respond(
            body,
            status,
            reward=1.0 if outcome.verified else 0.0,
            proof=outcome.proof,
            aux_lemmas=outcome.aux_lemmas,
            reasoning=reasoning,
            err_msg=outcome.error_msg,
            conflict_resolution_attempts=outcome.conflict_resolution_attempts,
            lean_time_s=lean_time,
            needs_repair=needs_repair,
            repair_prompt=repair_prompt,
            repair_state=next_state.model_dump() if next_state else None,
            repair_prompt_error=repair_prompt_error,
        )

    async def _entry_for(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Upstream's cleaned statement and verification context, built once per task."""
        task_id = metadata["id"]
        cached = self._entries.get(task_id)
        if cached is not None:
            return cached
        lock = self._entry_locks.setdefault(task_id, asyncio.Lock())
        async with lock:
            cached = self._entries.get(task_id)
            if cached is None:
                cached = await asyncio.to_thread(self._harness.prepare_entry, metadata)
                self._entries[task_id] = cached
        return cached

    def _respond(
        self,
        body: VeriSoftBenchVerifyRequest,
        status: Status,
        reward: float = 0.0,
        failure_reason: Optional[str] = None,
        lean_time_s: float = 0.0,
        **fields: Any,
    ) -> VeriSoftBenchVerifyResponse:
        harness_fault = status in HARNESS_FAULTS
        echoed = _sanitize(body.model_dump(exclude={"repair_round", "repair_state"}))
        # Echoed provenance is model- and dataset-controlled: take it from the sanitized dump.
        metadata = echoed["verifier_metadata"] if isinstance(echoed.get("verifier_metadata"), dict) else {}
        task_id = metadata.get("id")
        payload = dict(
            **echoed,
            reward=reward,
            status=status.value,
            repair_round=body.repair_round,
            verified=status is Status.VERIFIED,
            harness_failure=1.0 if harness_fault else 0.0,
            mask_sample=harness_fault,
            failure_kind=VERIFIER_ERROR if harness_fault else None,
            failure_reason=_sanitize(failure_reason or (HARNESS_FAULTS[status] if harness_fault else None)),
            task_id=task_id if isinstance(task_id, int) else None,
            lean_root=metadata.get("lean_root") if isinstance(metadata.get("lean_root"), str) else None,
            category=metadata.get("category") if isinstance(metadata.get("category"), str) else None,
            lean_time_s=lean_time_s,
            **_sanitize(fields),
        )
        return VeriSoftBenchVerifyResponse(**payload)


__all__ = [
    "CONTEXT_MODE",
    "Status",
    "VeriSoftBenchResourcesServer",
    "VeriSoftBenchResourcesServerConfig",
    "VeriSoftBenchVerifyRequest",
    "VeriSoftBenchVerifyResponse",
]

if __name__ == "__main__":
    VeriSoftBenchResourcesServer.run_webserver()
