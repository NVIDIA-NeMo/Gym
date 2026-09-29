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

"""CombiBench resources server: Lean 4 combinatorics with one-stage Fine-Eval scoring."""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Optional

from pydantic import Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.failure_kinds import PROVIDER_UNAVAILABLE
from nemo_gym.reward_profile import compute_pass_majority_metrics, compute_subset_metrics
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.combibench.fine_eval import (
    LeanResult,
    abbrev_types,
    answer_tags,
    build_submission,
    classify_lean_result,
    extract_lean_code,
    has_forbidden_substring,
    missing_chunks,
    statement_chunks,
)
from resources_servers.combibench.lean_client import DEFAULT_MAX_CONCURRENCY, KiminaLeanClient
from resources_servers.lean_proof.status import (
    STATUS_BANNED_TOKENS,
    STATUS_COMPILE_ERROR,
    STATUS_COMPLETED,
    STATUS_EMPTY_GENERATION,
    STATUS_HAS_SORRY,
    STATUS_SANDBOX_ERROR,
    STATUS_STATEMENT_MODIFIED,
    STATUS_TIMEOUT,
)


LOG = logging.getLogger(__name__)

# Lean diagnostics are echoed for debugging; cap them so a pathological proof
# cannot bloat every rollout record.
MAX_ECHOED_MESSAGES = 20
MAX_MESSAGE_CHARACTERS = 2000


class CombibenchStatus(str, Enum):
    """CombiBench's outcomes, named from ``lean_proof.status`` wherever the concept is shared.

    The six statuses every Lean benchmark has mean the same string here as in
    ``leancat``, so a rollout dump reads the same across them. The rest are
    CombiBench's own, which is the extension ``lean_proof.status`` describes:
    upstream's Fine-Eval distinguishes outcomes a whole-file benchmark has no
    equivalent for (a missing fence, an oversized submission, the Kimina REPL's
    two kinds of timeout) and those are added next to the shared names, not
    folded into them.
    """

    SUCCESS = STATUS_COMPLETED
    EMPTY_OUTPUT = STATUS_EMPTY_GENERATION
    FORBIDDEN_KEYWORD = STATUS_BANNED_TOKENS  # axiom / local_instance
    STATEMENT_MISMATCH = STATUS_STATEMENT_MODIFIED  # reference statement not reproduced
    PROOF_FAILED = STATUS_COMPILE_ERROR  # Lean reported an error
    HAS_SORRY = STATUS_HAS_SORRY
    TIMEOUT = STATUS_TIMEOUT  # Lean server timed out compiling the submission
    # CombiBench-specific outcomes.
    FORMAT_ERROR = "format_error"  # no fenced Lean block
    CODE_TOO_LONG = "code_too_long"
    LEAN_ERROR = "lean_error"  # Lean server reported a non-timeout REPL error
    # Harness faults: the model did not cause these.
    LEAN_SERVER_ERROR = STATUS_SANDBOX_ERROR
    HEADER_TIMEOUT = "header_timeout"  # a cold REPL could not load 'import Mathlib' in time
    BAD_TASK = "bad_task"


HARNESS_FAULTS = {
    CombibenchStatus.LEAN_SERVER_ERROR,
    CombibenchStatus.HEADER_TIMEOUT,
    CombibenchStatus.BAD_TASK,
}

# The shared vocabulary in ``nemo_gym.failure_kinds`` has no name for "the task row itself
# is malformed", so that one is namespaced rather than borrowed from a name that means
# something else. The two Lean-server faults are both "the provider we depend on did not
# answer", which is exactly ``provider_unavailable``.
FAILURE_KINDS: dict[CombibenchStatus, str] = {
    CombibenchStatus.LEAN_SERVER_ERROR: PROVIDER_UNAVAILABLE,
    CombibenchStatus.HEADER_TIMEOUT: PROVIDER_UNAVAILABLE,
    CombibenchStatus.BAD_TASK: "combibench:bad_task",
}


class CombibenchResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    lean_server_url: str = "http://127.0.0.1:8000"
    lean_server_api_key: Optional[str] = None
    # Upstream's Lean4Client default. The value is sent to the server, which
    # kills the REPL command at that point and reports a timeout.
    lean_timeout_seconds: int = 60
    # Bound on the extracted code before it is sent anywhere. The longest
    # pinned statement is 3,054 characters; this is a safety cap, not a measure.
    max_code_characters: int = 200_000
    # See fine_eval.missing_chunks for why this defaults on.
    normalize_trailing_whitespace: bool = True
    # See fine_eval.answer_check: elaborate the gold answer at the abbrev's
    # declared type. False reproduces upstream's unascribed check.
    answer_check_ascription: bool = True
    # Bound on in-flight Lean calls. Rollout fan-out is unbounded, and the Lean
    # server can only run LEAN_SERVER_MAX_REPLS (8 by default) of them at once.
    max_concurrent_lean_requests: int = DEFAULT_MAX_CONCURRENCY


class CombibenchRunRequest(BaseRunRequest):
    """Task fields ride at the row top level so the benchmark prompt can template them.

    Types are deliberately loose: a malformed row must become a ``bad_task``
    status, not a validation error that ends the whole run.
    """

    theorem_name: Any = Field(default=None, description="Upstream problem identifier, e.g. 'imo_2000_p4'.")
    formal_statement: Any = Field(
        default=None,
        description="Lean 4 statement with 'sorry' placeholders; the prompt shows it and verify() checks it.",
    )
    answers: Any = Field(
        default=None,
        description="Ground-truth answers for fill-in-the-blank problems, one per abbrev; null for proof-only.",
    )
    natural_language: Any = Field(default=None, description="Informal statement. Carried for provenance; unused.")
    tag: Any = Field(default=None, description="Upstream source family: hackmath, brualdi, imo, math_competitions.")
    source: Any = Field(default=None, description="Upstream source URL where published. Provenance only.")
    split: Any = Field(default=None, description="'test' (answer withheld) or 'test_with_solution'.")
    dataset_source: Any = Field(default=None, description="'github', 'hf' or 'synthetic': which upstream copy.")
    dataset_revision: Any = Field(default=None, description="Pinned upstream revision the row came from.")


class CombibenchVerifyRequest(CombibenchRunRequest, BaseVerifyRequest):
    pass


class CombibenchVerifyResponse(CombibenchRunRequest, BaseVerifyResponse):
    """Echoes the task fields so per-family metrics and post-hoc analysis can key on them."""

    status: str
    # 1.0 when the outcome is a harness fault. This is a per-row flag for reading
    # the rollout file, not a rate: harness faults also set ``mask_sample``, and
    # ``reward_profile.select_measured`` drops masked rows before any mean, so
    # ``mean/harness_failure`` is identically 0.0 even in a run where the Lean
    # server was down throughout. ``coverage/masked_rollouts`` is the aggregate
    # signal for how much of the run was lost to the harness.
    harness_failure: float
    answer_tags: list[str] = Field(default_factory=list)
    lean_code: Optional[str] = None  # exactly what was compiled, answer checks included
    lean_error: Optional[str] = None
    lean_messages: list[dict[str, Any]] = Field(default_factory=list)
    lean_time: Optional[float] = None
    # Lean version the server actually ran, so a toolchain mismatch is visible
    # in the rollouts rather than showing up as 100 failed proofs.
    lean_version: Optional[str] = None


def _text_of(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    for item in body.response.output:
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            content = getattr(item, "content", None)
            if isinstance(content, list):
                texts.extend(c.text for c in content if isinstance(getattr(c, "text", None), str))
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


def _clean_text(value: Any) -> Any:
    """Make every string safe to encode on the wire.

    A lone surrogate — from a model writing ``\\udcff`` or from an error
    message — survives JSON parsing but raises when the response is encoded.
    The echoed request is model-controlled too, so the whole response is
    cleaned, not only the fields added here.
    """
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, list):
        return [_clean_text(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean_text(v) for k, v in value.items()}
    return value


def _truncate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for message in messages[:MAX_ECHOED_MESSAGES]:
        data = message.get("data")
        if isinstance(data, str) and len(data) > MAX_MESSAGE_CHARACTERS:
            data = data[:MAX_MESSAGE_CHARACTERS] + "... [truncated]"
        out.append({"severity": message.get("severity"), "pos": message.get("pos"), "data": data})
    return out


def _validate_task(body: CombibenchRunRequest) -> Optional[str]:
    """Return a reason when the row cannot be scored, else None."""
    if not isinstance(body.formal_statement, str) or not body.formal_statement.strip():
        return "formal_statement must be a non-empty string"
    if body.answers is not None:
        if not isinstance(body.answers, list) or not all(isinstance(a, str) for a in body.answers):
            return "answers must be null or a list of strings"
    return None


class CombibenchVerifier:
    """Scoring logic, separated from the FastAPI server so the fixture can run it in process."""

    def __init__(self, config: CombibenchResourcesServerConfig, lean_client: Any):
        self.config = config
        self.lean_client = lean_client

    async def verify(self, body: CombibenchVerifyRequest) -> CombibenchVerifyResponse:
        reason = _validate_task(body)
        if reason is not None:
            return self._respond(body, CombibenchStatus.BAD_TASK, failure_reason=reason)

        chunks = statement_chunks(body.formal_statement)
        tags = answer_tags(chunks)

        text = _text_of(body)
        if not text:
            return self._respond(body, CombibenchStatus.EMPTY_OUTPUT, tags=tags)

        code = extract_lean_code(text)
        if code is None:
            return self._respond(body, CombibenchStatus.FORMAT_ERROR, tags=tags)
        if len(code) > self.config.max_code_characters:
            return self._respond(body, CombibenchStatus.CODE_TOO_LONG, tags=tags)
        if has_forbidden_substring(code):
            return self._respond(body, CombibenchStatus.FORBIDDEN_KEYWORD, tags=tags, code=code)
        if missing_chunks(code, chunks, self.config.normalize_trailing_whitespace):
            return self._respond(body, CombibenchStatus.STATEMENT_MISMATCH, tags=tags, code=code)

        types = abbrev_types(chunks) if self.config.answer_check_ascription else None
        submission = build_submission(code, tags, body.answers, types)
        # Probed once per process and cached; the first call also warms the REPL.
        lean_version = await self.lean_client.toolchain_version()
        result: LeanResult = await self.lean_client.verify(submission, self.config.lean_timeout_seconds)
        status = CombibenchStatus(classify_lean_result(result))
        failure_reason = None
        if status is CombibenchStatus.LEAN_SERVER_ERROR:
            failure_reason = f"Lean server unavailable or replied malformed: {result.error}"
        elif status is CombibenchStatus.HEADER_TIMEOUT:
            failure_reason = f"Lean server could not load its import header in time: {result.error}"
        return self._respond(
            body,
            status,
            tags=tags,
            code=submission,
            result=result,
            failure_reason=failure_reason,
            lean_version=lean_version,
        )

    def _respond(
        self,
        body: CombibenchVerifyRequest,
        status: CombibenchStatus,
        *,
        tags: Optional[list[str]] = None,
        code: Optional[str] = None,
        result: Optional[LeanResult] = None,
        failure_reason: Optional[str] = None,
        lean_version: Optional[str] = None,
    ) -> CombibenchVerifyResponse:
        harness_fault = status in HARNESS_FAULTS
        extra = {
            "status": status.value,
            "harness_failure": 1.0 if harness_fault else 0.0,
            # The model did not fail here, the harness did, so the 0.0 reward is not a
            # measurement of the model: masking keeps a Lean-server outage or an unscorable
            # row out of mean/reward and pass@k instead of averaging it in as a failure.
            "mask_sample": harness_fault,
            "failure_kind": FAILURE_KINDS.get(status),
            "answer_tags": tags or [],
            "lean_code": code,
            "lean_error": result.error if result else None,
            "lean_messages": _truncate_messages(result.messages) if result else [],
            "lean_time": result.time if result else None,
            "lean_version": lean_version,
            "failure_reason": failure_reason,
        }
        payload = _clean_text({**body.model_dump(exclude=set(extra)), **extra})
        return CombibenchVerifyResponse(**payload, reward=1.0 if status is CombibenchStatus.SUCCESS else 0.0)


class CombibenchResourcesServer(SimpleResourcesServer):
    config: CombibenchResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._verifier = CombibenchVerifier(
            self.config,
            KiminaLeanClient(
                self.config.lean_server_url,
                self.config.lean_server_api_key,
                max_concurrency=self.config.max_concurrent_lean_requests,
            ),
        )

    async def verify(self, body: CombibenchVerifyRequest) -> CombibenchVerifyResponse:
        return await self._verifier.verify(body)

    def compute_metrics(self, tasks: list[list[dict]]) -> dict:
        """Pooled pass@k, plus per-source-family pass rates next to it.

        Upstream reports one pooled figure ("solved out of 100"), so the pooled
        ``pass@k`` keys are what a reported number is read off, and the
        inherited ``mean/reward`` headline is kept. The ``hackmath/``,
        ``brualdi/``, ``imo/`` and ``math_competitions/`` keys are supplementary
        and are not promoted to ``key_metrics``.
        """
        if not tasks:
            return {}
        metrics = compute_pass_majority_metrics(tasks)[0]
        metrics.update(compute_subset_metrics(tasks, "tag"))
        return metrics


class _StubLeanClient:
    """In-process stand-in for the verifier fixture, which runs without a Lean server.

    The fixture attests the pipeline around Lean, not Lean itself: it reports a
    ``sorry`` warning when the body still contains ``sorry`` and success
    otherwise. The live-server checks live in ``scripts/harness_validation.py``.
    """

    async def verify(self, code: str, timeout_seconds: int) -> LeanResult:
        if "sorry" in code:
            return LeanResult(messages=[{"severity": "warning", "data": "declaration uses 'sorry'"}])
        return LeanResult()

    async def toolchain_version(self) -> Optional[str]:
        return None  # no Lean behind the fixture, so there is no version to report


def _fixture_verifier() -> CombibenchVerifier:
    config = CombibenchResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="combibench")
    return CombibenchVerifier(config, _StubLeanClient())


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_fixture_verifier,
    request_model=CombibenchVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    CombibenchResourcesServer.run_webserver()
