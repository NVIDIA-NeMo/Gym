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
from nemo_gym.reward_profile import compute_subset_metrics
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
from resources_servers.combibench.lean_client import KiminaLeanClient


LOG = logging.getLogger(__name__)

# Lean diagnostics are echoed for debugging; cap them so a pathological proof
# cannot bloat every rollout record.
MAX_ECHOED_MESSAGES = 20
MAX_MESSAGE_CHARACTERS = 2000


class CombibenchStatus(str, Enum):
    SUCCESS = "success"
    EMPTY_OUTPUT = "empty_output"
    FORMAT_ERROR = "format_error"  # no fenced Lean block
    FORBIDDEN_KEYWORD = "forbidden_keyword"  # axiom / local_instance
    STATEMENT_MISMATCH = "statement_mismatch"  # reference statement not reproduced
    CODE_TOO_LONG = "code_too_long"
    PROOF_FAILED = "proof_failed"  # Lean reported an error
    HAS_SORRY = "has_sorry"
    TIMEOUT = "timeout"  # Lean server timed out compiling the submission
    LEAN_ERROR = "lean_error"  # Lean server reported a non-timeout REPL error
    # Harness faults: the model did not cause these.
    LEAN_SERVER_ERROR = "lean_server_error"
    BAD_TASK = "bad_task"


HARNESS_FAULTS = {CombibenchStatus.LEAN_SERVER_ERROR, CombibenchStatus.BAD_TASK}


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
    dataset_revision: Any = Field(default=None, description="Pinned upstream revision the row came from.")


class CombibenchVerifyRequest(CombibenchRunRequest, BaseVerifyRequest):
    pass


class CombibenchVerifyResponse(BaseVerifyResponse):
    status: str
    # 1.0 when the outcome is a harness fault, so the rate is a metric line.
    harness_failure: float
    answer_tags: list[str] = Field(default_factory=list)
    lean_code: Optional[str] = None  # exactly what was compiled, answer checks included
    lean_error: Optional[str] = None
    lean_messages: list[dict[str, Any]] = Field(default_factory=list)
    lean_time: Optional[float] = None


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
        result: LeanResult = await self.lean_client.verify(submission, self.config.lean_timeout_seconds)
        status = CombibenchStatus(classify_lean_result(result))
        failure_reason = None
        if status is CombibenchStatus.LEAN_SERVER_ERROR:
            failure_reason = f"Lean server unavailable or replied malformed: {result.error}"
        return self._respond(body, status, tags=tags, code=submission, result=result, failure_reason=failure_reason)

    def _respond(
        self,
        body: CombibenchVerifyRequest,
        status: CombibenchStatus,
        *,
        tags: Optional[list[str]] = None,
        code: Optional[str] = None,
        result: Optional[LeanResult] = None,
        failure_reason: Optional[str] = None,
    ) -> CombibenchVerifyResponse:
        extra = {
            "status": status.value,
            "harness_failure": 1.0 if status in HARNESS_FAULTS else 0.0,
            "answer_tags": tags or [],
            "lean_code": code,
            "lean_error": result.error if result else None,
            "lean_messages": _truncate_messages(result.messages) if result else [],
            "lean_time": result.time if result else None,
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
            KiminaLeanClient(self.config.lean_server_url, self.config.lean_server_api_key),
        )

    async def verify(self, body: CombibenchVerifyRequest) -> CombibenchVerifyResponse:
        return await self._verifier.verify(body)

    def compute_metrics(self, tasks: list[list[dict]]) -> dict:
        """Add per-source-family pass rates next to the pooled headline.

        Upstream reports one pooled figure ("solved out of 100"), so the
        inherited ``mean/reward`` headline is kept. The ``hackmath/``,
        ``brualdi/``, ``imo/`` and ``math_competitions/`` keys are supplementary
        and are not promoted to ``key_metrics``.
        """
        return compute_subset_metrics(tasks, "tag")


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
