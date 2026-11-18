# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade the final boxed answer for non-math STEM questions."""

import asyncio
import json
import os
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

from aiohttp import ClientTimeout
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.verification import (
    JUDGE_FAILURES,
    JudgeError,
    call_judge,
    is_judge_failure,
    judge_failsafe,
    record_parsed_judge_verdict,
)
from resources_servers.question_bank_judge.answer_extraction import extract_last_boxed_answer
from resources_servers.question_bank_judge.judgment import hash_text, parse_judgment
from resources_servers.question_bank_judge.routing import grading_route


class QuestionBankJudgeConfig(BaseResourcesServerConfig):
    name: str = "question_bank_judge"
    grading_profile: Literal["semantic"] = "semantic"
    judge_model_server: ModelServerRef
    judge_model_revision: str | None = None
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_endpoint_max_concurrency: int | None = Field(default=64, gt=0)
    judge_system_message: str | None = None
    shared_prompt_template_fpath: str = str(Path(__file__).parent / "prompt_templates/shared_answer_v2.txt")
    judge_domain_instructions: dict[str, str] = Field(default_factory=dict)
    judge_source_instructions: dict[str, str] = Field(default_factory=dict)
    judge_timeout_seconds: float | None = Field(default=300.0, gt=0, le=300, allow_inf_nan=False)
    judge_max_attempts: int = Field(default=2, ge=1, le=2)
    boxed_judge_input: Literal["box"] = "box"
    max_box_chars: int | None = Field(default=None, gt=0)
    max_answer_chars: int | None = Field(default=16000, gt=0)
    generation_log_dir: str | None = None
    generation_log_filename: str = "question_bank_judge_generations.jsonl"


class QuestionBankJudgeRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    question: str | None = None
    expected_answer: str | None = None
    metadata: dict[str, Any] | None = None
    answer_format: Literal["free", "boxed"] | None = None


class QuestionBankJudgeVerifyRequest(QuestionBankJudgeRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    verdict_label: str | None = None


class QuestionBankJudgeVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    expected_answer: str
    answer_format: Literal["boxed"]
    grading_status: str
    judge_evaluations: list[JudgeEvaluation]
    verdict: Literal["correct", "incorrect", "unresolved"]
    correctness: bool | None
    evaluation_completed: bool
    grading_provenance: dict[str, Any]


def _assistant_text(response: NeMoGymResponse) -> str:
    """Text of the last assistant message; content blocks are joined with newlines."""
    for item in reversed(response.output):
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            content = getattr(item, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(t for t in (getattr(c, "text", None) for c in content) if isinstance(t, str))
            return ""
    return ""


def _response_completed(response: NeMoGymResponse) -> bool:
    """Require explicit completion of the response and all assistant messages."""
    return (
        response.status == "completed"
        and response.incomplete_details is None
        and response.error is None
        and all(
            getattr(item, "status", None) == "completed"
            for item in response.output
            if getattr(item, "type", None) == "message"
        )
    )


def _is_length_truncated(response: NeMoGymResponse) -> bool:
    """True if the policy hit its length cap.

    Gym's vLLM model server maps ``finish_reason == "length"`` to
    ``incomplete_details.reason == "max_output_tokens"``.
    """
    details = getattr(response, "incomplete_details", None)
    return getattr(details, "reason", None) == "max_output_tokens"


class QuestionBankJudgeResourcesServer(SimpleResourcesServer):
    config: QuestionBankJudgeConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_endpoint_max_concurrency = asyncio.Semaphore(value=self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_endpoint_max_concurrency = nullcontext()

        with open(self.config.shared_prompt_template_fpath) as f:
            self._shared_template = f.read().strip()
        # The pid suffix keeps server replicas from sharing one file; appends within
        # one process are serialized by the event loop (no await between open and close).
        self._gen_log_path: str | None = None
        if self.config.generation_log_dir:
            try:
                os.makedirs(self.config.generation_log_dir, exist_ok=True)
                base, ext = os.path.splitext(self.config.generation_log_filename)
                self._gen_log_path = os.path.join(
                    self.config.generation_log_dir, f"{base}.{os.getpid()}{ext or '.jsonl'}"
                )
            except Exception as e:
                print(f"DEBUG: question_bank_judge generation logging disabled: {type(e).__name__} {e}", flush=True)

    async def verify(self, body: QuestionBankJudgeVerifyRequest) -> QuestionBankJudgeVerifyResponse:
        """Grade only declared obligations, preserving unusable measurements as unresolved."""
        started = time.monotonic()
        cfg = self.config
        profile = cfg.grading_profile
        metadata = body.metadata or {}
        expected = body.expected_answer or ""
        raw_final = _assistant_text(body.response)
        final = raw_final.strip()
        evaluations: list[JudgeEvaluation] = []
        provenance: dict[str, Any] = {
            "verifier_revision": "question-bank-stem-v4",
            "grading_profile": profile,
            "grading_config_sha256": hash_text(cfg.model_dump_json()),
            "candidate_sha256": hash_text(raw_final),
            "reference_sha256": hash_text(expected),
            "policy_input": body.responses_create_params.model_dump(mode="json"),
            "judge_model_server": cfg.judge_model_server.name,
            "judge_model_revision": cfg.judge_model_revision,
            "route": "validation",
        }

        def finish(
            verdict: Literal["correct", "incorrect", "unresolved"],
            reason: str,
            *,
            evidence: str = "",
            failure_fields: dict[str, Any] | None = None,
        ) -> QuestionBankJudgeVerifyResponse:
            unresolved = verdict == "unresolved"
            correct = None if unresolved else verdict == "correct"
            provenance.update(reason_code=reason, evidence=evidence, elapsed_seconds=time.monotonic() - started)
            payload = body.model_dump()
            for key in (
                "_ng_failure_class",
                "_ng_failure_judge_error",
                "failure_kind",
                "failure_reason",
                "failure_outcome_action",
                "failure_outcome_attribution",
            ):
                payload.pop(key, None)
            if isinstance(payload.get("instance_config"), dict):
                payload["instance_config"] = {
                    key: value for key, value in payload["instance_config"].items() if key != "mask_sample"
                }
            if failure_fields:
                payload.update(failure_fields)
            instance_config = payload.get("instance_config")
            payload.update(
                expected_answer=expected,
                answer_format="boxed",
                grading_status=reason,
                reward=float(verdict == "correct"),
                judge_evaluations=evaluations,
                verdict=verdict,
                correctness=correct,
                evaluation_completed=not unresolved,
                mask_sample=unresolved,
                failure_kind=payload.get("failure_kind") or ("stem_judge:unresolved" if unresolved else None),
                failure_reason=(evidence or reason) if unresolved else None,
                grading_provenance=provenance,
            )
            if unresolved:
                # Older NRL consumers read the nested flag; the file collector uses the sidecar marker.
                payload["instance_config"] = {
                    **(instance_config if isinstance(instance_config, dict) else {}),
                    "mask_sample": True,
                }
                payload["_ng_failure_class"] = "judge_failed"
            result = QuestionBankJudgeVerifyResponse.model_validate(payload)
            if self._gen_log_path:
                with open(self._gen_log_path, "a") as stream:
                    stream.write(result.model_dump_json() + "\n")
            return result

        async def judge_failure(error: Exception, reason: str) -> QuestionBankJudgeVerifyResponse:
            @judge_failsafe
            async def failed(body: QuestionBankJudgeVerifyRequest) -> None:
                raise error

            response = await failed(body)
            return finish("unresolved", reason, evidence=str(error), failure_fields=json.loads(response.body))

        if grading_route(metadata.get("domain")) != "semantic":
            return finish("unresolved", "unsupported_domain")
        extras = body.model_extra or {}
        if extras.get("task_contract") is not None or extras.get("rubric") is not None:
            return finish("unresolved", "unsupported_shared_grading_task")
        methods = metadata.get("declared_methods") or []
        if isinstance(methods, str):
            methods = re.split(r"[|,;\s]+", methods)
        if (
            metadata.get("review_group") not in (None, "answer_comparison_review")
            or metadata.get("reference_kind") in {"tests_only", "native_grader"}
            or any(
                method in {"NATIVE_CODE_TESTS", "RUBRIC_ASSESSMENT", "CONSTRUCTION_VALIDATION"} for method in methods
            )
        ):
            return finish("unresolved", "unsupported_shared_grading_task")
        if body.answer_format is not None and body.answer_format != "boxed":
            return finish("unresolved", "conflicting_answer_format")
        if not expected.strip():
            return finish("unresolved", "missing_reference")
        reference_hash = metadata.get("reference_sha256")
        if reference_hash is not None and reference_hash != hash_text(expected):
            return finish("unresolved", "reference_hash_mismatch")
        question = ""
        for message in body.responses_create_params.input or []:
            if getattr(message, "role", None) != "user":
                continue
            content = getattr(message, "content", None)
            if isinstance(content, str):
                question = content.strip()
            elif isinstance(content, list) and all(
                (block.get("type") if isinstance(block, dict) else getattr(block, "type", None)) == "input_text"
                for block in content
            ):
                question = "\n".join(
                    block["text"] if isinstance(block, dict) else block.text for block in content
                ).strip()
            else:
                return finish("unresolved", "unsupported_question_content")
        if body.question is not None:
            question = body.question
        provenance["question_sha256"] = hash_text(question)
        if not question.strip():
            return finish("unresolved", "missing_question")
        if metadata.get("problem_sha256") not in (None, hash_text(question)):
            return finish("unresolved", "question_hash_mismatch")
        if _is_length_truncated(body.response):
            return finish("incorrect", "policy_truncated")
        if not _response_completed(body.response):
            return finish("unresolved", "policy_incomplete")
        if not final:
            return finish("incorrect", "empty_final_answer")
        solution = extract_last_boxed_answer(final)
        if solution is None:
            return finish("incorrect", "no_boxed")
        if cfg.max_box_chars is not None and len(solution) > cfg.max_box_chars:
            return finish("unresolved", "box_exceeds_grading_limit")
        if cfg.max_answer_chars is not None and len(solution) > cfg.max_answer_chars:
            return finish("unresolved", "answer_exceeds_grading_limit")
        provenance["judged_candidate_sha256"] = hash_text(solution)
        if not cfg.judge_model_revision or cfg.judge_model_server.name == "policy_model":
            return finish("unresolved", "fixed_judge_not_configured")

        system = self._shared_template
        if cfg.judge_system_message:
            system += "\n\n" + cfg.judge_system_message
        for field, overrides in (
            ("domain", cfg.judge_domain_instructions),
            ("source", cfg.judge_source_instructions),
        ):
            instruction = overrides.get(str(metadata.get(field, "")))
            if instruction:
                system += "\n\n" + instruction
        prompt = json.dumps(
            {
                "grading_profile": profile,
                "question": question,
                "policy_input": body.responses_create_params.model_dump(mode="json"),
                "reference": expected,
                "rubric": None,  # Keep the serialized input compatible with saved prompt replays.
                "candidate": solution,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        params = cfg.judge_responses_create_params.model_copy(deep=True)
        params.input = [
            NeMoGymEasyInputMessage(role="system", content=system),
            NeMoGymEasyInputMessage(role="user", content=prompt),
        ]
        provenance.update(
            route="semantic_judge",
            prompt_sha256=hash_text(system + "\n" + prompt),
            judge_request=params.model_dump(mode="json"),
            judge_max_attempts=cfg.judge_max_attempts,
            judge_timeout_seconds=cfg.judge_timeout_seconds,
        )
        try:
            async with self._judge_endpoint_max_concurrency:
                response = await call_judge(
                    self.server_client,
                    server_name=cfg.judge_model_server.name,
                    url_path="/v1/responses",
                    json=params,
                    response_model=NeMoGymResponse,
                    timeout=ClientTimeout(total=cfg.judge_timeout_seconds),
                    max_attempts=cfg.judge_max_attempts,
                )
        except JUDGE_FAILURES as error:
            if not is_judge_failure(error):
                raise
            return await judge_failure(error, "judge_transport_failed")
        evaluation = JudgeEvaluation(responses_create_params=params, response=response)
        evaluations.append(evaluation)
        if not _response_completed(response):
            return await judge_failure(JudgeError("Judge response did not complete"), "judge_incomplete")
        judge_messages = [item for item in response.output if getattr(item, "role", None) == "assistant"]
        if len(judge_messages) != 1:
            return await judge_failure(JudgeError("Expected one complete judgment message"), "judge_unparsed")
        judgment = parse_judgment(_assistant_text(response))
        if judgment is None:
            return await judge_failure(JudgeError("Judge output violated the strict verdict schema"), "judge_unparsed")
        evaluation.verdict_label = judgment.verdict
        if judgment.verdict == "unresolved":
            return await judge_failure(JudgeError(judgment.reason_code + ": " + judgment.evidence), "judge_unresolved")
        record_parsed_judge_verdict()
        return finish(judgment.verdict, judgment.reason_code, evidence=judgment.evidence)


if __name__ == "__main__":
    QuestionBankJudgeResourcesServer.run_webserver()
