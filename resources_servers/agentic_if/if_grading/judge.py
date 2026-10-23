# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The LLM judge of `judge` checks and of `language` checks that code cannot settle."""

import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from copy import deepcopy

from aiohttp import ClientResponseError
from pydantic import JsonValue

from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponse

from .configuration import config_hash, load_config, votes_for
from .contracts import InstructionContext, JudgeDecision
from .instructions import instruction_parts


LOG = logging.getLogger(__name__)

JudgeRequest = Callable[[dict[str, JsonValue]], NeMoGymResponse]


def _verdict(text: str) -> tuple[str, str]:
    """(status, reason) of a judge reply; a reply with no verdict raises ValueError."""
    for match in reversed(list(re.finditer(r'\{[^{}]*"(?:answer|kept)"[^{}]*\}', text, re.S))):
        try:
            candidate = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        answer = candidate.get("answer")
        if isinstance(answer, str) and answer.strip().lower() in ("yes", "no"):
            return ("pass" if answer.strip().lower() == "yes" else "fail", str(candidate.get("why", "")))
        if isinstance(candidate.get("kept"), bool):
            return ("pass" if candidate["kept"] else "fail", str(candidate.get("why", "")))
    match = re.search(r'answer"?\s*:\s*"?(yes|no)\b', text, re.I)
    if match:
        reason = re.search(r'why"?\s*:\s*"(.*?)"?\s*\}?\s*$', text, re.S)
        return ("pass" if match[1].lower() == "yes" else "fail", reason[1] if reason else "")
    raise ValueError("judge returned no verdict")


def _failure(error: Exception) -> str:
    """What failed, without the reply or the request: the exception type, and the HTTP status of a failed call."""
    cause = error.__cause__ if isinstance(error, JudgeError) and error.__cause__ is not None else error
    if isinstance(cause, ClientResponseError):
        return f"{type(cause).__name__} (HTTP {cause.status})"
    return type(cause).__name__


class Judge:
    """Ask the judge model the questions of one grade, with the settings of `data/judge.yaml`.

    `send` posts one Responses request to the judge model server and returns its reply; the Resources Server sends
    it through Gym's `call_judge`, so the model server's retries and logging apply. Grading runs on a worker thread,
    so `send` blocks. Each vote that gets no verdict is asked again up to `model.attempts` times; a vote that still
    has none is an `error` decision, never a failure. Verdicts are cached by question and vote for the grade, so a
    unit asked twice gets the same answer; errors are not cached.
    """

    def __init__(self, send: JudgeRequest, *, config: dict | None = None) -> None:
        self.config = deepcopy(config) if config is not None else load_config()
        model = self.config["model"]
        if model.get("api") != "responses":
            raise ValueError("judge model.api must be responses")
        votes, attempts = model.get("votes", 1), model.get("attempts", 1)
        if type(votes) is not int or votes < 1 or votes % 2 == 0:
            raise ValueError("model.votes must be a positive odd integer")
        if type(attempts) is not int or attempts < 1:
            raise ValueError("model.attempts must be a positive integer")
        for context, settings in self.config["contexts"].items():
            context_votes = settings.get("votes", votes)
            if type(context_votes) is not int or context_votes < 1 or context_votes % 2 == 0:
                raise ValueError(f"{context}.votes must be a positive odd integer")
        self.hash = config_hash(self.config)
        self.version, self.model = self.config["version"], model["name"]
        self._send = send
        self._cache: dict[str, JudgeDecision] = {}
        self._pending: dict[str, Future] = {}
        self._lock = threading.Lock()

    def prompt(self, rubric: str, context: str, evidence: str, instruction: InstructionContext | None = None) -> str:
        """The judge's whole question: with the instruction the model was given when it is known."""
        if instruction is not None:
            return self.config["instruction_prompt"].format(
                question=rubric,
                context_name=self.config["contexts"][context]["title"],
                context=evidence,
                **instruction_parts(self.config, instruction),
            )
        return self.config["prompt"].format(
            question=rubric, context_name=self.config["contexts"][context]["title"], context=evidence
        )

    def _key(self, prompt: str, vote: int) -> str:
        return hashlib.sha256(json.dumps([self.hash, prompt, vote], ensure_ascii=False).encode()).hexdigest()

    def _request(self, prompt: str) -> str:
        model = self.config["model"]
        body = {
            "model": model["name"],
            "input": [{"role": "user", "content": prompt}],
            "temperature": model["temperature"],
            "reasoning": {"effort": model["reasoning_effort"], "summary": model.get("reasoning_summary", "auto")},
            "max_output_tokens": model.get("max_output_tokens", 8192),
        }
        return self._send(body).output_text

    def _vote(self, prompt: str, vote: int, instruction: InstructionContext | None) -> JudgeDecision:
        key = self._key(prompt, vote)
        with self._lock:
            if key in self._cache:
                return self._cache[key].model_copy(deep=True)
            pending = self._pending.get(key)
            owner = pending is None
            if owner:
                pending = self._pending[key] = Future()
        if not owner:
            return pending.result().model_copy(deep=True)

        metadata = {
            "config_hash": self.hash,
            "version": self.version,
            "model": self.model,
            "key": key,
            "vote": vote,
            "judge_saw_instruction": instruction is not None,
        }
        attempts = self.config["model"].get("attempts", 1)
        decision = JudgeDecision(status="error", explanation="No judge verdict", metadata=metadata)
        try:
            for attempt in range(attempts):
                try:
                    status, why = _verdict(self._request(prompt))
                except Exception as error:
                    # Only the failure's kind is kept: a reply or an error body may quote the evidence.
                    reason = _failure(error)
                    LOG.warning("Judge vote %d: attempt %d of %d failed: %s", vote, attempt + 1, attempts, reason)
                    decision = JudgeDecision(
                        status="error",
                        explanation=f"No judge verdict after {attempt + 1} attempt(s): {reason}",
                        metadata=metadata | {"attempts": attempt + 1},
                    )
                    if attempt + 1 < attempts:
                        time.sleep(3 * (attempt + 1))
                    continue
                decision = JudgeDecision(status=status, explanation=why, metadata=metadata | {"attempts": attempt + 1})
                with self._lock:
                    self._cache[key] = decision.model_copy(deep=True)
                break
        finally:
            with self._lock:
                pending.set_result(decision)
                del self._pending[key]
        return decision

    def evaluate(
        self,
        *,
        rubric: str,
        context: str,
        evidence: str,
        n_steps: int | None = None,
        instruction: InstructionContext | dict | None = None,
    ) -> JudgeDecision:
        """One decision from the context's votes, by majority; any vote without a verdict makes it an error."""
        instruction = InstructionContext.model_validate(instruction) if instruction is not None else None
        prompt = self.prompt(rubric, context, evidence, instruction)
        votes = [self._vote(prompt, vote, instruction) for vote in range(votes_for(self.config, context, n_steps))]
        error = next((decision for decision in votes if decision.status == "error"), None)
        status = (
            "error"
            if error
            else "pass"
            if sum(decision.status == "pass" for decision in votes) * 2 > len(votes)
            else "fail"
        )
        example = error or next(decision for decision in votes if decision.status == status)
        explanation = example.explanation
        if len(votes) > 1:
            answers = ", ".join(
                "yes" if vote.status == "pass" else "no" if vote.status == "fail" else "error" for vote in votes
            )
            explanation = f"[votes: {answers}] {explanation}"
        return JudgeDecision(
            status=status,
            explanation=explanation,
            metadata={
                "config_hash": self.hash,
                "version": self.version,
                "model": self.model,
                "judge_saw_instruction": instruction is not None,
                "votes": [decision.model_dump(mode="json") for decision in votes],
            },
        )
