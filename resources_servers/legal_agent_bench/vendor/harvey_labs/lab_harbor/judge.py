# SPDX-FileCopyrightText: Copyright (c) 2026 Harvey AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""OpenAI-compatible judge transport for the Legal Agent Bench verifier.

The rubric prompt is adapted from Harvey LAB ``evaluation/prompts/rubric_criterion.txt``
at the pinned source revision. Transport, retries, repair, and tracing are Gym-specific.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE = """You are evaluating a legal AI agent's work product against a specific quality criterion.

## Task
{task_description}

## Agent's Output
{agent_output}

## Criterion
**{criterion_title}**

{match_criteria}

## Instructions
Evaluate the agent's output against the criterion above.
- **PASS**: The agent's output satisfies the criterion as described
- **FAIL**: The agent's output does not satisfy the criterion as described

Respond with JSON only:

```json
{{
  "verdict": "pass" | "fail",
  "reasoning": "Brief explanation"
}}
```
"""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "reasoning"],
    "additionalProperties": False,
}

_TRANSCRIPT_WRITE_LOCK = threading.Lock()


class OpenAICompatibleJudge:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        api_key: str | None,
        temperature: float | None,
        timeout_seconds: float,
        max_retries: int = 1,
        structured_output: bool = True,
        parse_repair_attempts: int = 2,
        repair_max_tokens: int = 1024,
        transcript_path: Path | None = None,
    ):
        self.model = model
        self.api_model = self._normalize_api_model(model)
        self.base_url = base_url
        self.api_key = api_key or "EMPTY"
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self.structured_output = structured_output
        self.parse_repair_attempts = max(0, parse_repair_attempts)
        self.repair_max_tokens = max(1, repair_max_tokens)
        self.transcript_path = transcript_path
        self.trace_context: dict[str, Any] = {}
        self.last_raw_response: str | None = None
        self.last_structured: bool | None = None
        if not self.base_url:
            raise ValueError("Legal Agent Bench verifier requires LAB_JUDGE_BASE_URL for OpenAI-compatible judging.")

    def evaluate(self, variables: dict[str, str]) -> dict[str, str]:
        prompt = PROMPT_TEMPLATE.format(**variables)
        return self.evaluate_prompt(prompt, VERDICT_SCHEMA, max_tokens=1024)

    def evaluate_prompt(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        max_tokens: int,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        self.last_raw_response = None
        self.last_structured = None
        structured_modes = (True, False) if self.structured_output else (False,)
        for structured in structured_modes:
            for attempt_index in range(self.max_retries):
                attempt = attempt_index + 1
                started = time.monotonic()
                self._trace(
                    {
                        "type": "judge_attempt_start",
                        "attempt": attempt,
                        "structured": structured,
                        "request_timeout_seconds": self.timeout_seconds,
                    }
                )
                try:
                    text = self._chat_completion(
                        prompt,
                        schema=schema,
                        max_tokens=max_tokens,
                        structured=structured,
                    )
                    self.last_raw_response = text
                    self.last_structured = structured
                    self._trace(
                        {
                            "type": "judge_attempt_response",
                            "attempt": attempt,
                            "structured": structured,
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                            "raw_response": text,
                        }
                    )
                    parsed = self._parse_or_repair(
                        text,
                        schema,
                        attempt=attempt,
                        structured=structured,
                    )
                    self._trace(
                        {
                            "type": "judge_attempt_parsed",
                            "attempt": attempt,
                            "structured": structured,
                            "parsed_response": parsed,
                        }
                    )
                    return parsed
                except Exception as exc:
                    last_error = exc
                    self._trace(
                        {
                            "type": "judge_attempt_error",
                            "attempt": attempt,
                            "structured": structured,
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        }
                    )
                    if attempt < self.max_retries:
                        time.sleep(min(2**attempt_index, 8))
        raise ValueError(f"Judge returned unparseable response: {last_error}")

    def _parse_or_repair(
        self,
        text: str,
        schema: dict[str, Any],
        *,
        attempt: int,
        structured: bool,
    ) -> dict[str, Any]:
        try:
            return _parse_json(text)
        except Exception as parse_exc:
            self._trace(
                {
                    "type": "judge_attempt_parse_error",
                    "attempt": attempt,
                    "structured": structured,
                    "error": str(parse_exc),
                    "error_type": type(parse_exc).__name__,
                    "raw_response": text,
                }
            )
            repaired = self._repair_response(
                text,
                schema,
                parse_error=parse_exc,
                parent_attempt=attempt,
            )
            if repaired is not None:
                return repaired
            if schema == VERDICT_SCHEMA:
                return self._parse_verdict_fallback_after_repair(
                    text,
                    attempt=attempt,
                    structured=structured,
                    parse_error=parse_exc,
                )
            raise parse_exc

    def _parse_verdict_fallback_after_repair(
        self,
        text: str,
        *,
        attempt: int,
        structured: bool,
        parse_error: Exception,
    ) -> dict[str, str]:
        candidates = []
        for candidate in (self.last_raw_response, text):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        last_error: Exception = parse_error
        for candidate in candidates:
            try:
                parsed = _parse_verdict_fallback(candidate)
                self._trace(
                    {
                        "type": "judge_attempt_fallback_parsed",
                        "attempt": attempt,
                        "structured": structured,
                        "parsed_response": parsed,
                    }
                )
                return parsed
            except Exception as exc:
                last_error = exc
        raise last_error

    def _repair_response(
        self,
        text: str,
        schema: dict[str, Any],
        *,
        parse_error: Exception,
        parent_attempt: int,
    ) -> dict[str, Any] | None:
        for repair_attempt in range(1, self.parse_repair_attempts + 1):
            started = time.monotonic()
            prompt = _repair_prompt(text, schema, parse_error)
            self._trace(
                {
                    "type": "judge_repair_attempt_start",
                    "parent_attempt": parent_attempt,
                    "repair_attempt": repair_attempt,
                    "request_timeout_seconds": self.timeout_seconds,
                }
            )
            try:
                repaired_text = self._chat_completion(
                    prompt,
                    schema=schema,
                    max_tokens=self.repair_max_tokens,
                    structured=False,
                )
                self.last_raw_response = repaired_text
                self.last_structured = False
                self._trace(
                    {
                        "type": "judge_repair_attempt_response",
                        "parent_attempt": parent_attempt,
                        "repair_attempt": repair_attempt,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "raw_response": repaired_text,
                    }
                )
                parsed = _parse_json(repaired_text)
                self._trace(
                    {
                        "type": "judge_repair_attempt_parsed",
                        "parent_attempt": parent_attempt,
                        "repair_attempt": repair_attempt,
                        "parsed_response": parsed,
                    }
                )
                return parsed
            except Exception as repair_exc:
                self._trace(
                    {
                        "type": "judge_repair_attempt_error",
                        "parent_attempt": parent_attempt,
                        "repair_attempt": repair_attempt,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "error": str(repair_exc),
                        "error_type": type(repair_exc).__name__,
                    }
                )
        return None

    def set_trace_context(self, context: dict[str, Any]) -> None:
        self.trace_context = context

    def _trace(self, event: dict[str, Any]) -> None:
        write_transcript_event(
            self.transcript_path,
            {**self.trace_context, **event},
        )

    def _chat_completion(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        max_tokens: int,
        structured: bool,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.api_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if structured:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_response",
                    "schema": schema,
                    "strict": True,
                },
            }

        body = json.dumps(payload).encode("utf-8")
        url = self.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Connection": "close",
            },
            method="POST",
        )
        started = time.monotonic()
        self._trace(
            {
                "type": "judge_http_request_start",
                "url": url,
                "model": self.api_model,
                "structured": structured,
                "max_tokens": max_tokens,
                "prompt_chars": len(prompt),
                "payload_bytes": len(body),
                "request_timeout_seconds": self.timeout_seconds,
            }
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read()
                self._trace(
                    {
                        "type": "judge_http_response",
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "status": getattr(response, "status", None),
                        "response_bytes": len(raw_body),
                    }
                )
                data = json.loads(raw_body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._trace(
                {
                    "type": "judge_http_error",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "status": exc.code,
                    "error": detail,
                    "error_type": type(exc).__name__,
                }
            )
            raise RuntimeError(f"Judge HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            self._trace(
                {
                    "type": "judge_http_error",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            raise

        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("Judge response truncated with finish_reason=length")
        message = choice.get("message") or {}
        content, content_source = _extract_judge_message_text(message)
        self._trace(
            {
                "type": "judge_http_message",
                "content_source": content_source,
                "content_chars": len(content),
            }
        )
        return content

    @staticmethod
    def _normalize_api_model(model: str) -> str:
        if "/" not in model:
            return model
        provider, model_id = model.split("/", 1)
        if provider in {"anthropic", "nvidia", "vllm", "openai-compatible", "trtllm"}:
            return model_id
        return model


def _extract_judge_message_text(message: Any) -> tuple[str, str]:
    if not isinstance(message, dict):
        raise ValueError("Judge response message is not a JSON object")
    for field in ("content", "reasoning_content", "reasoning"):
        text = _flatten_message_text(message.get(field))
        if text.strip():
            return text, field
    raise ValueError(
        "Judge response contained no text in message.content, message.reasoning_content, or message.reasoning"
    )


def _flatten_message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_flatten_message_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content"):
            text = _flatten_message_text(value.get(key))
            if text:
                return text
        return ""
    return str(value)


def _parse_json(text: str) -> dict[str, Any]:
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            if depth == 0:
                return json.loads(text[i : j + 1])
    raise ValueError(f"No JSON found in judge response: {text[:200]}")


def _repair_prompt(text: str, schema: dict[str, Any], parse_error: Exception) -> str:
    return (
        "Your previous response could not be parsed as valid JSON.\n"
        "Rewrite it as valid JSON only, with no markdown or commentary.\n"
        "Do not change the substantive verdict or reasoning.\n\n"
        f"Parse error:\n{type(parse_error).__name__}: {parse_error}\n\n"
        "Required JSON schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Previous response:\n"
        f"{text}"
    )


def _parse_verdict_fallback(text: str) -> dict[str, str]:
    verdict_match = re.search(
        r'["\']?verdict["\']?\s*:\s*["\']?(pass|fail)["\']?',
        text,
        re.IGNORECASE,
    )
    if not verdict_match:
        raise ValueError(f"No pass/fail verdict found in judge response: {text[:200]}")

    reasoning = ""
    reasoning_match = re.search(
        r'["\']?reasoning["\']?\s*:\s*(.*?)(?:\n\s*}\s*$|\s*}\s*$)',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip().rstrip(",").strip()
        if len(reasoning) >= 2 and reasoning[0] == reasoning[-1] and reasoning[0] in {"'", '"'}:
            reasoning = reasoning[1:-1]
        reasoning = reasoning.replace(r"\"", '"').replace(r"\n", "\n").strip()
    if not reasoning:
        reasoning = "Judge response did not include parseable reasoning."

    return {
        "verdict": verdict_match.group(1).lower(),
        "reasoning": reasoning,
    }


def write_transcript_event(path: Path | None, event: dict[str, Any]) -> None:
    if path is None:
        return
    event = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    with _TRANSCRIPT_WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
