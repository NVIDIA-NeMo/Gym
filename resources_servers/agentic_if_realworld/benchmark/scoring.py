# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent judge recovery on immutable policy answers."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import jsonschema

from .core import (
    DEFAULT_JUDGE_MODEL,
    ROOT,
    bounded_schema,
    digest,
    judge,
    provider_schema,
    read,
    unscored,
    write,
)
from .registry import private_path
from .transport import verified_response, wire_call


@asynccontextmanager
async def _cache_lock(work: Path):
    # Advisory locks distinguish live duplicate callers from abandoned receipts.
    fd = os.open(work / "score.lock", os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


async def score(
    payload: dict,
    *,
    directory: Path,
    endpoint: str,
    policy_prompt: str,
    model: str = DEFAULT_JUDGE_MODEL,
    key_env: str = "AGENTIC_IF_JUDGE_API_KEY",
    reasoning: str = "medium",
    cached_only: bool = False,
    validate_only: bool = False,
) -> dict:
    directory = private_path(directory)
    if not isinstance(policy_prompt, str) or not policy_prompt.strip():
        raise ValueError("Explicit judge policy prompt is required")
    judge.validate_payload(payload)
    schema = bounded_schema(provider_schema(read(ROOT / "policy/judge_output.schema.json")), payload)
    body = {
        "model": model,
        "reasoning_effort": reasoning,
        "messages": [
            {
                "role": "user",
                "content": policy_prompt + "\nUNTRUSTED_EVALUATION_PAYLOAD:\n" + judge.encode(payload),
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "agentic_if_strict_review",
                "strict": True,
                "schema": schema,
            },
        },
    }
    identity = judge.sha({"payload": payload, "request": body, "endpoint": endpoint})
    work = directory / "judges" / identity
    if cached_only and not (work / "result.json").is_file():
        raise ValueError("Saved judge result missing; cached-only verification cannot dispatch")
    if validate_only and not work.exists():
        return unscored(payload["item_id"], "judge_cache_validated_without_scoring")
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    async with _cache_lock(work):
        if (work / "result.json").exists():
            cached = read(work / "result.json")
            if (
                cached["payload_sha256"] != judge.sha(payload)
                or read(work / "payload.json") != payload
                or read(work / "transport_schema.json") != schema
            ):
                raise ValueError("Cached judge payload changed")
            name = cached["receipt_name"]
            if name not in {"attempt0.json", "attempt1.json"}:
                raise ValueError("Cached judge receipt path changed")
            receipt_path = work / name
            if digest(receipt_path) != cached["receipt_sha256"]:
                raise ValueError("Cached judge receipt changed")
            receipt = read(receipt_path)
            if (
                receipt["request_body"] != body
                or receipt["request_sha256"] != judge.sha(body)
                or receipt["endpoint"] != endpoint
            ):
                raise ValueError("Cached judge request changed")
            review, checked = _checked_result(verified_response(receipt), payload, schema, body)
            if cached["raw_review"] != review or cached["result"] != checked:
                raise ValueError("Cached grade differs from validated review")
            return checked
        if cached_only:
            raise ValueError("Saved judge result missing; cached-only verification cannot dispatch")
        if validate_only:
            for name, expected in [("payload.json", payload), ("transport_schema.json", schema)]:
                if (work / name).exists() and read(work / name) != expected:
                    raise ValueError("Saved pending judge payload or schema changed")
            for attempt in range(2):
                receipt_path = work / f"attempt{attempt}.json"
                if not receipt_path.exists():
                    continue
                # The cache lock and existing receipt keep this on the no-dispatch transport path.
                receipt = await wire_call(body, endpoint=endpoint, key_env=key_env, path=receipt_path)
                if receipt["status"] == "generated":
                    verified_response(receipt)
            return unscored(payload["item_id"], "judge_cache_validated_without_scoring")
        write(work / "payload.json", payload)
        write(work / "transport_schema.json", schema)
        for attempt in range(2):
            receipt_path = work / f"attempt{attempt}.json"
            receipt = await wire_call(
                body,
                endpoint=endpoint,
                key_env=key_env,
                path=receipt_path,
            )
            if receipt["status"] != "generated":
                if receipt.get("terminal"):
                    break
                continue
            response = verified_response(receipt)
            try:
                review, result = _checked_result(response, payload, schema, body)
            except (ValueError, TypeError, KeyError, jsonschema.ValidationError):
                continue
            write(
                work / "result.json",
                {
                    "result": result,
                    "raw_review": review,
                    "payload_sha256": judge.sha(payload),
                    "receipt_name": receipt_path.name,
                    "receipt_sha256": digest(receipt_path),
                },
            )
            return result
        return unscored(payload["item_id"], "judge_delivery_or_validation_error")


def _checked_result(response: dict, payload: dict, schema: dict, body: dict) -> tuple[dict, dict]:
    """Derive the full grade from the authoritative judge response."""
    choices = response["choices"]
    if len(choices) != 1 or choices[0]["finish_reason"] != "stop":
        raise ValueError("Incomplete judge delivery")
    review = json.loads(choices[0]["message"]["content"])
    jsonschema.validate(review, schema)
    result = judge.validate_review(review, payload)
    result.update(
        reward=float(result["item_verdict"] == "pass"),
        mask_sample=False,
        judge_request_sha256=judge.sha(body),
        judge_model=body["model"],
        reasoning_effort=body["reasoning_effort"],
        judge_response_id=response.get("id"),
        judge_usage=response.get("usage"),
    )
    return review, result
