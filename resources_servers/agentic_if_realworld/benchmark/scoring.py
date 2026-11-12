# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent judge recovery on immutable policy answers."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import jsonschema

from . import evidence
from .core import (
    DEFAULT_JUDGE_MODEL,
    ROOT,
    TRANSPORT_VERSION,
    bounded_schema,
    digest,
    judge,
    provider_schema,
    read,
    unscored,
    write,
)
from .registry import private_path
from .transport import RetryableTransportError, validate_endpoint, verified_response, wire_call


def validate_judge_config(endpoint: str, model: str) -> None:
    """Validate judge routing before any paid policy generation, without requiring a secret."""
    validate_endpoint(endpoint)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Judge model must be nonempty")


def _validate_dispatch(receipt: dict) -> None:
    if receipt["status"] == "generated" or receipt.get("dispatch_status") == "not_dispatched":
        return
    status = receipt.get("http_status", 0)
    # Non-2xx headers establish no review delivery even if the error body is truncated.
    if 300 <= status < 600:
        return
    if (
        receipt.get("dispatch_status") == "unknown"
        or not 200 <= status < 300
        or receipt.get("response_body_complete") is not True
    ):
        raise ValueError("Ambiguous judge dispatch; reconcile saved transport before retrying")


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
    evidence_transport: str = "quotes",
    cached_only: bool = False,
    validate_only: bool = False,
) -> dict:
    directory = private_path(directory)
    if not isinstance(policy_prompt, str) or not policy_prompt.strip():
        raise ValueError("Explicit judge policy prompt is required")
    validate_judge_config(endpoint, model)
    judge.validate_payload(payload)
    schema = bounded_schema(provider_schema(read(ROOT / "policy/judge_output.schema.json")), payload)
    if evidence_transport not in ("quotes", "evidence_ids_v1"):
        raise ValueError("Unknown judge evidence transport")
    indexed = evidence_transport == "evidence_ids_v1"
    wire_schema = evidence.wire_schema(schema, payload) if indexed else provider_schema(schema)
    body = {
        "model": model,
        "reasoning_effort": reasoning,
        "messages": [
            {
                "role": "user",
                "content": policy_prompt
                + "\nUNTRUSTED_EVALUATION_PAYLOAD:\n"
                + judge.encode(payload)
                + (evidence.prompt_note(payload) if indexed else ""),
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "agentic_if_strict_review",
                "strict": True,
                "schema": wire_schema,
            },
        },
    }
    identity = judge.sha(
        {
            "payload": payload,
            "request": body,
            "endpoint": endpoint,
            "validation_schema": schema,
            "transport_version": evidence.VERSION if indexed else TRANSPORT_VERSION,
        }
    )
    work = directory / "judges" / identity
    if cached_only and not (work / "result.json").is_file():
        raise ValueError("Saved judge result missing; cached-only verification cannot dispatch")
    if validate_only and not work.exists():
        return unscored(payload["item_id"], "judge_cache_validated_without_scoring")
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    async with _cache_lock(work):
        for name, expected in [
            ("payload.json", payload),
            ("transport_schema.json", schema),
            ("provider_schema.json", wire_schema),
        ]:
            if (work / name).exists() and read(work / name) != expected:
                raise ValueError("Saved pending judge payload or schema changed")
        receipt_paths = sorted(work.glob("attempt*.json"), key=lambda path: int(path.stem.removeprefix("attempt")))
        if [path.name for path in receipt_paths] != [f"attempt{index}.json" for index in range(len(receipt_paths))]:
            raise ValueError("Saved judge attempt sequence changed")
        receipts = []
        for receipt_path in receipt_paths:
            # Existing paths are read-only: validate every prior receipt before any recovery dispatch.
            receipt = await wire_call(body, endpoint=endpoint, key_env=key_env, path=receipt_path)
            if receipt["status"] == "generated":
                verified_response(receipt)
            receipts.append(receipt)
        if (work / "result.json").exists():
            cached = read(work / "result.json")
            if (
                cached["payload_sha256"] != judge.sha(payload)
                or read(work / "payload.json") != payload
                or read(work / "transport_schema.json") != schema
                or read(work / "provider_schema.json") != wire_schema
            ):
                raise ValueError("Cached judge payload changed")
            name = cached["receipt_name"]
            if not isinstance(name, str) or not re.fullmatch(r"attempt(0|[1-9][0-9]*)\.json", name):
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
            review, checked = _checked_result(verified_response(receipt), payload, schema, body, indexed=indexed)
            if cached["raw_review"] != review or cached["result"] != checked:
                raise ValueError("Cached grade differs from validated review")
            return checked
        if cached_only:
            raise ValueError("Saved judge result missing; cached-only verification cannot dispatch")
        for receipt in receipts:
            _validate_dispatch(receipt)
        if validate_only:
            return unscored(payload["item_id"], "judge_cache_validated_without_scoring")
        write(work / "payload.json", payload)
        write(work / "transport_schema.json", schema)
        write(work / "provider_schema.json", wire_schema)
        if indexed:
            write(work / "response_span_index.json", evidence.build_index(payload))
        delivered_reviews = 0
        # At most two 2xx review deliveries; transport-only failures end this invocation.
        for attempt in range(len(receipts) + 2):
            if delivered_reviews == 2:
                break
            receipt_path = work / f"attempt{attempt}.json"
            saved = attempt < len(receipts)
            receipt = (
                receipts[attempt]
                if saved
                else await wire_call(body, endpoint=endpoint, key_env=key_env, path=receipt_path)
            )
            _validate_dispatch(receipt)
            if receipt["status"] != "generated":
                status = receipt.get("http_status", 0)
                if 200 <= status < 300:
                    delivered_reviews += 1
                    continue
                if not saved:
                    raise RetryableTransportError(
                        "Judge transport failed; fix the service or credentials and explicitly resume the saved answer"
                    )
                continue
            delivered_reviews += 1
            response = verified_response(receipt)
            try:
                review, result = _checked_result(response, payload, schema, body, indexed=indexed)
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


def _checked_result(
    response: dict, payload: dict, schema: dict, body: dict, *, indexed: bool = False
) -> tuple[dict, dict]:
    """Derive the full grade from the authoritative judge response."""
    choices = response["choices"]
    if (
        len(choices) != 1
        or choices[0]["finish_reason"] != "stop"
        or choices[0].get("native_finish_reason") == "repetition"
        or choices[0].get("stop_reason") == "repetition_detected"
    ):
        raise ValueError("Incomplete judge delivery")
    review = json.loads(choices[0]["message"]["content"])
    if indexed:
        jsonschema.validate(review, body["response_format"]["json_schema"]["schema"])
        review, proof = evidence.decode_review(review, payload)
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
    if indexed:
        result.update(evidence_transport=evidence.VERSION, evidence_resolution=proof)
    return review, result
