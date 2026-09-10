# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Batch-API transport for evaluations at 50% token cost.

The rollout, tool-execution, and scoring machinery is untouched: the clients
here implement the same verifiers Client interface as the interactive ones and
reuse their exact request builders (StreamingAnthropicClient.build_create_kwargs
/ OpenAIResponsesClient.build_call_kwargs), so every turn's request body is
byte-identical to what the interactive runner would send. Only the transport
differs — instead of one live HTTP call per turn, concurrently-running tasks'
turns are coalesced into "waves" and each wave is submitted as one provider
batch (Anthropic Message Batches / OpenAI Batch). Conversations are simply
split across batch requests; scores are drawn from the same distribution.

Trade-off: waves are sequential per task (turn N+1 can only be requested after
turn N's batch resolves), so wall-clock time is bounded by the slowest task's
step count x batch turnaround, in exchange for 50% off all token costs.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import anthropic
import openai
from openai import AsyncOpenAI

from verifiers.errors import OverlongPromptError

from automationbench.clients import (
    OpenAIResponsesClient,
    StreamingAnthropicClient,
    _record_model_call,
)

# Provider-side transient error markers -> the item is resubmitted in the next
# wave (the batch-mode analog of the interactive clients' retry loop).
_RETRYABLE_ANTHROPIC = {"api_error", "overloaded_error", "rate_limit_error", "timeout_error"}
_RETRYABLE_OPENAI_ITEM_ERRORS = {"batch_expired", "request_timeout"}
_MAX_ITEM_ATTEMPTS = 5

# Conservative per-batch ceilings (Anthropic: 100k reqs/256MB; OpenAI: 50k/200MB).
_MAX_WAVE_REQUESTS = 10_000
_MAX_WAVE_BYTES = 150 * 1024 * 1024


class BatchRequestError(Exception):
    """A batch item failed with a non-retryable provider error."""

    def __init__(self, custom_id: str, payload: Any):
        self.custom_id = custom_id
        self.payload = payload
        super().__init__(f"batch request {custom_id} failed: {payload}")


@dataclass
class _PendingItem:
    custom_id: str
    body: dict
    future: asyncio.Future
    attempts: int = 0


@dataclass
class WaveStats:
    waves: int = 0
    requests: int = 0
    retried: int = 0
    failed: int = 0


class WaveBatcher:
    """Coalesce per-turn requests from concurrent rollouts into provider batches.

    submit() parks the caller on a future; a flusher task collects everything
    that arrives within a quiet period (``linger_seconds`` with no new
    arrivals, or ``max_linger_seconds`` since the first pending item) into one
    wave and hands it to ``run_wave``. Multiple waves may be in flight at once
    (stragglers and retries ride later waves) — a wave is a batching unit, not
    a turn barrier, so tasks progress independently exactly as they do in the
    interactive runner.

    ``run_wave(items)`` maps {custom_id: body} -> {custom_id: (status, value)}
    where status is "ok" (value = native response), "retry" (transient), or
    "error" (value raised as BatchRequestError). Missing ids are treated as
    "retry" (e.g. expired batch items).
    """

    def __init__(
        self,
        run_wave: Callable[[dict[str, dict]], Awaitable[dict[str, tuple[str, Any]]]],
        linger_seconds: float = 20.0,
        max_linger_seconds: float = 120.0,
        on_wave: Callable[[int, int], None] | None = None,
    ):
        self._run_wave = run_wave
        self._linger = linger_seconds
        self._max_linger = max_linger_seconds
        self._on_wave = on_wave
        self._pending: dict[str, _PendingItem] = {}
        self._seq = 0
        self._last_arrival = 0.0
        self._first_pending = 0.0
        self._flusher: asyncio.Task | None = None
        self._wave_tasks: set[asyncio.Task] = set()
        self.stats = WaveStats()

    async def submit(self, body: dict) -> Any:
        loop = asyncio.get_running_loop()
        self._seq += 1
        custom_id = f"req_{self._seq:06d}"
        item = _PendingItem(custom_id=custom_id, body=body, future=loop.create_future())
        self._enqueue(item)
        return await item.future

    def _enqueue(self, item: _PendingItem) -> None:
        now = time.monotonic()
        if not self._pending:
            self._first_pending = now
        self._pending[item.custom_id] = item
        self._last_arrival = now
        if self._flusher is None or self._flusher.done():
            self._flusher = asyncio.get_running_loop().create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        while self._pending:
            now = time.monotonic()
            quiet_for = now - self._last_arrival
            waited = now - self._first_pending
            if (
                quiet_for >= self._linger
                or waited >= self._max_linger
                or len(self._pending) >= _MAX_WAVE_REQUESTS
            ):
                wave = dict(self._pending)
                self._pending.clear()
                task = asyncio.get_running_loop().create_task(self._execute_wave(wave))
                self._wave_tasks.add(task)
                task.add_done_callback(self._wave_tasks.discard)
            else:
                await asyncio.sleep(min(1.0, self._linger - quiet_for))
                continue
            # After launching a wave, keep looping if new items arrived meanwhile.
            if self._pending:
                self._first_pending = time.monotonic()
            else:
                return

    async def _execute_wave(self, wave: dict[str, _PendingItem]) -> None:
        # A caller can cancel while its item is waiting for the linger window.
        # Do not submit work that no longer has a listener.
        wave = {cid: item for cid, item in wave.items() if not item.future.done()}
        if not wave:
            return
        self.stats.waves += 1
        self.stats.requests += len(wave)
        if self._on_wave:
            self._on_wave(self.stats.waves, len(wave))
        try:
            results = await self._run_wave({cid: it.body for cid, it in wave.items()})
        except Exception as e:  # noqa: BLE001 — a wave-level failure fails its items
            for it in wave.values():
                if not it.future.done():
                    it.future.set_exception(e)
            return

        for cid, it in wave.items():
            # The caller can also cancel while the provider batch is in flight.
            # Future.set_result/set_exception raises InvalidStateError on a
            # cancelled future, which previously stranded every later result.
            if it.future.done():
                continue
            status, value = results.get(cid, ("retry", "missing from batch results"))
            if status == "ok":
                it.future.set_result(value)
            elif status == "retry" and it.attempts + 1 < _MAX_ITEM_ATTEMPTS:
                it.attempts += 1
                self.stats.retried += 1
                self._enqueue(it)
            else:
                self.stats.failed += 1
                payload = value if status != "ok" else "exhausted retries"
                it.future.set_exception(BatchRequestError(cid, payload))


def _to_ns(obj: Any) -> Any:
    """Recursively wrap a raw JSON body in attribute-accessible namespaces.

    A batch result body is raw API JSON. The OpenAI SDK's strict
    Response.model_validate rejects it whenever the body carries a field/type
    the pinned SDK version doesn't model (e.g. a newer reasoning model's output
    shape) — which aborts every task. The interactive client never validates;
    its from_native_response reads the response purely via getattr. Wrapping the
    body as namespaces lets that same getattr extraction run on the batch body,
    so batch scoring is identical to interactive and never aborts on a
    well-formed-but-newer body.
    """
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _chunk_items(items: dict[str, dict]) -> list[dict[str, dict]]:
    """Split a wave into provider-safe chunks by request count and byte size."""
    chunks: list[dict[str, dict]] = []
    current: dict[str, dict] = {}
    current_bytes = 0
    for cid, body in items.items():
        size = len(json.dumps(body))
        if current and (
            len(current) >= _MAX_WAVE_REQUESTS or current_bytes + size > _MAX_WAVE_BYTES
        ):
            chunks.append(current)
            current, current_bytes = {}, 0
        current[cid] = body
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks


def _is_retryable_anthropic_exception(error: BaseException) -> bool:
    if isinstance(error, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return True
    if isinstance(error, anthropic.APIStatusError):
        status = getattr(error, "status_code", None)
        text = str(error)
        return bool(
            (status is not None and (status == 429 or status >= 500))
            or "overloaded_error" in text
            or "rate_limit_error" in text
        )
    return False


def _is_retryable_openai_exception(error: BaseException) -> bool:
    if isinstance(error, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(error, openai.APIStatusError):
        status = getattr(error, "status_code", None)
        return status is not None and (status == 429 or status >= 500)
    return False


def _openai_item_error_is_retryable(error: Any) -> bool:
    code = error.get("code") if isinstance(error, dict) else getattr(error, "code", None)
    return code in _RETRYABLE_OPENAI_ITEM_ERRORS


class AnthropicBatchBackend:
    """Run a wave through the Anthropic Message Batches API."""

    def __init__(self, client, poll_seconds: float = 30.0):
        self.client = client
        self.poll_seconds = poll_seconds

    async def run_wave(self, items: dict[str, dict]) -> dict[str, tuple[str, Any]]:
        out: dict[str, tuple[str, Any]] = {}
        for chunk in _chunk_items(items):
            try:
                out.update(await self._run_chunk(chunk))
            except Exception as error:  # noqa: BLE001 — translate provider failures per item
                status = "retry" if _is_retryable_anthropic_exception(error) else "error"
                out.update({cid: (status, error) for cid in chunk})
        return out

    async def _run_chunk(self, items: dict[str, dict]) -> dict[str, tuple[str, Any]]:
        requests = [{"custom_id": cid, "params": body} for cid, body in items.items()]
        # Interactive path sends the interleaved-thinking beta per-request when
        # thinking is enabled; batch applies it batch-wide (same semantics — the
        # beta only affects requests that carry a thinking param).
        extra_headers = None
        if any("thinking" in body for body in items.values()):
            extra_headers = {"anthropic-beta": "interleaved-thinking-2025-05-14"}

        batch = await self.client.messages.batches.create(
            requests=requests, extra_headers=extra_headers
        )
        while True:
            info = await self.client.messages.batches.retrieve(batch.id)
            if info.processing_status == "ended":
                break
            await asyncio.sleep(self.poll_seconds)

        out: dict[str, tuple[str, Any]] = {}
        async for entry in await self.client.messages.batches.results(batch.id):
            r = entry.result
            if r.type == "succeeded":
                out[entry.custom_id] = ("ok", r.message)
            elif r.type == "errored":
                err = r.error
                etype = getattr(getattr(err, "error", None), "type", None) or getattr(
                    err, "type", ""
                )
                status = "retry" if etype in _RETRYABLE_ANTHROPIC else "error"
                out[entry.custom_id] = (status, err)
            elif r.type == "expired":
                out[entry.custom_id] = ("retry", "expired")
            else:  # canceled
                out[entry.custom_id] = ("error", r.type)
        return out


async def _read_file_text(client: AsyncOpenAI, file_id: str) -> str:
    resp = await client.files.content(file_id)
    text = getattr(resp, "text", None)
    if isinstance(text, str):
        return text
    data = await resp.aread() if hasattr(resp, "aread") else resp.read()
    return data.decode("utf-8")


class OpenAIResponsesBatchBackend:
    """Run a wave through the OpenAI Batch API against /v1/responses."""

    def __init__(self, client: AsyncOpenAI, poll_seconds: float = 30.0):
        self.client = client
        self.poll_seconds = poll_seconds

    async def run_wave(self, items: dict[str, dict]) -> dict[str, tuple[str, Any]]:
        out: dict[str, tuple[str, Any]] = {}
        for chunk in _chunk_items(items):
            try:
                out.update(await self._run_chunk(chunk))
            except Exception as error:  # noqa: BLE001 — translate provider failures per item
                status = "retry" if _is_retryable_openai_exception(error) else "error"
                out.update({cid: (status, error) for cid in chunk})
        return out

    async def _run_chunk(self, items: dict[str, dict]) -> dict[str, tuple[str, Any]]:
        from openai.types.responses import Response as OAIResponse

        jsonl = "\n".join(
            json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/responses", "body": body})
            for cid, body in items.items()
        )
        f = await self.client.files.create(
            file=("automationbench_batch.jsonl", jsonl.encode("utf-8")), purpose="batch"
        )
        batch = await self.client.batches.create(
            input_file_id=f.id, endpoint="/v1/responses", completion_window="24h"
        )
        terminal = {"completed", "failed", "expired", "cancelled"}
        while True:
            info = await self.client.batches.retrieve(batch.id)
            if info.status in terminal:
                break
            await asyncio.sleep(self.poll_seconds)

        out: dict[str, tuple[str, Any]] = {}
        if info.status == "failed" and not (info.output_file_id or info.error_file_id):
            # Batch-level rejection. Some causes are transient (enqueued-token
            # limit) and worth resubmitting; others are permanent for every
            # request in the wave (unsupported model, malformed input) and must
            # fail fast — retrying just burns _MAX_ITEM_ATTEMPTS waves to the
            # same end. Classify by error code.
            errs = getattr(info.errors, "data", None) or []
            codes = {getattr(e, "code", "") for e in errs}
            reason = "; ".join(f"{e.code}: {e.message}" for e in errs) or info.status
            fatal = {"model_not_found", "invalid_request", "unsupported_endpoint"}
            status = "error" if codes & fatal else "retry"
            return {cid: (status, f"batch failed: {reason}") for cid in items}

        for file_id in (info.output_file_id, info.error_file_id):
            if not file_id:
                continue
            for line in (await _read_file_text(self.client, file_id)).splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj.get("custom_id")
                if cid not in items:
                    continue
                resp = obj.get("response") or {}
                err = obj.get("error")
                status_code = resp.get("status_code")
                if status_code == 200:
                    body = resp["body"]
                    try:
                        parsed = OAIResponse.model_validate(body)
                    except Exception:  # noqa: BLE001 — strict schema rejects newer bodies
                        parsed = _to_ns(body)
                    out[cid] = ("ok", parsed)
                elif err is not None:
                    status = "retry" if _openai_item_error_is_retryable(err) else "error"
                    out[cid] = (status, err)
                elif status_code in (429,) or (status_code or 0) >= 500:
                    out[cid] = ("retry", resp.get("body"))
                else:
                    out[cid] = ("error", resp.get("body"))
        # ids absent from both files (expired mid-window) fall through as
        # "missing from batch results" -> retried by the WaveBatcher.
        return out


class BatchStreamingAnthropicClient(StreamingAnthropicClient):
    """StreamingAnthropicClient that transports turns via Message Batches.

    Request bodies come from the same build_create_kwargs as the interactive
    client; responses are the same anthropic Message objects the streaming
    path returns, so prompt/response conversion and scoring are identical.
    """

    def attach_batcher(self, batcher: WaveBatcher) -> None:
        self._batcher = batcher

    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        state = kwargs.pop("state", None)
        create_kwargs = self.build_create_kwargs(prompt, model, sampling_args, tools, **kwargs)
        # On the interactive path extra_body is an SDK transport kwarg whose keys
        # the SDK merges into the JSON body; the Batch API validates params
        # strictly and rejects it as an unknown field. Apply the same merge here
        # so the batch request body is byte-identical to the live one.
        extra_body = create_kwargs.pop("extra_body", None)
        if extra_body:
            create_kwargs.update(extra_body)
        t0 = time.monotonic()
        response = await self._batcher.submit(create_kwargs)
        # Wall time here includes batch queue + turnaround, not pure model time.
        _record_model_call(state, time.monotonic() - t0, response)
        return response


class BatchOpenAIResponsesClient(OpenAIResponsesClient):
    """OpenAIResponsesClient that transports turns via the OpenAI Batch API."""

    def attach_batcher(self, batcher: WaveBatcher) -> None:
        self._batcher = batcher

    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        state = kwargs.pop("state", None)
        call_kwargs = self.build_call_kwargs(prompt, model, sampling_args, tools, **kwargs)
        t0 = time.monotonic()
        try:
            response = await self._batcher.submit(call_kwargs)
        except BatchRequestError as e:
            # Mirror the interactive client's overlong-prompt mapping so the
            # rollout ends the same way it would on the live path.
            text = str(e.payload).lower()
            if "context length" in text or "too long" in text:
                raise OverlongPromptError from e
            raise
        _record_model_call(state, time.monotonic() - t0, response)
        return response
