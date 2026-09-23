# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic generation-cut lifecycle and race tests."""

import asyncio

import pytest

from nemo_gym._checkpoint import (
    GATED_MODEL_ROUTE_SUFFIXES,
    AdmissionLimiter,
    AdmissionMiddleware,
    GenerationCutInventory,
    GenerationCutPrefix,
    GenerationCutPrefixAck,
    GenerationCutReceipt,
    bind_current_model_call,
    mark_current_generation_safe,
    mark_current_generation_started,
)
from nemo_gym.token_id_capture import (
    CaptureContext,
    mark_external_staging_committed,
    reset_token_sink,
    set_token_sink,
)


class _FakeCutBackend:
    def __init__(self, *, block: bool = False, disposition: str = "durable_prefix") -> None:
        self.block = block
        self.disposition = disposition
        self.calls: list[GenerationCutInventory] = []
        self.release = asyncio.Event()

    async def checkpoint_generation_cut(self, inventory: GenerationCutInventory) -> GenerationCutReceipt:
        self.calls.append(inventory)
        if self.block:
            await self.release.wait()
        return GenerationCutReceipt(
            checkpoint_id=inventory.checkpoint_id,
            cut_id=f"cut-{len(self.calls)}",
            inventory_digest=inventory.inventory_digest,
            inventory=inventory,
            backend_snapshot_id="snapshot-1",
            prefixes=tuple(
                GenerationCutPrefixAck(
                    ticket_id=prefix.ticket_id,
                    rollout_id=prefix.rollout_id,
                    attempt_index=prefix.attempt_index,
                    model_call_id=prefix.model_call_id,
                    admitted_at=prefix.admitted_at,
                    disposition=self.disposition,
                    **(
                        {
                            "frozen_buffer_id": f"buffer/{prefix.ticket_id}",
                            "staging_key": f"prefix/{prefix.ticket_id}",
                            "prefix_token_count": 3,
                            "prefix_digest": "a" * 64,
                        }
                        if self.disposition == "durable_prefix"
                        else {}
                    ),
                )
                for prefix in inventory.active_prefixes
            ),
        )

    async def restore_generation_cut(self, receipt):
        raise AssertionError("restore is not used by admission tests")


def _scope() -> dict:
    return {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "headers": [
            (b"x-nemo-gym-rollout-id", b"rollout-1"),
            (b"x-nemo-gym-attempt-index", b"0"),
        ],
    }


@pytest.mark.asyncio
async def test_pre_generation_ticket_waits_for_no_generation_exit() -> None:
    backend = _FakeCutBackend()
    limiter = AdmissionLimiter(backend)
    entered = asyncio.Event()
    finish = asyncio.Event()
    messages = []

    async def app(scope, receive, send) -> None:
        entered.set()
        await finish.wait()
        mark_current_generation_started()

    async def send(message) -> None:
        messages.append(message)

    task = asyncio.create_task(AdmissionMiddleware(app, limiter, GATED_MODEL_ROUTE_SUFFIXES)(_scope(), None, send))
    await entered.wait()
    limiter.close("ckpt-1")

    assert not await limiter.prepare_generation_cut("ckpt-1", server_name="policy", timeout_s=1)
    assert backend.calls == []
    assert limiter.counts()["generation_pending_total"] == 1

    finish.set()
    await task
    assert limiter.state.value == "paused"
    assert limiter.counts()["inflight_total"] == 0
    assert messages[0]["status"] == 409
    assert b"checkpoint_parked" in messages[1]["body"]


@pytest.mark.asyncio
async def test_mid_decode_buffer_cut_is_safe_before_response_exit() -> None:
    backend = _FakeCutBackend()
    limiter = AdmissionLimiter(backend)
    decoding = asyncio.Event()
    finish = asyncio.Event()

    async def app(scope, receive, send) -> None:
        bind_current_model_call("call-1")
        mark_current_generation_started()
        decoding.set()
        await finish.wait()

    task = asyncio.create_task(AdmissionMiddleware(app, limiter, GATED_MODEL_ROUTE_SUFFIXES)(_scope(), None, None))
    await decoding.wait()
    limiter.close("ckpt-1")

    assert await limiter.prepare_generation_cut("ckpt-1", server_name="policy", timeout_s=1)
    assert limiter.state.value == "paused"
    assert limiter.counts()["response_inflight_total"] == 1
    assert limiter.counts()["generation_pending_total"] == 0

    finish.set()
    await task


@pytest.mark.asyncio
async def test_completed_capture_before_egress_is_prepare_safe() -> None:
    backend = _FakeCutBackend()
    limiter = AdmissionLimiter(backend)
    durable = asyncio.Event()
    finish_response = asyncio.Event()

    async def app(scope, receive, send) -> None:
        mark_current_generation_started()
        capture_token = set_token_sink(
            CaptureContext(
                rollout_id="rollout-1",
                model_call_id="call-1",
                token_sink=None,
                external_staging=True,
            )
        )
        try:
            mark_external_staging_committed(rollout_id="rollout-1", model_call_id="call-1")
        finally:
            reset_token_sink(capture_token)
        durable.set()
        await finish_response.wait()

    task = asyncio.create_task(AdmissionMiddleware(app, limiter, GATED_MODEL_ROUTE_SUFFIXES)(_scope(), None, None))
    await durable.wait()
    limiter.close("ckpt-1")

    assert limiter.state.value == "paused"
    assert limiter.counts()["response_inflight_total"] == 1
    assert backend.calls == []

    finish_response.set()
    await task


@pytest.mark.asyncio
async def test_duplicate_cut_request_is_idempotent() -> None:
    backend = _FakeCutBackend(block=True)
    limiter = AdmissionLimiter(backend)
    ticket = limiter.admit(rollout_id="r", attempt_index=0)
    ticket.generation_started = True
    ticket.model_call_id = "call-1"
    limiter.close("ckpt-1")

    first = asyncio.create_task(limiter.prepare_generation_cut("ckpt-1", server_name="policy", timeout_s=1))
    await asyncio.sleep(0)
    second = asyncio.create_task(limiter.prepare_generation_cut("ckpt-1", server_name="policy", timeout_s=1))
    backend.release.set()

    assert await first
    assert await second
    assert len(backend.calls) == 1
    limiter.release(ticket)


@pytest.mark.asyncio
async def test_cut_timeout_leaves_ticket_pending_until_abort() -> None:
    backend = _FakeCutBackend(block=True)
    limiter = AdmissionLimiter(backend)
    ticket = limiter.admit(rollout_id="r", attempt_index=2)
    ticket.generation_started = True
    ticket.model_call_id = "call-1"
    limiter.close("ckpt-1")

    assert not await limiter.prepare_generation_cut("ckpt-1", server_name="policy", timeout_s=0.01)
    assert limiter.state.value == "draining"
    assert limiter.counts()["generation_pending_total"] == 1

    limiter.abort_inflight("r", 2)
    assert limiter.state.value == "draining"
    assert limiter.counts()["response_inflight_total"] == 1
    limiter.release(ticket)
    assert limiter.state.value == "paused"


@pytest.mark.asyncio
async def test_durable_failure_is_safe_before_response_exit() -> None:
    backend = _FakeCutBackend()
    limiter = AdmissionLimiter(backend)
    failed = asyncio.Event()
    finish = asyncio.Event()

    async def app(scope, receive, send) -> None:
        mark_current_generation_started()
        mark_current_generation_safe("durable_failure")
        failed.set()
        await finish.wait()

    task = asyncio.create_task(AdmissionMiddleware(app, limiter, GATED_MODEL_ROUTE_SUFFIXES)(_scope(), None, None))
    await failed.wait()
    limiter.close("ckpt-1")

    assert limiter.state.value == "paused"
    assert limiter.counts()["response_inflight_total"] == 1
    finish.set()
    await task


@pytest.mark.asyncio
async def test_backend_can_ack_explicit_durable_failure() -> None:
    backend = _FakeCutBackend(disposition="durable_failure")
    limiter = AdmissionLimiter(backend)
    ticket = limiter.admit(rollout_id="r", attempt_index=0)
    ticket.generation_started = True
    ticket.model_call_id = "call-1"
    limiter.close("ckpt-1")

    assert await limiter.prepare_generation_cut("ckpt-1", server_name="policy", timeout_s=1)
    assert ticket.prepare_safe_reason == "durable_failure"
    limiter.release(ticket)


@pytest.mark.asyncio
async def test_close_before_response_start_fences_egress_until_resume() -> None:
    backend = _FakeCutBackend()
    limiter = AdmissionLimiter(backend)
    ready = asyncio.Event()
    send_response = asyncio.Event()
    messages = []

    async def app(scope, receive, send) -> None:
        bind_current_model_call("call-1")
        mark_current_generation_started()
        mark_current_generation_safe("durable_completed")
        ready.set()
        await send_response.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"complete", "more_body": False})

    async def send(message) -> None:
        messages.append(message)

    task = asyncio.create_task(AdmissionMiddleware(app, limiter, GATED_MODEL_ROUTE_SUFFIXES)(_scope(), None, send))
    await ready.wait()
    limiter.close("ckpt-1")
    send_response.set()
    await asyncio.sleep(0)

    assert limiter.state.value == "paused"
    assert messages == []
    assert not task.done()

    limiter.resume()
    await task
    assert [message["type"] for message in messages] == ["http.response.start", "http.response.body"]


@pytest.mark.asyncio
async def test_response_start_before_close_forces_full_response_drain() -> None:
    backend = _FakeCutBackend()
    limiter = AdmissionLimiter(backend)
    started = asyncio.Event()
    finish = asyncio.Event()
    messages = []

    async def app(scope, receive, send) -> None:
        bind_current_model_call("call-1")
        mark_current_generation_started()
        mark_current_generation_safe("durable_completed")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        started.set()
        await finish.wait()
        await send({"type": "http.response.body", "body": b"complete", "more_body": False})

    async def send(message) -> None:
        messages.append(message)

    task = asyncio.create_task(AdmissionMiddleware(app, limiter, GATED_MODEL_ROUTE_SUFFIXES)(_scope(), None, send))
    await started.wait()
    limiter.close("ckpt-1")

    assert not await limiter.prepare_generation_cut("ckpt-1", server_name="policy", timeout_s=1)
    assert backend.calls == []
    assert limiter.state.value == "draining"

    finish.set()
    await task
    assert limiter.state.value == "paused"
    assert [message["type"] for message in messages] == ["http.response.start", "http.response.body"]


@pytest.mark.asyncio
async def test_abort_cancels_response_waiting_at_egress_fence() -> None:
    backend = _FakeCutBackend()
    limiter = AdmissionLimiter(backend)
    ready = asyncio.Event()
    send_response = asyncio.Event()
    messages = []

    async def app(scope, receive, send) -> None:
        bind_current_model_call("call-1")
        mark_current_generation_started()
        mark_current_generation_safe("durable_completed")
        ready.set()
        await send_response.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def send(message) -> None:
        messages.append(message)

    task = asyncio.create_task(AdmissionMiddleware(app, limiter, GATED_MODEL_ROUTE_SUFFIXES)(_scope(), None, send))
    await ready.wait()
    limiter.close("ckpt-1")
    send_response.set()
    await asyncio.sleep(0)
    assert messages == []

    limiter.abort_inflight("rollout-1", 0)
    await task
    assert messages[0]["status"] == 409
    assert b"stale_attempt" in messages[1]["body"]
    assert all(message.get("status") != 200 for message in messages)


@pytest.mark.asyncio
async def test_abort_stays_pending_until_cancelled_task_exits() -> None:
    limiter = AdmissionLimiter(_FakeCutBackend())
    running = asyncio.Event()
    cancellation_caught = asyncio.Event()
    allow_exit = asyncio.Event()

    async def app(scope, receive, send) -> None:
        bind_current_model_call("call-1")
        mark_current_generation_started()
        mark_current_generation_safe("durable_completed")
        running.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_caught.set()
            mark_current_generation_safe("durable_failure")
            await allow_exit.wait()

    task = asyncio.create_task(AdmissionMiddleware(app, limiter, GATED_MODEL_ROUTE_SUFFIXES)(_scope(), None, None))
    await running.wait()
    limiter.close("ckpt-1")
    assert limiter.state.value == "paused"

    limiter.abort_inflight("rollout-1", 0)
    await cancellation_caught.wait()
    assert not task.done()
    assert not limiter.is_prepare_safe()
    assert limiter.state.value == "draining"
    assert limiter.counts()["generation_pending_total"] == 1

    allow_exit.set()
    await task
    assert limiter.is_prepare_safe()
    assert limiter.state.value == "paused"


@pytest.mark.asyncio
async def test_generated_return_without_egress_fails_closed() -> None:
    limiter = AdmissionLimiter(_FakeCutBackend())
    generated = asyncio.Event()
    finish = asyncio.Event()

    async def app(scope, receive, send) -> None:
        bind_current_model_call("call-1")
        mark_current_generation_started()
        generated.set()
        await finish.wait()

    task = asyncio.create_task(AdmissionMiddleware(app, limiter, GATED_MODEL_ROUTE_SUFFIXES)(_scope(), None, None))
    await generated.wait()
    limiter.close("ckpt-1")
    finish.set()
    await task

    assert limiter.counts()["response_inflight_total"] == 0
    assert limiter.counts()["generation_pending_total"] == 1
    assert limiter.state.value == "draining"
    assert limiter.abort_inflight("rollout-1", 0)
    assert limiter.state.value == "paused"


def test_receipt_rejects_logical_identity_different_from_inventory() -> None:
    admitted = GenerationCutPrefix(
        ticket_id="ticket-1",
        rollout_id="rollout-1",
        attempt_index=2,
        model_call_id="call-1",
        admitted_at=1.0,
    )
    inventory = GenerationCutInventory.build(
        checkpoint_id="ckpt-1",
        server_name="policy",
        active_prefixes=[admitted],
    )
    ack = GenerationCutReceipt(
        checkpoint_id="ckpt-1",
        cut_id="cut-1",
        inventory_digest=inventory.inventory_digest,
        inventory=inventory,
        backend_snapshot_id="snapshot-1",
        prefixes=(
            GenerationCutPrefixAck(
                ticket_id="ticket-1",
                rollout_id="rollout-1",
                attempt_index=2,
                model_call_id="wrong-call",
                admitted_at=1.0,
                disposition="durable_prefix",
                frozen_buffer_id="buffer-1",
                staging_key="staging-1",
                prefix_token_count=3,
                prefix_digest="c" * 64,
            ),
        ),
    )

    with pytest.raises(ValueError, match="identity differs"):
        ack.validate_for(inventory)
