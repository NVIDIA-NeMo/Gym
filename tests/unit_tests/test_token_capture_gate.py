# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""State-machine, security, retry, and expiry tests for the rollout gate."""

import asyncio
from typing import Any

import pytest

from nemo_gym.token_id_capture.gate import (
    DataCapabilityError,
    GateStateError,
    OperationConflictError,
    RolloutCaptureGate,
)
from nemo_gym.token_id_capture.protocols import LineageMatch
from nemo_gym.token_id_capture.sink import CaptureContext
from nemo_gym.token_id_capture.staging.records import CommitCoords


class _LineageStore:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.fail_record = False

    async def resolve(self, rollout_id: str, request_items: list[dict]) -> LineageMatch | None:
        return None

    async def record(
        self,
        rollout_id: str,
        model_call_id: str,
        request_items: list[dict],
        response_items: list[dict],
        cumulative_token_ids: list[int],
        digest: str,
    ) -> None:
        if self.fail_record:
            raise OSError("lineage unavailable")
        self.records.append(
            {
                "rollout_id": rollout_id,
                "model_call_id": model_call_id,
                "request_items": request_items,
                "response_items": response_items,
                "cumulative_token_ids": cumulative_token_ids,
                "digest": digest,
            }
        )

    async def close(self) -> None:
        return None


def _gate(
    store: _LineageStore | None = None,
    *,
    ttl: float = 60.0,
) -> tuple[RolloutCaptureGate, _LineageStore]:
    actual_store = store or _LineageStore()
    return (
        RolloutCaptureGate(
            lineage_store=actual_store,
            registration_ttl_s=ttl,
            tombstone_ttl_s=60.0,
        ),
        actual_store,
    )


def _context(
    model_call_id: str,
    *,
    parent_call_id: str | None = None,
    parent_tokens: list[int] | None = None,
    store: _LineageStore,
) -> CaptureContext:
    return CaptureContext(
        rollout_id="rollout-1",
        model_call_id=model_call_id,
        token_sink=None,
        lineage_store=store,
        parent_resolved=True,
        parent_call_id=parent_call_id,
        parent_tokens=list(parent_tokens or []),
    )


def _coords(
    model_call_id: str,
    *,
    parent_call_id: str | None = None,
    prev_len: int = 0,
    token_ids: list[int],
    weight_version: int = 7,
) -> CommitCoords:
    return CommitCoords(
        rollout_id="rollout-1",
        model_call_id=model_call_id,
        parent_call_id=parent_call_id,
        prev_len=prev_len,
        delta_len=len(token_ids),
        cum_len=prev_len + len(token_ids),
        weight_version=weight_version,
        disposition="staged",
        digest="a" * 64,
        extras_digest="b" * 64,
        staging_key=f"stage/{model_call_id}",
        token_ids_delta=token_ids,
    )


async def _register(gate: RolloutCaptureGate, *, operation_id: str = "register-1"):
    return await gate.register_rollout(
        "rollout-1",
        owner_id="controller-1",
        operation_id=operation_id,
    )


def test_register_retry_returns_the_identical_capability() -> None:
    async def scenario() -> None:
        gate, _ = _gate()
        first = await _register(gate)
        second = await _register(gate)
        assert first == second
        assert first.data_capability
        assert first.data_capability not in first.capability_id
        with pytest.raises(OperationConflictError):
            await gate.register_rollout(
                "rollout-1",
                owner_id="controller-2",
                operation_id="register-1",
            )

    asyncio.run(scenario())


def test_data_capability_is_required_and_not_present_in_receipt() -> None:
    async def scenario() -> None:
        gate, store = _gate()
        registration = await _register(gate)
        context = _context("root", store=store)
        with pytest.raises(DataCapabilityError):
            await gate.admit_context(
                context,
                data_capability="attacker-token",
                request_items=[{"role": "user", "content": "hello"}],
                logical_request_id="main-request",
            )
        await gate.admit_context(
            context,
            data_capability=registration.data_capability,
            request_items=[{"role": "user", "content": "hello"}],
            logical_request_id="main-request",
        )
        await gate.commit_coords(
            _coords("root", token_ids=[10, 11]),
            response_items=[{"role": "assistant", "content": "hi"}],
        )
        receipt = await gate.seal_rollout(
            "rollout-1",
            owner_id="controller-1",
            operation_id="seal-1",
            reward=1.0,
            terminal_logical_request_id="main-request",
        )
        serialized = str(receipt.model_dump())
        assert registration.data_capability not in serialized
        assert registration.capability_id not in serialized

    asyncio.run(scenario())


def test_commit_publishes_full_lineage_and_child_uses_exact_prefix() -> None:
    async def scenario() -> None:
        gate, store = _gate()
        registration = await _register(gate)
        await gate.admit_context(
            _context("root", store=store),
            data_capability=registration.data_capability,
            request_items=[{"role": "user", "content": "first"}],
            logical_request_id="first",
        )
        assert await gate.commit_coords(
            _coords("root", token_ids=[10, 11, 12]),
            response_items=[{"role": "assistant", "content": "one"}],
        )
        admission = await gate.admit_context(
            _context(
                "child",
                parent_call_id="root",
                parent_tokens=[10, 11, 12],
                store=store,
            ),
            data_capability=registration.data_capability,
            request_items=[{"role": "assistant", "content": "one"}],
            logical_request_id="second",
        )
        assert admission.required_prefix_token_ids == [10, 11, 12]
        assert await gate.commit_coords(
            _coords(
                "child",
                parent_call_id="root",
                prev_len=3,
                token_ids=[20, 21],
            ),
            response_items=[{"role": "assistant", "content": "two"}],
        )
        assert store.records[-1]["cumulative_token_ids"] == [10, 11, 12, 20, 21]
        assert store.records[-1]["request_items"] == [{"role": "assistant", "content": "one"}]

    asyncio.run(scenario())


def test_concurrent_branches_seal_the_trusted_logical_terminal() -> None:
    async def scenario() -> None:
        gate, store = _gate()
        registration = await _register(gate)
        await gate.admit_context(
            _context("root", store=store),
            data_capability=registration.data_capability,
            request_items=[],
            logical_request_id="root-logical",
        )
        await gate.commit_coords(_coords("root", token_ids=[1]), response_items=[])
        for call_id, logical_id in (("main", "main-logical"), ("subagent", "sub-logical")):
            await gate.admit_context(
                _context(call_id, parent_call_id="root", parent_tokens=[1], store=store),
                data_capability=registration.data_capability,
                request_items=[],
                logical_request_id=logical_id,
            )
        await asyncio.gather(
            gate.commit_coords(
                _coords("subagent", parent_call_id="root", prev_len=1, token_ids=[8, 9]),
                response_items=[],
            ),
            gate.commit_coords(
                _coords("main", parent_call_id="root", prev_len=1, token_ids=[2]),
                response_items=[],
            ),
        )
        receipt = await gate.seal_rollout(
            "rollout-1",
            owner_id="controller-1",
            operation_id="seal-main",
            reward=2.0,
            terminal_logical_request_id="main-logical",
        )
        assert receipt.terminal_model_call_id == "main"

    asyncio.run(scenario())


def test_seal_retry_replays_receipt_and_conflicting_payload_is_rejected() -> None:
    async def scenario() -> None:
        gate, store = _gate()
        registration = await _register(gate)
        await gate.admit_context(
            _context("root", store=store),
            data_capability=registration.data_capability,
            request_items=[],
            logical_request_id="main",
        )
        await gate.commit_coords(_coords("root", token_ids=[1]), response_items=[])
        kwargs = {
            "owner_id": "controller-1",
            "operation_id": "seal-1",
            "reward": 1.0,
            "terminal_logical_request_id": "main",
        }
        first = await gate.seal_rollout("rollout-1", **kwargs)
        second = await gate.seal_rollout("rollout-1", **kwargs)
        assert first == second
        with pytest.raises(OperationConflictError):
            await gate.seal_rollout("rollout-1", **{**kwargs, "reward": 2.0})

    asyncio.run(scenario())


def test_invalid_coordinates_and_lineage_failure_poison_capture() -> None:
    async def scenario() -> None:
        store = _LineageStore()
        gate, _ = _gate(store)
        registration = await _register(gate)
        await gate.admit_context(
            _context("root", store=store),
            data_capability=registration.data_capability,
            request_items=[],
            logical_request_id="main",
        )
        invalid = _coords("root", token_ids=[1]).model_copy(update={"delta_len": 2})
        with pytest.raises(GateStateError, match="inconsistent lengths"):
            await gate.commit_coords(invalid, response_items=[])
        store.fail_record = True
        assert not await gate.commit_coords(_coords("root", token_ids=[1]), response_items=[])
        receipt = await gate.seal_rollout(
            "rollout-1",
            owner_id="controller-1",
            operation_id="seal-1",
            reward=None,
            terminal_logical_request_id="main",
        )
        assert receipt.capture_poisoned

    asyncio.run(scenario())


def test_fail_retry_returns_identical_cleanup_manifest() -> None:
    async def scenario() -> None:
        gate, store = _gate()
        registration = await _register(gate)
        await gate.admit_context(
            _context("root", store=store),
            data_capability=registration.data_capability,
            request_items=[],
        )
        await gate.commit_coords(_coords("root", token_ids=[1]), response_items=[])
        kwargs = {
            "owner_id": "controller-1",
            "operation_id": "fail-1",
            "reason": "controller_abort",
        }
        first = await gate.fail_rollout("rollout-1", **kwargs)
        second = await gate.fail_rollout("rollout-1", **kwargs)
        assert first == second
        assert first.staging_keys == ["stage/root"]

    asyncio.run(scenario())


def test_live_ttl_expiry_produces_cleanup_work(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        now = 1_000.0
        monkeypatch.setattr("nemo_gym.token_id_capture.gate.time.time", lambda: now)
        gate, store = _gate(ttl=5.0)
        registration = await _register(gate)
        await gate.admit_context(
            _context("root", store=store),
            data_capability=registration.data_capability,
            request_items=[],
        )
        await gate.commit_coords(_coords("root", token_ids=[1]), response_items=[])
        now = 1_006.0
        expired = await gate.expire_stale()
        assert expired[0].reason == "registration_expired"
        assert expired[0].staging_keys == ["stage/root"]
        assert await gate.drain_cleanup_manifests() == expired

    asyncio.run(scenario())
