# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only stress benchmark for Gym checkpoint control and model lineage.

This benchmark deliberately avoids Ray, vLLM, TQ token payloads, model
weights, and GPUs.  It exercises the control-plane pieces that can become
expensive independently of decoding:

* ``/run`` completion receipt creation and bounded bulk acknowledgements;
* checkpoint-triggered ACK flushing while terminal completions keep arriving;
* separate HTTP connection pools for data and acknowledgement traffic;
* mixed multi-worker generation-cut journals over the real coordinator socket;
* coordinator evidence validation, model-ledger commit, and ledger restore.

Use a Lustre-backed ``--root`` to measure the filesystem behavior seen by a
cluster job.  The default temporary directory is useful only for correctness
and CPU profiling.

Examples::

    python scripts/benchmark_checkpoint_cpu.py --profile smoke
    python scripts/benchmark_checkpoint_cpu.py --profile micro \
        --root /lustre/path/to/checkpoint-cpu-benchmark --keep-artifacts
    python scripts/benchmark_checkpoint_cpu.py --profile scale \
        --components all --runs 131072 --cuts 131072 --workers 8 --agents 32
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import resource
import shutil
import socket
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import aiohttp
import uvicorn
from fastapi import FastAPI, Header, Response
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym._checkpoint.agent import (
    AGENT_ACKNOWLEDGEMENT_BATCH_MAX_RECEIPTS,
    AGENT_CHECKPOINT_URL_PREFIX,
    AGENT_COMPLETION_RECEIPT_HEADER,
    AgentAcknowledgeBatchRequest,
    AgentAcknowledgeRequest,
    AgentCheckpointParticipant,
    agent_acknowledgement_batch_digest,
    decode_agent_completion_receipt,
    encode_agent_completion_receipt,
)
from nemo_gym._checkpoint.coordinator import AdmissionCoordinator, AdmissionLimiter, WorkerAdmissionAgent
from nemo_gym._checkpoint.ledger import CaptureLedgerCheckpointer, PolicyModelCheckpointCoordinatorService
from nemo_gym._checkpoint.model_control_contracts import (
    GenerationCutInventory,
    GenerationCutPrefixAck,
    GenerationCutReceipt,
)
from nemo_gym.token_id_capture.control_routes import require_control_auth
from nemo_gym.token_id_capture.lineage import FileLineageStore


_AUTH_TOKEN = "checkpoint-cpu-benchmark"
_SERVER_NAME = "policy"


@dataclass(frozen=True)
class Profile:
    runs: int
    cuts: int
    run_concurrency: int
    data_connections: int
    control_connections: int
    agents: int
    completion_batch_size: int


_PROFILES = {
    "micro": Profile(
        runs=1_000,
        cuts=3_000,
        run_concurrency=128,
        data_connections=128,
        control_connections=16,
        agents=4,
        completion_batch_size=64,
    ),
    "smoke": Profile(
        runs=2_000,
        cuts=3_000,
        run_concurrency=256,
        data_connections=256,
        control_connections=32,
        agents=8,
        completion_batch_size=128,
    ),
    "scale": Profile(
        runs=131_072,
        cuts=131_072,
        run_concurrency=4_096,
        data_connections=4_096,
        control_connections=128,
        agents=32,
        completion_batch_size=1_024,
    ),
}


class _RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollout_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    attempt_index: int = Field(default=0, ge=0)
    delay_ms: float = Field(default=0.0, ge=0.0)


class _SyntheticCutBackend:
    def __init__(self, worker_id: str, *, prefix_tokens: int, staging_key_bytes: int) -> None:
        self.worker_id = worker_id
        self.prefix_tokens = prefix_tokens
        self.staging_key_bytes = staging_key_bytes

    async def checkpoint_generation_cut(self, inventory: GenerationCutInventory) -> GenerationCutReceipt:
        # A real generation backend crosses a transport boundary. Yield once so
        # benchmark-side terminal transitions can race with cut collection in
        # the same ordering they have in production.
        await asyncio.sleep(0)
        prefixes = []
        padding = "x" * self.staging_key_bytes
        for prefix in inventory.active_prefixes:
            digest = hashlib.sha256(prefix.ticket_id.encode()).hexdigest()
            prefixes.append(
                GenerationCutPrefixAck(
                    **prefix.model_dump(mode="json"),
                    disposition="durable_prefix",
                    cut_kind="active_prefix",
                    frozen_buffer_id=f"active/{inventory.checkpoint_id}/{prefix.ticket_id}",
                    staging_keys=(f"prefix/{prefix.ticket_id}/{padding}",),
                    prefix_token_count=self.prefix_tokens,
                    prefix_digest=digest,
                    effective_output_limit=max(self.prefix_tokens + 1, 128),
                )
            )
        return GenerationCutReceipt(
            checkpoint_id=inventory.checkpoint_id,
            cut_id=f"cut-{self.worker_id}-{inventory.inventory_digest[:16]}",
            inventory_digest=inventory.inventory_digest,
            inventory=inventory,
            backend_snapshot_id=f"snapshot-{self.worker_id}-{inventory.inventory_digest[:16]}",
            prefixes=tuple(prefixes),
        )


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
        return ordered[index] * 1_000

    return {
        "count": len(values),
        "mean_ms": sum(values) * 1_000 / len(values),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": ordered[-1] * 1_000,
    }


def _worker_counts(total: int, workers: int, hot_worker_fraction: float) -> list[int]:
    if workers == 1:
        return [total]
    minimum = 1.0 / workers
    if not minimum <= hot_worker_fraction <= 1.0:
        raise ValueError(f"hot-worker fraction must be in [{minimum:.6f}, 1.0] for {workers} workers")
    hot = min(total, round(total * hot_worker_fraction))
    remaining = total - hot
    counts = [hot]
    counts.extend(remaining // (workers - 1) for _ in range(workers - 1))
    for index in range(remaining % (workers - 1)):
        counts[index + 1] += 1
    return counts


def _unused_tcp_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _control_app(participant: AgentCheckpointParticipant) -> FastAPI:
    app = FastAPI()

    @app.post("/run")
    async def run(body: _RunRequest, response: Response) -> dict[str, Any]:
        execution = await participant.begin(
            body.rollout_id,
            body.attempt_index,
            task=asyncio.current_task(),
        )
        if body.delay_ms:
            await asyncio.sleep(body.delay_ms / 1_000)
        result = {
            "rollout_id": body.rollout_id,
            "attempt_index": body.attempt_index,
            "reward": 1.0,
        }
        await participant.finish(execution, outcome="completed", result=result)
        receipt = participant.completion_receipt(body.rollout_id, body.attempt_index)
        response.headers[AGENT_COMPLETION_RECEIPT_HEADER] = encode_agent_completion_receipt(receipt)
        return result

    @app.post(f"{AGENT_CHECKPOINT_URL_PREFIX}/acknowledge-batch")
    async def acknowledge_batch(
        body: AgentAcknowledgeBatchRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_control_auth(authorization, _AUTH_TOKEN)
        return await participant.acknowledge_batch(body)

    return app


async def _start_tcp_app(app: FastAPI) -> tuple[str, uvicorn.Server, asyncio.Task[None]]:
    port = _unused_tcp_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
    )
    task = asyncio.create_task(server.serve())
    deadline = time.monotonic() + 10
    while not server.started:
        if task.done():
            await task
            raise RuntimeError("CPU benchmark HTTP server exited before startup")
        if time.monotonic() >= deadline:
            server.should_exit = True
            await task
            raise TimeoutError("CPU benchmark HTTP server did not start")
        await asyncio.sleep(0.01)
    return f"http://127.0.0.1:{port}", server, task


async def run_control_benchmark(
    *,
    runs: int,
    concurrency: int,
    ack_batch_size: int,
    ack_concurrency: int,
    ack_coalesce_ms: float,
    delay_ms: float,
    data_connections: int,
    control_connections: int,
    transport: Literal["tcp", "asgi"],
) -> dict[str, Any]:
    """Exercise real completion and bulk-ACK state with synthetic ``/run`` work."""
    if runs < 1 or concurrency < 1 or ack_concurrency < 1 or ack_coalesce_ms < 0:
        raise ValueError("runs and concurrency settings must be positive")
    if not 1 <= ack_batch_size <= AGENT_ACKNOWLEDGEMENT_BATCH_MAX_RECEIPTS:
        raise ValueError(f"ack batch size must be between 1 and {AGENT_ACKNOWLEDGEMENT_BATCH_MAX_RECEIPTS}")

    participant = AgentCheckpointParticipant("checkpoint-cpu-benchmark")
    app = _control_app(participant)
    server: uvicorn.Server | None = None
    server_task: asyncio.Task[None] | None = None
    if transport == "tcp":
        base_url, server, server_task = await _start_tcp_app(app)
    else:
        base_url = "http://checkpoint-cpu-benchmark"

    receipt_queue: asyncio.Queue[AgentAcknowledgeRequest | None] = asyncio.Queue(
        maxsize=max(concurrency * 2, ack_batch_size * ack_concurrency * 2)
    )
    ack_batch_queue: asyncio.Queue[list[AgentAcknowledgeRequest] | None] = asyncio.Queue(
        maxsize=max(ack_concurrency * 2, 1)
    )
    run_latencies: list[float] = []
    ack_latencies: list[float] = []
    ack_batch_sizes: list[int] = []
    next_run = 0
    ack_calls = 0
    acknowledged = 0
    maximum_ack_queue = 0

    data_client: aiohttp.ClientSession | None = None
    control_client: aiohttp.ClientSession | None = None
    if transport == "tcp":
        data_client = aiohttp.ClientSession(
            base_url=base_url,
            connector=aiohttp.TCPConnector(
                limit=data_connections,
                limit_per_host=data_connections,
            ),
            timeout=aiohttp.ClientTimeout(total=120.0),
        )
        control_client = aiohttp.ClientSession(
            base_url=base_url,
            connector=aiohttp.TCPConnector(
                limit=control_connections,
                limit_per_host=control_connections,
            ),
            headers={"authorization": f"Bearer {_AUTH_TOKEN}"},
            timeout=aiohttp.ClientTimeout(total=120.0),
        )
    started = time.monotonic()
    try:

        async def dispatch_runs() -> None:
            nonlocal next_run, maximum_ack_queue
            while True:
                index = next_run
                next_run += 1
                if index >= runs:
                    return
                call_started = time.monotonic()
                request = _RunRequest(
                    rollout_id=f"run-{index:09d}",
                    attempt_index=0,
                    delay_ms=delay_ms,
                )
                if data_client is None:
                    execution = await participant.begin(
                        request.rollout_id,
                        request.attempt_index,
                        task=asyncio.current_task(),
                    )
                    if request.delay_ms:
                        await asyncio.sleep(request.delay_ms / 1_000)
                    await participant.finish(
                        execution,
                        outcome="completed",
                        result={
                            "rollout_id": request.rollout_id,
                            "attempt_index": request.attempt_index,
                            "reward": 1.0,
                        },
                    )
                    receipt = participant.completion_receipt(
                        request.rollout_id,
                        request.attempt_index,
                    )
                else:
                    async with data_client.post(
                        "/run",
                        json=request.model_dump(mode="json"),
                    ) as response:
                        response.raise_for_status()
                        await response.read()
                        encoded = response.headers.get(AGENT_COMPLETION_RECEIPT_HEADER)
                    if encoded is None:
                        raise RuntimeError("synthetic /run omitted its completion receipt")
                    receipt = decode_agent_completion_receipt(encoded)
                await receipt_queue.put(receipt)
                maximum_ack_queue = max(maximum_ack_queue, receipt_queue.qsize())
                run_latencies.append(time.monotonic() - call_started)

        async def coalesce_acknowledgements() -> None:
            """Build batches in one place so concurrent senders cannot fragment them."""
            while True:
                first = await receipt_queue.get()
                if first is None:
                    break
                receipts = [first]
                deadline = time.monotonic() + ack_coalesce_ms / 1_000
                while len(receipts) < ack_batch_size:
                    if ack_coalesce_ms == 0:
                        try:
                            item = receipt_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            item = await asyncio.wait_for(receipt_queue.get(), timeout=remaining)
                        except TimeoutError:
                            break
                    if item is None:
                        await ack_batch_queue.put(receipts)
                        for _ in range(ack_concurrency):
                            await ack_batch_queue.put(None)
                        return
                    receipts.append(item)
                await ack_batch_queue.put(receipts)

            for _ in range(ack_concurrency):
                await ack_batch_queue.put(None)

        async def send_acknowledgements() -> None:
            nonlocal ack_calls, acknowledged
            while True:
                receipts = await ack_batch_queue.get()
                if receipts is None:
                    return
                request = AgentAcknowledgeBatchRequest(
                    receipts=receipts,
                    batch_digest=agent_acknowledgement_batch_digest(receipts),
                )
                call_started = time.monotonic()
                if control_client is None:
                    payload = await participant.acknowledge_batch(request)
                else:
                    async with control_client.post(
                        f"{AGENT_CHECKPOINT_URL_PREFIX}/acknowledge-batch",
                        json=request.model_dump(mode="json"),
                    ) as response:
                        response.raise_for_status()
                        payload = await response.json()
                ack_calls += 1
                acknowledged += int(payload["newly_acknowledged_count"])
                ack_latencies.append(time.monotonic() - call_started)
                ack_batch_sizes.append(len(receipts))

        coalescer_task = asyncio.create_task(coalesce_acknowledgements())
        ack_tasks = [asyncio.create_task(send_acknowledgements()) for _ in range(ack_concurrency)]
        await asyncio.gather(*(dispatch_runs() for _ in range(min(concurrency, runs))))
        await receipt_queue.put(None)
        await coalescer_task
        await asyncio.gather(*ack_tasks)
    finally:
        if data_client is not None:
            await data_client.close()
        if control_client is not None:
            await control_client.close()
        if server is not None and server_task is not None:
            server.should_exit = True
            await server_task

    elapsed = time.monotonic() - started
    status = participant.status()
    if acknowledged != runs or status["active"] != 0 or status["completed_unacknowledged"] != 0:
        raise RuntimeError(
            f"control benchmark leaked completion state: runs={runs}, acknowledged={acknowledged}, status={status!r}"
        )
    return {
        "runs": runs,
        "transport": transport,
        "concurrency": concurrency,
        "data_connections": data_connections,
        "control_connections": control_connections,
        "ack_batch_size": ack_batch_size,
        "ack_concurrency": ack_concurrency,
        "ack_coalesce_ms": ack_coalesce_ms,
        "ack_calls": ack_calls,
        "acknowledged": acknowledged,
        "average_ack_batch_size": sum(ack_batch_sizes) / len(ack_batch_sizes),
        "minimum_ack_batch_size": min(ack_batch_sizes),
        "maximum_ack_batch_size": max(ack_batch_sizes),
        "maximum_ack_queue": maximum_ack_queue,
        "elapsed_seconds": elapsed,
        "runs_per_second": runs / elapsed,
        "run_latency": _latency_summary(run_latencies),
        "ack_latency": _latency_summary(ack_latencies),
    }


class _CheckpointAwareAcknowledgementDrain:
    """Production-shaped bounded ACK coalescing for the CPU stress harness."""

    def __init__(
        self,
        participants: dict[str, AgentCheckpointParticipant],
        *,
        batch_size: int,
        concurrency: int,
        normal_coalesce_ms: float,
        checkpoint_coalesce_ms: float,
    ) -> None:
        self._participants = participants
        self._batch_size = batch_size
        self._concurrency = concurrency
        self._normal_coalesce_s = normal_coalesce_ms / 1_000
        self._checkpoint_coalesce_s = checkpoint_coalesce_ms / 1_000
        self._pending: dict[str, deque[AgentAcknowledgeRequest]] = defaultdict(deque)
        self._condition = asyncio.Condition()
        self._checkpoint_active = False
        self._flush_requested = False
        self._producers_finished = False
        self.normal_wait_started = asyncio.Event()
        self.ack_calls = 0
        self.acknowledged = 0
        self.ack_batch_sizes: list[int] = []
        self.ack_latencies: list[float] = []
        self.maximum_pending = 0
        self.checkpoint_started_at: float | None = None
        self.first_checkpoint_flush_at: float | None = None

    def _pending_count(self) -> int:
        return sum(len(receipts) for receipts in self._pending.values())

    async def add(self, agent_name: str, receipt: AgentAcknowledgeRequest) -> None:
        async with self._condition:
            self._pending[agent_name].append(receipt)
            pending = self._pending_count()
            self.maximum_pending = max(self.maximum_pending, pending)
            if pending >= self._batch_size:
                self._flush_requested = True
            self._condition.notify_all()

    async def begin_checkpoint(self) -> None:
        """Interrupt the normal timer and use the short prepare-time cadence."""
        async with self._condition:
            self._checkpoint_active = True
            self._flush_requested = True
            self.checkpoint_started_at = time.monotonic()
            self._condition.notify_all()

    async def finish_producing(self) -> None:
        async with self._condition:
            self._producers_finished = True
            self._flush_requested = True
            self._condition.notify_all()

    async def _next_flush(self) -> dict[str, list[AgentAcknowledgeRequest]] | None:
        async with self._condition:
            while self._pending_count() == 0:
                if self._producers_finished:
                    return None
                await self._condition.wait()

            timeout_s = self._checkpoint_coalesce_s if self._checkpoint_active else self._normal_coalesce_s
            deadline = time.monotonic() + timeout_s
            if not self._checkpoint_active:
                self.normal_wait_started.set()
            while (
                not self._flush_requested and not self._producers_finished and self._pending_count() < self._batch_size
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except TimeoutError:
                    break

            grouped = {agent_name: list(receipts) for agent_name, receipts in self._pending.items() if receipts}
            self._pending.clear()
            self._flush_requested = False
            if self._checkpoint_active and self.first_checkpoint_flush_at is None:
                self.first_checkpoint_flush_at = time.monotonic()
            return grouped

    async def _acknowledge_agent(
        self,
        semaphore: asyncio.Semaphore,
        agent_name: str,
        receipts: list[AgentAcknowledgeRequest],
    ) -> None:
        participant = self._participants[agent_name]
        async with semaphore:
            for offset in range(0, len(receipts), self._batch_size):
                chunk = receipts[offset : offset + self._batch_size]
                request = AgentAcknowledgeBatchRequest(
                    receipts=chunk,
                    batch_digest=agent_acknowledgement_batch_digest(chunk),
                )
                started = time.monotonic()
                result = await participant.acknowledge_batch(request)
                self.ack_latencies.append(time.monotonic() - started)
                self.ack_calls += 1
                self.acknowledged += int(result["newly_acknowledged_count"])
                self.ack_batch_sizes.append(len(chunk))

    async def run(self) -> None:
        while True:
            grouped = await self._next_flush()
            if grouped is None:
                return
            semaphore = asyncio.Semaphore(self._concurrency)
            await asyncio.gather(
                *(
                    self._acknowledge_agent(semaphore, agent_name, receipts)
                    for agent_name, receipts in sorted(grouped.items())
                )
            )


def _open_file_descriptor_count() -> int | None:
    try:
        return len(tuple(Path("/proc/self/fd").iterdir()))
    except OSError:
        return None


async def run_checkpoint_overlap_benchmark(
    *,
    runs: int,
    agents: int,
    hot_agent_fraction: float,
    completion_batch_size: int,
    completion_interval_ms: float,
    ack_batch_size: int,
    ack_concurrency: int,
    normal_ack_coalesce_ms: float,
    checkpoint_ack_coalesce_ms: float,
    progress_interval_s: float,
    timeout_s: float,
) -> dict[str, Any]:
    """Complete and ACK many executions while agent prepare is in progress."""
    if runs < 1 or agents < 1 or completion_batch_size < 1:
        raise ValueError("runs, agents, and completion batch size must be positive")
    if agents > runs:
        raise ValueError("agents cannot exceed runs")
    if completion_interval_ms < 0 or progress_interval_s <= 0 or timeout_s <= 0:
        raise ValueError("timing settings must be non-negative and timeout must be positive")
    if not 1 <= ack_batch_size <= AGENT_ACKNOWLEDGEMENT_BATCH_MAX_RECEIPTS:
        raise ValueError(f"ack batch size must be between 1 and {AGENT_ACKNOWLEDGEMENT_BATCH_MAX_RECEIPTS}")

    agent_counts = _worker_counts(runs, agents, hot_agent_fraction)
    participants = {f"agent-{index:03d}": AgentCheckpointParticipant(f"agent-{index:03d}") for index in range(agents)}
    executions: dict[str, list[Any]] = {}
    rollout_index = 0
    setup_started = time.monotonic()
    for (agent_name, participant), count in zip(participants.items(), agent_counts, strict=True):
        agent_executions = []
        for _ in range(count):
            agent_executions.append(
                await participant.begin(
                    f"overlap-{rollout_index:09d}",
                    0,
                    task=None,
                )
            )
            rollout_index += 1
        executions[agent_name] = agent_executions
    setup_seconds = time.monotonic() - setup_started

    drain = _CheckpointAwareAcknowledgementDrain(
        participants,
        batch_size=ack_batch_size,
        concurrency=ack_concurrency,
        normal_coalesce_ms=normal_ack_coalesce_ms,
        checkpoint_coalesce_ms=checkpoint_ack_coalesce_ms,
    )
    drain_task = asyncio.create_task(drain.run())

    # Seed one completion under the normal timer, then prove checkpoint start
    # wakes that waiter instead of sleeping for the rest of the one-second window.
    first_agent = next(iter(participants))
    first_execution = executions[first_agent].pop(0)
    await participants[first_agent].finish(
        first_execution,
        outcome="completed",
        result={"rollout_id": first_execution.rollout_id, "reward": 1.0},
    )
    await drain.add(
        first_agent,
        participants[first_agent].completion_receipt(
            first_execution.rollout_id,
            first_execution.attempt_index,
        ),
    )
    await asyncio.wait_for(drain.normal_wait_started.wait(), timeout=timeout_s)
    await drain.begin_checkpoint()

    deadline_ts = time.time() + timeout_s

    async def prepare_agent(participant: AgentCheckpointParticipant) -> dict[str, Any]:
        report = await participant.prepare(deadline_ts)
        while not report["ready_to_commit"]:
            if time.time() >= deadline_ts:
                return report
            await asyncio.sleep(min(0.05, progress_interval_s))
            report = participant.status()
        return report

    prepare_tasks = {
        agent_name: asyncio.create_task(prepare_agent(participant)) for agent_name, participant in participants.items()
    }
    await asyncio.sleep(0)

    async def finish_agent(agent_name: str, participant: AgentCheckpointParticipant) -> None:
        agent_executions = executions[agent_name]
        for offset in range(0, len(agent_executions), completion_batch_size):
            for execution in agent_executions[offset : offset + completion_batch_size]:
                await participant.finish(
                    execution,
                    outcome="completed",
                    result={"rollout_id": execution.rollout_id, "reward": 1.0},
                )
                await drain.add(
                    agent_name,
                    participant.completion_receipt(
                        execution.rollout_id,
                        execution.attempt_index,
                    ),
                )
            if completion_interval_ms:
                await asyncio.sleep(completion_interval_ms / 1_000)

    progress_samples: list[dict[str, Any]] = []
    monitor_stop = asyncio.Event()
    maximum_event_loop_lag_s = 0.0
    maximum_open_fds = _open_file_descriptor_count()

    async def monitor_progress() -> None:
        nonlocal maximum_event_loop_lag_s, maximum_open_fds
        expected = time.monotonic() + progress_interval_s
        while True:
            try:
                await asyncio.wait_for(monitor_stop.wait(), timeout=progress_interval_s)
                return
            except TimeoutError:
                pass
            observed = time.monotonic()
            maximum_event_loop_lag_s = max(maximum_event_loop_lag_s, observed - expected)
            expected = observed + progress_interval_s
            statuses = [participant.status() for participant in participants.values()]
            sample = {
                "elapsed_seconds": observed - (drain.checkpoint_started_at or observed),
                "running": sum(status["running"] for status in statuses),
                "parked_without_boundary": sum(status["parked_without_boundary"] for status in statuses),
                "completed_unacknowledged": sum(status["completed_unacknowledged"] for status in statuses),
                "ack_pending": drain._pending_count(),
            }
            progress_samples.append(sample)
            print("checkpoint_cpu_benchmark " + json.dumps(sample, sort_keys=True), flush=True)
            open_fds = _open_file_descriptor_count()
            if open_fds is not None:
                maximum_open_fds = max(maximum_open_fds or 0, open_fds)

    overlap_started = time.monotonic()
    monitor_task = asyncio.create_task(monitor_progress())
    try:
        await asyncio.gather(
            *(finish_agent(agent_name, participant) for agent_name, participant in participants.items())
        )
        await drain.finish_producing()
        await asyncio.wait_for(drain_task, timeout=timeout_s)
        prepare_reports = {
            agent_name: await asyncio.wait_for(task, timeout=timeout_s) for agent_name, task in prepare_tasks.items()
        }
    finally:
        monitor_stop.set()
        await monitor_task
        if not drain_task.done():
            drain_task.cancel()
            await asyncio.gather(drain_task, return_exceptions=True)
        for task in prepare_tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*prepare_tasks.values(), return_exceptions=True)
    overlap_seconds = time.monotonic() - overlap_started

    final_statuses = {agent_name: participant.status() for agent_name, participant in participants.items()}
    not_ready = {agent_name: report for agent_name, report in prepare_reports.items() if not report["ready_to_commit"]}
    pending = sum(status["completed_unacknowledged"] for status in final_statuses.values())
    if not_ready or pending or drain.acknowledged != runs or drain._pending_count():
        raise RuntimeError(
            "checkpoint overlap did not drain cleanly: "
            f"not_ready={sorted(not_ready)}, pending={pending}, "
            f"acknowledged={drain.acknowledged}/{runs}, ack_queue={drain._pending_count()}"
        )

    await asyncio.gather(*(participant.resume() for participant in participants.values()))
    checkpoint_flush_latency_s = None
    if drain.checkpoint_started_at is not None and drain.first_checkpoint_flush_at is not None:
        checkpoint_flush_latency_s = drain.first_checkpoint_flush_at - drain.checkpoint_started_at
    return {
        "runs": runs,
        "agents": agents,
        "agent_counts": agent_counts,
        "hot_agent_fraction": hot_agent_fraction,
        "completion_batch_size": completion_batch_size,
        "completion_interval_ms": completion_interval_ms,
        "ack_batch_size": ack_batch_size,
        "ack_concurrency": ack_concurrency,
        "normal_ack_coalesce_ms": normal_ack_coalesce_ms,
        "checkpoint_ack_coalesce_ms": checkpoint_ack_coalesce_ms,
        "ack_calls": drain.ack_calls,
        "acknowledged": drain.acknowledged,
        "average_ack_batch_size": sum(drain.ack_batch_sizes) / len(drain.ack_batch_sizes),
        "minimum_ack_batch_size": min(drain.ack_batch_sizes),
        "maximum_ack_batch_size": max(drain.ack_batch_sizes),
        "maximum_ack_pending": drain.maximum_pending,
        "checkpoint_flush_latency_seconds": checkpoint_flush_latency_s,
        "setup_seconds": setup_seconds,
        "overlap_seconds": overlap_seconds,
        "rollouts_per_second": runs / overlap_seconds,
        "ack_latency": _latency_summary(drain.ack_latencies),
        "prepare_progress_samples": progress_samples,
        "final_running": sum(status["running"] for status in final_statuses.values()),
        "final_parked_without_boundary": sum(status["parked_without_boundary"] for status in final_statuses.values()),
        "final_completed_unacknowledged": pending,
        "maximum_event_loop_lag_seconds": maximum_event_loop_lag_s,
        "maximum_open_file_descriptors": maximum_open_fds,
        "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


async def run_checkpoint_benchmark(
    run_root: Path,
    *,
    workers: int,
    cuts: int,
    hot_worker_fraction: float,
    prefix_tokens: int,
    staging_key_bytes: int,
    timeout_s: float,
    mixed_inventory: bool = False,
) -> dict[str, Any]:
    """Cut, journal, commit, and restore synthetic active generations."""
    if workers < 1 or cuts < 1:
        raise ValueError("workers and cuts must be positive")
    checkpoint_id = f"cpu-{int(time.time())}"
    control_root = run_root / "control"
    checkpoint_dir = run_root / "checkpoint"
    live_lineage = run_root / "live-lineage"
    restored_lineage = run_root / "restored-lineage"
    # AF_UNIX paths are limited to 108 bytes on Linux. Benchmark artifact
    # roots are often intentionally descriptive and can easily exceed that
    # limit, so keep the transient coordinator socket under a short local
    # path while leaving all measured artifacts under ``run_root``.
    coordinator_root = Path(tempfile.mkdtemp(prefix="gym-ckpt-coordinator-", dir="/tmp"))
    worker_artifact_root = control_root / "worker-checkpoint-artifacts"
    worker_artifact_root.mkdir(parents=True, exist_ok=True)
    (coordinator_root / "worker-checkpoint-artifacts").symlink_to(
        worker_artifact_root,
        target_is_directory=True,
    )
    lineage = FileLineageStore(live_lineage)
    coordinator = AdmissionCoordinator(coordinator_root / "coordinator.sock", expected_workers=workers)
    service = PolicyModelCheckpointCoordinatorService(
        coordinator,
        ledger_provider=lambda: lineage,
        file_ledger_root_provider=lambda: lineage.checkpoint_root,
        instance_role="policy",
        server_name=_SERVER_NAME,
        supports_generation_cuts=True,
    )
    coordinator.service_handler = service
    await coordinator.start()

    worker_counts = _worker_counts(cuts, workers, hot_worker_fraction)
    agents: list[WorkerAdmissionAgent] = []
    limiters: list[AdmissionLimiter] = []
    tickets: list[tuple[AdmissionLimiter, Any]] = []
    delayed_pre_generation: list[tuple[AdmissionLimiter, Any]] = []
    delayed_response_egress: list[tuple[AdmissionLimiter, Any]] = []
    inventory_counts: dict[str, int] = defaultdict(int)
    expected_prefixes: dict[str, tuple[str, ...]] = {}
    try:
        rollout_index = 0
        for worker_index, worker_cut_count in enumerate(worker_counts):
            worker_id = f"worker-{worker_index:02d}"
            limiter = AdmissionLimiter(
                _SyntheticCutBackend(
                    worker_id,
                    prefix_tokens=prefix_tokens,
                    staging_key_bytes=staging_key_bytes,
                )
            )
            agent = WorkerAdmissionAgent(
                coordinator.socket_path,
                worker_id,
                limiter,
                pid=10_000 + worker_index,
                server_name=_SERVER_NAME,
                capture_ledger=lineage,
            )
            await agent.start()
            agents.append(agent)
            limiters.append(limiter)
            for _ in range(worker_cut_count):
                ticket = limiter.admit(
                    rollout_id=f"rollout-{rollout_index:09d}",
                    attempt_index=0,
                )
                state = (
                    (
                        "active_prefix",
                        "pre_generation",
                        "no_generation",
                        "durable_completed",
                        "durable_failure",
                        "response_egress",
                    )[rollout_index % 6]
                    if mixed_inventory
                    else "active_prefix"
                )
                inventory_counts[state] += 1
                if state != "pre_generation" and state != "no_generation":
                    ticket.bind_model_call(f"call-{rollout_index:09d}")
                    ticket.mark_generation_started()
                if state == "active_prefix":
                    padding = "x" * staging_key_bytes
                    expected_prefixes[ticket.model_call_id] = (f"prefix/{ticket.ticket_id}/{padding}",)
                elif state == "pre_generation":
                    delayed_pre_generation.append((limiter, ticket))
                elif state == "no_generation":
                    ticket.mark_no_generation()
                elif state == "durable_completed":
                    ticket.mark_durable_completed()
                elif state == "durable_failure":
                    ticket.mark_durable_failure()
                elif state == "response_egress":
                    ticket.response_started = True
                    delayed_response_egress.append((limiter, ticket))
                tickets.append((limiter, ticket))
                rollout_index += 1

        await coordinator.wait_until(
            lambda status: status["workers"]["live"] == workers,
            timeout_s=timeout_s,
        )
        deadline_ts = time.time() + timeout_s
        client = agents[0].service_client()

        prepare_started = time.monotonic()

        async def resolve_frozen_tickets(limiter: AdmissionLimiter) -> None:
            while limiter.state.value == "accepting":
                if time.time() >= deadline_ts:
                    raise TimeoutError("mixed checkpoint inventory did not freeze before its deadline")
                await asyncio.sleep(0)
            for owner, ticket in delayed_pre_generation:
                if owner is limiter:
                    ticket.mark_no_generation()
            for owner, ticket in delayed_response_egress:
                if owner is limiter:
                    owner.mark_response_egress_completed(ticket)
                    owner.release(ticket)

        resolver_tasks = (
            [asyncio.create_task(resolve_frozen_tickets(limiter)) for limiter in limiters] if mixed_inventory else []
        )
        prepare_task = asyncio.create_task(
            client.request(
                "model_admission_pause",
                {"checkpoint_id": checkpoint_id, "deadline_ts": deadline_ts},
                timeout_s=timeout_s,
            )
        )
        if resolver_tasks:
            await asyncio.gather(*resolver_tasks)
        prepare = await prepare_task
        prepare_seconds = time.monotonic() - prepare_started

        commit_started = time.monotonic()
        commit = await client.request(
            "model_checkpoint_commit",
            {
                "checkpoint_id": checkpoint_id,
                "deadline_ts": deadline_ts,
                "checkpoint_dir": str(checkpoint_dir),
                "continuation_indexes": [],
            },
            timeout_s=timeout_s,
        )
        commit_seconds = time.monotonic() - commit_started

        restore_started = time.monotonic()
        restore = await asyncio.to_thread(
            CaptureLedgerCheckpointer(restored_lineage, server_name=_SERVER_NAME).restore,
            checkpoint_dir,
        )
        restore_seconds = time.monotonic() - restore_started

        journal_files = sorted((control_root / "worker-checkpoint-artifacts").rglob("*.jsonl"))
        archive_files = sorted((checkpoint_dir / "model-ledger" / _SERVER_NAME).glob("lineage-part-*.tar"))
        restored_files = list(restored_lineage.glob("*.lineage.jsonl"))
        artifact_reference_sizes = [
            len(json.dumps(record.cut_artifact.model_dump(mode="json"), separators=(",", ":")).encode())
            for record in coordinator._workers.values()
            if record.cut_artifact is not None
        ]
        expected_generation_cuts = inventory_counts["active_prefix"]
        restored_prefixes = [
            prefix for receipt in restore["generation_cut_receipts"] for prefix in receipt["prefixes"]
        ]
        restored_prefix_coordinates = {
            prefix["model_call_id"]: tuple(prefix["staging_keys"]) for prefix in restored_prefixes
        }
        if (
            int(commit["generation_cut_records"]) != expected_generation_cuts
            or int(restore["rollouts"]) != expected_generation_cuts
            or restored_prefix_coordinates != expected_prefixes
        ):
            raise RuntimeError(
                "checkpoint benchmark mismatch: "
                f"expected_generation_cuts={expected_generation_cuts}, "
                f"commit={commit!r}, restore={restore!r}"
            )
        return {
            "workers": workers,
            "cuts": cuts,
            "worker_cut_counts": worker_counts,
            "hot_worker_fraction": hot_worker_fraction,
            "mixed_inventory": mixed_inventory,
            "inventory_counts": dict(sorted(inventory_counts.items())),
            "prepare_seconds": prepare_seconds,
            "commit_seconds": commit_seconds,
            "restore_seconds": restore_seconds,
            "prepare_records": prepare["generation_cut_summary"]["records"],
            "journal_files": len(journal_files),
            "journal_bytes": sum(path.stat().st_size for path in journal_files),
            "artifact_reference_wire_bytes_total": sum(artifact_reference_sizes),
            "artifact_reference_wire_bytes_max": max(artifact_reference_sizes, default=0),
            "archive_files": len(archive_files),
            "archive_bytes": sum(path.stat().st_size for path in archive_files),
            "restored_lineage_files": len(restored_files),
            "generation_cuts_restored": len(restored_prefixes),
            "restored_prefix_coordinates_match": restored_prefix_coordinates == expected_prefixes,
        }
    finally:
        for limiter, ticket in tickets:
            limiter.release(ticket)
        for agent in agents:
            await agent.stop()
        await coordinator.stop()
        await lineage.close()
        shutil.rmtree(coordinator_root, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(_PROFILES), default="smoke")
    parser.add_argument(
        "--components",
        choices=("all", "control", "overlap", "checkpoint"),
        default="all",
        help="Benchmark /run traffic, prepare/ACK overlap, checkpoint I/O, or all three.",
    )
    parser.add_argument("--root", type=Path, help="Parent directory for benchmark artifacts.")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--cuts", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--agents", type=int)
    parser.add_argument("--run-concurrency", type=int)
    parser.add_argument("--data-connections", type=int)
    parser.add_argument("--control-connections", type=int)
    parser.add_argument("--ack-batch-size", type=int, default=256)
    parser.add_argument("--ack-concurrency", type=int, default=8)
    parser.add_argument(
        "--ack-coalesce-ms",
        type=float,
        default=1_000.0,
        help="Wait after the first receipt so a bulk-ACK batch can fill; use 0 to drain immediately.",
    )
    parser.add_argument(
        "--checkpoint-ack-coalesce-ms",
        type=float,
        default=50.0,
        help="Maximum ACK coalescing delay while checkpoint prepare is active.",
    )
    parser.add_argument("--completion-batch-size", type=int)
    parser.add_argument("--completion-interval-ms", type=float, default=5.0)
    parser.add_argument("--progress-interval-s", type=float, default=1.0)
    parser.add_argument("--run-delay-ms", type=float, default=0.0)
    parser.add_argument("--transport", choices=("tcp", "asgi"), default="tcp")
    parser.add_argument("--hot-worker-fraction", type=float, default=0.45)
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--staging-key-bytes", type=int, default=64)
    parser.add_argument(
        "--checkpoint-inventory",
        choices=("mixed", "active-prefix"),
        default="mixed",
        help="Use realistic mixed ticket states or make every ticket an active prefix.",
    )
    parser.add_argument("--timeout-s", type=float, default=1_800.0)
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> dict[str, Any]:
    profile = _PROFILES[args.profile]
    parent = args.root
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="gym-checkpoint-cpu-", dir=parent))
    result: dict[str, Any] = {
        "profile": args.profile,
        "components": args.components,
        "run_root": str(run_root),
        "filesystem": "explicit" if parent is not None else "temporary",
        "profile_defaults": asdict(profile),
    }
    succeeded = False
    try:
        if args.components in {"all", "control"}:
            result["control"] = await run_control_benchmark(
                runs=args.runs if args.runs is not None else profile.runs,
                concurrency=args.run_concurrency or profile.run_concurrency,
                ack_batch_size=args.ack_batch_size,
                ack_concurrency=args.ack_concurrency,
                ack_coalesce_ms=args.ack_coalesce_ms,
                delay_ms=args.run_delay_ms,
                data_connections=args.data_connections or profile.data_connections,
                control_connections=args.control_connections or profile.control_connections,
                transport=args.transport,
            )
        if args.components in {"all", "overlap"}:
            result["overlap"] = await run_checkpoint_overlap_benchmark(
                runs=args.runs if args.runs is not None else profile.runs,
                agents=args.agents or profile.agents,
                hot_agent_fraction=args.hot_worker_fraction,
                completion_batch_size=args.completion_batch_size or profile.completion_batch_size,
                completion_interval_ms=args.completion_interval_ms,
                ack_batch_size=args.ack_batch_size,
                ack_concurrency=args.ack_concurrency,
                normal_ack_coalesce_ms=args.ack_coalesce_ms,
                checkpoint_ack_coalesce_ms=args.checkpoint_ack_coalesce_ms,
                progress_interval_s=args.progress_interval_s,
                timeout_s=args.timeout_s,
            )
        if args.components in {"all", "checkpoint"}:
            result["checkpoint"] = await run_checkpoint_benchmark(
                run_root,
                workers=args.workers,
                cuts=args.cuts if args.cuts is not None else profile.cuts,
                hot_worker_fraction=args.hot_worker_fraction,
                prefix_tokens=args.prefix_tokens,
                staging_key_bytes=args.staging_key_bytes,
                timeout_s=args.timeout_s,
                mixed_inventory=args.checkpoint_inventory == "mixed",
            )
        succeeded = True
        return result
    finally:
        result["artifacts_kept"] = args.keep_artifacts or not succeeded
        if succeeded and not args.keep_artifacts:
            shutil.rmtree(run_root)


if __name__ == "__main__":
    benchmark_result = asyncio.run(_main(_parse_args()))
    print(json.dumps(benchmark_result, indent=2, sort_keys=True))
