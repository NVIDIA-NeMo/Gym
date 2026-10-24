# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in CUDA allocator diagnostics for the first serving steps of a vLLM worker.

Use TRACE_ALLOCATIONS=1 with nemotron_3.5_super_dp4tp1_deepep.sh. This records
allocation stacks and can perturb timing; use a separate run from throughput
measurements. Snapshots can be opened in https://pytorch.org/memory_viz/.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker


if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

logger = init_logger(__name__)


class AllocationTraceWorker(Worker):
    """Attribute allocator retries to a host, worker, serving phase, and stack."""

    def init_device(self) -> None:
        super().init_device()
        self._allocation_trace_steps = 0
        self._allocation_trace_limit = int(os.environ.get("GYM_ALLOCATION_TRACE_STEPS", "120"))
        if self._allocation_trace_limit < 1:
            raise ValueError("GYM_ALLOCATION_TRACE_STEPS must be positive")
        self._allocation_trace_active = False
        self._allocation_trace_dumps = 0
        self._allocation_trace_dir = Path(os.environ["GYM_ALLOCATION_TRACE_DIR"])
        self._allocation_trace_dir.mkdir(parents=True, exist_ok=True)
        transfer = self.vllm_config.kv_transfer_config
        role = transfer.kv_role if transfer is not None else "none"
        self._allocation_trace_identity = (
            f"{socket.gethostname()}-pid{os.getpid()}-{role}"
            f"-dp{self.parallel_config.data_parallel_index}-rank{self.rank}"
        )

    def _allocation_snapshot(self, label: str) -> None:
        path = self._allocation_trace_dir / f"{self._allocation_trace_identity}-{label}.pickle"
        try:
            torch.cuda.memory._dump_snapshot(str(path))
        except Exception:
            # Diagnostics must not replace a serving error or stop a healthy worker.
            logger.exception("ALLOC_TRACE failed to write %s", path)
        else:
            logger.info("ALLOC_TRACE snapshot=%s", path)

    def _start_allocation_trace(self) -> None:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="all", max_entries=100_000, device=self.device
        )
        self._allocation_trace_active = True
        config = self.model_config.hf_text_config
        dimensions = {
            name: getattr(config, name, None)
            for name in (
                "hidden_size",
                "moe_latent_size",
                "moe_intermediate_size",
                "n_routed_experts",
                "num_experts_per_tok",
            )
        }
        logger.info(
            "ALLOC_TRACE start worker=%s steps=%d dimensions=%s",
            self._allocation_trace_identity,
            self._allocation_trace_limit,
            dimensions,
        )

    def _stop_allocation_trace(self) -> None:
        self._allocation_snapshot("final")
        torch.cuda.memory._record_memory_history(enabled=None)
        self._allocation_trace_active = False
        logger.info("ALLOC_TRACE stopped worker=%s", self._allocation_trace_identity)

    @contextmanager
    def _allocation_phase(self, phase: str, scheduled_tokens: int | None = None) -> Iterator[None]:
        before = torch.cuda.memory_stats(self.device)
        try:
            yield
        finally:
            after = torch.cuda.memory_stats(self.device)
            retries = after.get("num_alloc_retries", 0) - before.get("num_alloc_retries", 0)
            ooms = after.get("num_ooms", 0) - before.get("num_ooms", 0)
            if retries or ooms or self._allocation_trace_steps == 1:
                free, total = torch.cuda.mem_get_info(self.device)
                logger.info(
                    "ALLOC_TRACE worker=%s step=%d phase=%s scheduled_tokens=%s "
                    "retries=%d ooms=%d allocated_bytes=%d reserved_bytes=%d "
                    "inactive_split_bytes=%d device_free_bytes=%d device_total_bytes=%d",
                    self._allocation_trace_identity,
                    self._allocation_trace_steps,
                    phase,
                    scheduled_tokens,
                    retries,
                    ooms,
                    after.get("allocated_bytes.all.current", 0),
                    after.get("reserved_bytes.all.current", 0),
                    after.get("inactive_split_bytes.all.current", 0),
                    free,
                    total,
                )
            if (retries or ooms) and self._allocation_trace_dumps < 2:
                self._allocation_trace_dumps += 1
                self._allocation_snapshot(f"step{self._allocation_trace_steps}-{phase}")

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        if self._allocation_trace_steps == 0:
            self._start_allocation_trace()
        if self._allocation_trace_steps >= self._allocation_trace_limit:
            if self._allocation_trace_active:
                self._stop_allocation_trace()
            return super().execute_model(scheduler_output)
        self._allocation_trace_steps += 1
        with self._allocation_phase("execute", scheduler_output.total_num_scheduled_tokens):
            return super().execute_model(scheduler_output)

    def sample_tokens(self, grammar_output: GrammarOutput | None) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        if not self._allocation_trace_active:
            return super().sample_tokens(grammar_output)
        with self._allocation_phase("sample_and_draft"):
            return super().sample_tokens(grammar_output)
