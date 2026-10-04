# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned, fail-closed single-node NIXL prefill/decode groups (TP × DP per role)."""

import asyncio
import json
import os
import socket
from contextlib import ExitStack
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from responses_api_models.local_vllm_model.router_launcher import VLLMRouterConfig, VLLMRouterLauncher
from responses_api_models.local_vllm_model.subprocess_launcher import (
    PDWorkerPlacement,
    VLLMSubprocessConfig,
    VLLMSubprocessLauncher,
    normalize_kwargs,
    topology_manifest,
    validate_managed_kwargs,
)


class PDRoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gpu_indices: list[StrictInt] | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    rpc_port: int | None = Field(default=None, ge=1, le=65535)
    side_channel_port: int | None = Field(default=None, ge=1, le=65535)
    # Share weights, tokenizer, TP/DP, KV layout and speculative decoding across roles.
    # Only scheduler/performance knobs may differ; no arbitrary CLI escape hatch.
    serve_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("serve_kwargs")
    @classmethod
    def scheduler_only(cls, value: dict[str, Any]) -> dict[str, Any]:
        value = normalize_kwargs(value)
        allowed = {
            "max_num_batched_tokens",
            "max_num_seqs",
            "gpu_memory_utilization",
            "all2all_backend",
            "compilation_config",
        }
        if unsupported := value.keys() - allowed:
            raise ValueError(f"PD role overrides must be scheduler-only; unsupported: {sorted(unsupported)}")
        return value


class VLLMPDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prefill: PDRoleConfig = Field(default_factory=PDRoleConfig)
    decode: PDRoleConfig = Field(default_factory=PDRoleConfig)


def reserve_ports(stack: ExitStack, count: int = 1, first: int | None = None) -> int:
    """Hold a contiguous port range while planning all roles; release before exec."""
    for attempt in range(100):
        candidate = ExitStack()
        try:
            initial = candidate.enter_context(socket.socket())
            initial.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            initial.bind(("127.0.0.1", first or 0))
            initial.listen()
            port = initial.getsockname()[1]
            if port + count - 1 > 65535:
                raise ValueError("PD side-channel port range exceeds 65535")
            for offset in range(1, count):
                other = candidate.enter_context(socket.socket())
                other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                other.bind(("127.0.0.1", port + offset))
                other.listen()
        except OSError:
            candidate.close()
            if first is not None or attempt == 99:
                raise
        except BaseException:
            candidate.close()
            raise
        else:
            stack.enter_context(candidate)
            return port


class VLLMPDLauncher:
    def __init__(
        self,
        *,
        pd: VLLMPDConfig,
        config: VLLMSubprocessConfig,
        router: VLLMRouterConfig,
        model: str,
        kwargs: dict[str, Any],
        env: dict[str, str],
        api_key: str,
        cache_dir: str,
        show_stats: bool = False,
    ):
        self.pd, self.config, self.model = pd, config, model
        self.kwargs = normalize_kwargs(kwargs)
        validate_managed_kwargs(self.kwargs, env)
        self.env, self.api_key, self.cache_dir, self.show_stats = env, api_key, cache_dir, show_stats
        # Require the scheduler's inherited visibility, not an invented config allocation.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            raise ValueError("PD requires scheduler-provided CUDA_VISIBLE_DEVICES")
        if env.get("CUDA_VISIBLE_DEVICES", visible) != visible:
            raise ValueError("PD cannot override inherited CUDA_VISIBLE_DEVICES")
        self.topology = topology_manifest(self.kwargs, os.environ | env)
        count = self.topology["required_gpus"]
        self.indices = {
            "prefill": pd.prefill.gpu_indices if pd.prefill.gpu_indices is not None else list(range(count)),
            "decode": pd.decode.gpu_indices if pd.decode.gpu_indices is not None else list(range(count, 2 * count)),
        }
        selected = self.indices["prefill"] + self.indices["decode"]
        if (
            any(len(indices) != count for indices in self.indices.values())
            or len(set(selected)) != 2 * count
            or any(index < 0 or index >= len(visible.split(",")) for index in selected)
        ):
            raise ValueError("P/D must each own TP × DP distinct, non-overlapping inherited GPU indices")
        self.router = VLLMRouterLauncher(config=router, model=model, api_key=api_key)
        self.workers: dict[str, VLLMSubprocessLauncher] = {}
        self.run_dir: Path | None = None

    async def start(self, port: int, *, router_port: int | None = None, dry_run: bool = False) -> str:
        if self.run_dir is not None:
            raise RuntimeError("VLLMPDLauncher is single-use")
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = Path(mkdtemp(prefix="pd-", dir=self.config.log_dir)).resolve()
        ports = {}
        try:
            with ExitStack() as reservations:
                for role in ("prefill", "decode"):
                    role_config = getattr(self.pd, role)
                    if role == "prefill" and role_config.port and role_config.port != port:
                        raise ValueError("pd.prefill.port must agree with subprocess.port / the requested port")
                    ports[role] = reserve_ports(reservations, first=port if role == "prefill" else role_config.port)
                    placement = PDWorkerPlacement(
                        role=role,
                        gpu_indices=self.indices[role],
                        rpc_port=reserve_ports(reservations, first=role_config.rpc_port),
                        side_channel_port=reserve_ports(
                            reservations, self.topology["data_parallel_size"], role_config.side_channel_port
                        ),
                    )
                    self.workers[role] = VLLMSubprocessLauncher(
                        config=self.config.model_copy(update={"log_dir": self.run_dir / role}),
                        model=self.model,
                        kwargs=self.kwargs | role_config.serve_kwargs,
                        env=self.env,
                        api_key=self.api_key,
                        cache_dir=self.cache_dir,
                        show_stats=self.show_stats,
                        pd_placement=placement,
                    )
                router_port = reserve_ports(reservations, first=router_port or self.router.config.port)
            # A failing startup cancels its sibling and both owned process groups are reaped.
            async with asyncio.TaskGroup() as tasks:
                for role, worker in self.workers.items():
                    tasks.create_task(worker.start(ports[role], dry_run=dry_run))
            versions = {
                json.loads((worker.run_dir / "manifest.json").read_text())["vllm_version"]
                for worker in self.workers.values()
            }
            if len(versions) != 1:
                raise ValueError("P/D executable version mismatch")
            base_url = await self.router.start(
                router_port,
                worker_urls=[],
                prefill_urls=[self.workers["prefill"].base_url.removesuffix("/v1")],
                decode_urls=[self.workers["decode"].base_url.removesuffix("/v1")],
                dp_size=self.topology["data_parallel_size"],
                dry_run=dry_run,
            )
            manifest = {
                "scope": "single-node NIXL PD, symmetric TP/DP, MP, PP=1",
                "dry_run": dry_run,
                "base_url": base_url,
                "vllm_version": versions.pop(),
                "combined_fallback": False,
                "gpu_indices": self.indices,
                "workers": {role: str(worker.run_dir / "manifest.json") for role, worker in self.workers.items()},
                "router": str(self.router.run_dir / "manifest.json"),
            }
            (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            return base_url
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        try:
            await self.router.stop()
        finally:
            # Attempt every cleanup even when one supervisor reports a failure.
            results = await asyncio.gather(
                *(worker.stop() for worker in self.workers.values()), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
