# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import shlex
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.providers import SandboxSpec

from .ccc_eval import CCCEvaluator, _load_metadata_file


class SandboxSubtaskResult(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    score: float = Field(allow_inf_nan=False)
    outputs: list[dict[str, Any]]


class SandboxCCCResult(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    test_case_results: dict[str, SandboxSubtaskResult] = Field(min_length=1)


class SandboxCCCEvaluator(CCCEvaluator):
    """Execute the native CCC evaluator in a fresh provider sandbox per candidate."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        num_parallel_requests: int,
        provider: str | Mapping[str, Any],
        spec: SandboxSpec | None,
        python: str,
        setup_command: str | None,
        timeout: int,
        named_configs: Mapping[str, Any],
    ) -> None:
        if spec is None:
            raise ValueError("CCC sandbox execution requires sandbox_spec")
        super().__init__(config, num_parallel_requests)
        self.provider = resolve_provider_config(provider, named_configs)
        self.spec = replace(spec, metadata={**resolve_provider_metadata(provider, named_configs), **spec.metadata})
        self.python = python
        self.setup_command = setup_command
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(num_parallel_requests)

    async def _initialize_runtime(self) -> None:
        async with self._init_lock:
            if self.metadata_by_competition is None:
                self.metadata_by_competition, self.problem_index = await asyncio.to_thread(
                    _load_metadata_file, self.eval_cfg.test_file
                )

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        await self._initialize_runtime()
        problem_id = data_point["problem_id"]
        competition_id = self._get_competition_id(data_point)
        metadata = self.get_problem_metadata(problem_id, competition_id)
        request = {
            "entry": data_point,
            "metadata": {"competition_id": competition_id, "metadata": {problem_id: metadata}},
            "config": {
                "test_batch_size": self.eval_cfg.test_batch_size,
                "time_scale": self.eval_cfg.time_scale,
                "run_all_tests": self.eval_cfg.run_all_tests,
            },
        }
        root = Path(__file__).parent
        files = {
            **self.spec.files,
            "/tmp/resources_servers/__init__.py": "",
            "/tmp/resources_servers/competitive_coding_challenges/__init__.py": "",
            **{
                f"/tmp/resources_servers/competitive_coding_challenges/{name}": (root / name).read_text()
                for name in ("ccc_eval.py", "sandbox_worker.py")
            },
            "/tmp/ccc-request.json": json.dumps(request),
        }
        setup_timeout = 300 if self.setup_command else 0
        lifetime = self.timeout + setup_timeout + (self.spec.ready_timeout_s or 120)
        if self.spec.ttl_s is not None and self.spec.ttl_s <= lifetime:
            raise ValueError("CCC sandbox TTL must cover readiness, setup and execution")
        spec = replace(
            self.spec,
            files={},
            ttl_s=self.spec.ttl_s or lifetime + 60,
            metadata={**self.spec.metadata, "instance_id": f"ccc-{uuid4().hex}"},
        )
        async with self.semaphore, AsyncSandbox(self.provider, spec) as sandbox:
            await sandbox.start()
            if self.setup_command:
                setup = await sandbox.exec(self.setup_command, timeout_s=setup_timeout)
                if setup.error_type or setup.return_code != 0:
                    raise RuntimeError(f"CCC sandbox setup failed: {setup.stderr}")
            with TemporaryDirectory(prefix="ccc-transfer-") as directory:
                path = Path(directory) / "artifact"
                for target, content in files.items():
                    path.write_text(content)
                    await sandbox.upload(path, target)
                result = await sandbox.exec(
                    f"{shlex.quote(self.python)} -m resources_servers.competitive_coding_challenges.sandbox_worker "
                    "/tmp/ccc-request.json /tmp/ccc-result.json",
                    cwd="/tmp",
                    timeout_s=self.timeout,
                )
                if result.error_type or result.return_code != 0:
                    raise RuntimeError(
                        f"CCC sandbox worker failed (exit={result.return_code}, "
                        f"error={result.error_type}): {(result.stderr or '')[-2000:]}"
                    )
                await sandbox.download("/tmp/ccc-result.json", path)
                output = json.loads(path.read_text())
                SandboxCCCResult.model_validate(output)
                return output
