# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shlex
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, StrictBool, StrictInt

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.providers import SandboxSpec


class SandboxHarnessResult(BaseModel):
    """Validate the native checker artifact before it affects a reward."""

    result: list[StrictBool | StrictInt] = Field(min_length=1)
    metadata: dict[str, Any] | None


async def check_in_sandbox(
    *,
    sample: dict[str, str],
    generation: str,
    timeout: int,
    debug: bool,
    provider: str | Mapping[str, Any],
    spec: SandboxSpec | None,
    python: str,
    named_configs: Mapping[str, Any],
    setup_command: str | None = None,
) -> tuple[list[bool | int], dict[str, Any] | None]:
    """Run the native checker in a fresh sandbox, propagating infrastructure failures."""
    if spec is None:
        raise ValueError("code_gen sandbox execution requires sandbox_spec")
    # Each sampled completion must own a different sandbox, even for the same task.
    request = json.dumps(dict(sample=sample, generation=generation, timeout=timeout, debug=debug))
    test_count = len(json.loads(sample["input_output"])["inputs"])
    if test_count == 0:
        raise ValueError("code_gen sandbox execution requires at least one unit test")
    harness_timeout = (timeout + 1) * test_count + 5
    execution_timeout = harness_timeout + 30  # Worker startup, child reaping, and artifact serialization.
    setup_timeout = 120 if setup_command else 0
    lifetime = execution_timeout + setup_timeout + (spec.ready_timeout_s or 120)
    if spec.ttl_s is not None and spec.ttl_s <= lifetime:
        raise ValueError("code_gen sandbox TTL must cover readiness and the complete harness deadline")
    root = Path(__file__).parent
    files = {
        **spec.files,
        "/tmp/resources_servers/__init__.py": "",
        "/tmp/resources_servers/code_gen/__init__.py": "",
        "/tmp/resources_servers/code_gen/lcb_integration/__init__.py": "",
        **{
            f"/tmp/resources_servers/code_gen/{name}": (root / name).read_text()
            for name in ("sandbox_worker.py", "lcb_integration/checker.py", "lcb_integration/testing_util.py")
        },
        "/tmp/code-gen-request.json": request,
    }
    spec = replace(
        spec,
        ttl_s=spec.ttl_s or lifetime + 60,
        files={},
        metadata={
            **resolve_provider_metadata(provider, named_configs),
            **spec.metadata,
            "instance_id": f"code-gen-{uuid4().hex}",
        },
    )
    async with AsyncSandbox(resolve_provider_config(provider, named_configs), spec) as sandbox:
        await sandbox.start()
        if setup_command:
            setup = await sandbox.exec(setup_command, timeout_s=setup_timeout)
            if setup.error_type or setup.return_code != 0:
                raise RuntimeError(f"code_gen sandbox setup failed: {setup.stderr}")
        # Upload after start so Gym has the handle even if an upload is cancelled.
        with TemporaryDirectory(prefix="code-gen-input-") as directory:
            path = Path(directory) / "input"
            for target, contents in files.items():
                path.write_text(contents)
                await sandbox.upload(path, target)
        execution = await sandbox.exec(
            f"{shlex.quote(python)} -m resources_servers.code_gen.sandbox_worker "
            "/tmp/code-gen-request.json /tmp/code-gen-result.json",
            cwd="/tmp",
            timeout_s=execution_timeout,
        )
        # Candidate failures/timeouts are native harness results. Failure of the outer
        # worker/provider is infrastructure failure and must never become reward zero.
        if execution.error_type or execution.return_code != 0:
            raise RuntimeError(
                f"code_gen sandbox worker failed (exit={execution.return_code}, "
                f"error={execution.error_type}): {(execution.stderr or '')[-2000:]}"
            )
        with TemporaryDirectory(prefix="code-gen-result-") as directory:
            path = Path(directory) / "result.json"
            await sandbox.download("/tmp/code-gen-result.json", path)
            result = SandboxHarnessResult.model_validate_json(path.read_bytes())
        return result.result, result.metadata
