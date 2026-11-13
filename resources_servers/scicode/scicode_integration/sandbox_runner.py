# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the existing SciCode grader in a fresh sandbox per environment attempt."""

import asyncio
import json
import shlex
from pathlib import Path

from pydantic import BaseModel, Field

from nemo_gym.config import get_global_config_dict
from nemo_gym.core import PROVIDER_OOM_KILLED, NgFailureEvent, observe_failure
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, resolve_provider_config, resolve_provider_metadata


class ScicodeSandboxConfig(BaseModel):
    """Image containing the configured grading interpreters and test-data release."""

    provider: str = "sandbox"
    image: str
    cpu: float = Field(default=1, gt=0)
    memory_mib: int = Field(default=8192, gt=0)
    ready_timeout_secs: float = Field(default=300, gt=0)


_PREFLIGHT = """import hashlib, json, pathlib
from runner import preflight_grading_interpreter
request = json.loads(pathlib.Path("/tmp/scicode-request.json").read_text())
error = preflight_grading_interpreter(request["python_executable"], timeout_secs=60)
if error:
    raise RuntimeError("SciCode sandbox grading preflight failed: " + error)
data = pathlib.Path(request["test_data_path"])
if not data.is_file():
    raise RuntimeError("SciCode sandbox test data missing: " + str(data))
if request["test_data_md5"]:
    with data.open("rb") as stream:
        digest = hashlib.file_digest(stream, "md5").hexdigest()
    if digest != request["test_data_md5"].lower():
        raise RuntimeError("SciCode sandbox test data checksum mismatch")
"""

_DRIVER = """import json, pathlib
from runner import run_substep
request = json.loads(pathlib.Path("/tmp/scicode-request.json").read_text())
result = run_substep(pathlib.Path("/tmp/scicode-program.py").read_text(),
                     request["timeout_secs"], request["python_executable"],
                     output_tail_chars=request["output_tail_chars"])
print(json.dumps(result))
"""


async def run_sandbox_substep(
    program: str,
    *,
    config: ScicodeSandboxConfig,
    python_executable: str,
    timeout_secs: float,
    test_data_path: str,
    test_data_md5: str | None,
    output_tail_chars: int,
) -> dict:
    """Score confirmed execution OOM as a failed step; propagate inconclusive sandbox failures."""
    global_config = get_global_config_dict()
    provider = resolve_provider_config(config.provider, global_config)
    request = dict(
        python_executable=python_executable,
        timeout_secs=timeout_secs,
        test_data_path=test_data_path,
        test_data_md5=test_data_md5,
        output_tail_chars=output_tail_chars,
    )
    spec = SandboxSpec(
        image=config.image,
        resources={"cpu": config.cpu, "memory_mib": config.memory_mib},
        ready_timeout_s=config.ready_timeout_secs,
        # Includes preflight and cleanup headroom; a lost client cannot leave an indefinite sandbox.
        ttl_s=timeout_secs + config.ready_timeout_secs + 300,
        metadata=resolve_provider_metadata(config.provider, global_config),
        files={
            "/tmp/runner.py": Path(__file__).with_name("runner.py").read_text(),
            "/tmp/scicode-driver.py": _DRIVER,
            "/tmp/scicode-preflight.py": _PREFLIGHT,
            "/tmp/scicode-program.py": program,
            "/tmp/scicode-request.json": json.dumps(request),
        },
    )
    sandbox = AsyncSandbox(provider=provider, spec=spec)
    try:
        await sandbox.start()
        preflight = await sandbox.exec(
            f"{shlex.quote(python_executable)} /tmp/scicode-preflight.py",
            timeout_s=120,
        )
        if preflight.error_type or preflight.return_code != 0:
            raise RuntimeError(f"SciCode sandbox preflight failed: {(preflight.stderr or '')[-2000:]}")
        try:
            execution = await sandbox.exec(
                f"{shlex.quote(python_executable)} /tmp/scicode-driver.py",
                timeout_s=timeout_secs + 120,
            )
        except NgFailureEvent as exc:
            if exc.failure_kind != PROVIDER_OOM_KILLED:
                raise
            observe_failure(exc)
            return {
                "passed": False,
                "error": "Generated code exceeded the sandbox memory limit",
                "returncode": None,
                "exit_unknown": True,
                "exception_class": "MemoryLimitExceeded",
            }
        if execution.error_type or execution.return_code != 0:
            raise RuntimeError(
                f"SciCode sandbox infrastructure failure ({sandbox.sandbox_id}, "
                f"exit={execution.return_code}, error_type={execution.error_type}): "
                f"{(execution.stderr or '')[-2000:]}"
            )
        result = json.loads(execution.stdout or "")
        if not isinstance(result, dict) or not isinstance(result.get("passed"), bool) or "error" not in result:
            raise RuntimeError("SciCode sandbox returned an invalid grading result")
        if result.get("returncode") == -9:
            raise RuntimeError(
                f"SciCode grading process was killed by SIGKILL in sandbox {sandbox.sandbox_id}; "
                "check the sandbox memory limit before retrying"
            )
        return result
    finally:
        # Keep the attempt's permit until deletion finishes, even if cancellation is repeated.
        cleanup = asyncio.create_task(sandbox.stop())
        cancellation = None
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as exc:
                cancellation = exc
        cleanup.result()
        if cancellation is not None:
            raise cancellation
