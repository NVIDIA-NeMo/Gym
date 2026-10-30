# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a verifier's grading harness in a fresh, single-use provider sandbox.

A resources server uploads its runner and harness files with a JSON request; inside the
sandbox ``sandbox_runner.py request.json result.json`` grades the submission and writes a
result file, which is downloaded and parsed before the sandbox is stopped. Only the
provider-neutral sandbox API is used, so any provider works.

Submission failures (wrong answer, timeout, crash) are the harness's to report in the
result. A failure of the sandbox or the runner raises ``SandboxVerificationError`` with a
``failure_kind`` so the caller can mask the sample instead of scoring it zero.
"""

import json
import logging
import shlex
import uuid
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, Field, field_validator

from nemo_gym.core.failure_kinds import PROVIDER_UNAVAILABLE, VERIFIER_ERROR
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxProvider, SandboxResources, SandboxSpec
from nemo_gym.sandbox.utils import cpu_cap_env, read_text


LOG = logging.getLogger(__name__)

RUNNER_FILE = "sandbox_runner.py"
# Server-side backstop for a sandbox whose stop() never reaches the provider.
_TTL_MARGIN_S = 300
_LOG_TAIL_BYTES = 2000

T = TypeVar("T")


class SandboxVerificationConfig(BaseModel):
    """Sandbox settings for grading one submission per sandbox."""

    image: str | None = None
    # Interpreter inside the image that runs the runner.
    python: str = "python3"
    resources: SandboxResources = Field(default_factory=SandboxResources)
    ready_timeout_s: float = 600
    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("resources", mode="before")
    @classmethod
    def _parse_resources(cls, value: object) -> object:
        # from_mapping rejects unknown keys, so a typo fails at startup instead of being ignored.
        return SandboxResources.from_mapping(value) if isinstance(value, Mapping) else value


class SandboxVerificationError(Exception):
    """The sandbox or runner failed, so the sample has no valid reward."""

    def __init__(self, failure_kind: str, message: str) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


async def run_verification_sandbox(
    *,
    provider: Mapping[str, Any] | SandboxProvider,
    config: SandboxVerificationConfig,
    files: Mapping[str, str],
    request: Mapping[str, Any],
    exec_timeout_s: float,
    parse_result: Callable[[str], T],
    metadata: Mapping[str, str],
    work_dir: str,
) -> T:
    """Grade one submission in a fresh sandbox and return the parsed result.

    Args:
        provider: Provider instance or provider config mapping.
        config: Image, resources and provider settings.
        files: Text files to upload, keyed by path relative to ``work_dir``. Must include
            ``sandbox_runner.py``, which is run as ``sandbox_runner.py request.json result.json``.
        request: JSON-serializable request written to ``request.json``. It stays in the runner's
            working directory, so anything the runner launches can read it: keep expected outputs
            out of it or delete it before running the submission.
        exec_timeout_s: Deadline for the runner.
        parse_result: Parses and validates the text of ``result.json``. Any exception it raises
            is reported as a ``verifier_error``.
        metadata: Sandbox metadata; ``config.metadata`` entries win on conflicts.
        work_dir: Absolute directory in the sandbox; each call uses its own subdirectory of it for
            the uploaded files and the result.

    Raises:
        SandboxVerificationError: ``provider_unavailable`` if the sandbox could not start,
            ``verifier_error`` if the runner failed or its result is invalid.
    """
    if RUNNER_FILE not in files:
        raise ValueError(f"files must include {RUNNER_FILE}")
    # One directory per call: providers that share /tmp between sandboxes (local, and Apptainer, which binds the
    # host /tmp by default) would otherwise let concurrent verifications overwrite each other's files.
    work_dir = f"{work_dir.rstrip('/')}/{uuid.uuid4().hex}"
    uploads = {f"{work_dir}/{name}": content for name, content in files.items()}
    uploads[f"{work_dir}/request.json"] = json.dumps(request)
    spec = SandboxSpec(
        image=config.image,
        ttl_s=config.ready_timeout_s + exec_timeout_s + _TTL_MARGIN_S,
        ready_timeout_s=config.ready_timeout_s,
        env=cpu_cap_env(config.resources.cpu) | config.env,
        files=uploads,
        metadata={**metadata, **config.metadata},
        resources=config.resources,
        provider_options=config.provider_options,
    )

    sandbox = AsyncSandbox(provider)
    try:
        try:
            # Covers allocation, readiness and the file upload.
            await sandbox.start(spec)
        except ImportError:
            # A missing provider SDK is a deployment error, not an outage; masking it would drop every sample.
            raise
        except Exception as e:
            raise SandboxVerificationError(PROVIDER_UNAVAILABLE, f"sandbox start failed: {e!r}") from e

        # A submission may share the runner's stdout, so all output stays in the sandbox.
        command = (
            f"cd {shlex.quote(work_dir)} && "
            f"{shlex.quote(config.python)} {RUNNER_FILE} request.json result.json > runner.log 2>&1"
        )
        execution = await sandbox.exec(command, timeout_s=exec_timeout_s)
        # Providers report a timeout either as error_type or as exit code 124; both are nonzero here.
        if execution.error_type or execution.return_code != 0:
            raise SandboxVerificationError(
                VERIFIER_ERROR,
                f"runner failed (exit={execution.return_code}, error_type={execution.error_type}): "
                f"{(execution.stderr or '')[-_LOG_TAIL_BYTES:]}{await _runner_log_tail(sandbox, work_dir)}",
            )

        return parse_result(await read_text(sandbox, path=f"{work_dir}/result.json"))
    except (SandboxVerificationError, ImportError):
        raise
    except Exception as e:
        # Request boundary: any other sandbox or result failure invalidates only this sample.
        raise SandboxVerificationError(VERIFIER_ERROR, f"sandbox verification failed: {e!r}") from e
    finally:
        try:
            await sandbox.stop()
        except Exception:
            LOG.warning("Failed to stop verification sandbox", exc_info=True)


async def _runner_log_tail(sandbox: AsyncSandbox, work_dir: str) -> str:
    try:
        tail = await sandbox.exec(f"tail -c {_LOG_TAIL_BYTES} {shlex.quote(work_dir)}/runner.log", timeout_s=30)
    except Exception:
        return ""
    return f"\nrunner.log tail:\n{tail.stdout}" if tail.stdout else ""
