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
import json
import sys
from pathlib import Path

import pytest

from nemo_gym.core import PROVIDER_UNAVAILABLE, VERIFIER_ERROR
from nemo_gym.sandbox import SandboxCreateError, SandboxExecResult, SandboxHandle, SandboxSpec
from nemo_gym.verification import SandboxVerificationConfig, SandboxVerificationError, run_verification_sandbox
from tests.unit_tests.test_sandbox import FakeSandboxProvider


pytestmark = pytest.mark.sandbox

_RUNNER = "import json, sys\njson.dump({'ok': json.load(open(sys.argv[1]))['x'] * 2}, open(sys.argv[2], 'w'))\n"


class _VerificationProvider(FakeSandboxProvider):
    """Keeps uploaded text, replays a canned runner outcome and serves a canned result file."""

    def __init__(
        self,
        *,
        create_error: Exception | None = None,
        runner_result: SandboxExecResult | None = None,
        result_json: str = '{"ok": 42}',
    ) -> None:
        super().__init__()
        self._create_error = create_error
        self._runner_result = runner_result or SandboxExecResult(stdout="", stderr="", return_code=0)
        self._result_json = result_json
        self.uploaded: dict[str, str] = {}

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if self._create_error is not None:
            raise self._create_error
        return await super().create(spec)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        await super().upload_file(handle, source_path, target_path)
        self.uploaded[target_path] = Path(source_path).read_text()

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: object) -> SandboxExecResult:
        await super().exec(handle, command, **kwargs)
        if command.startswith("tail "):
            return SandboxExecResult(stdout="Traceback: boom\n", stderr="", return_code=0)
        return self._runner_result

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        self.download_calls.append((handle, source_path, target_path))
        target_path.write_text(self._result_json)


def _parse(raw: str) -> int:
    return json.loads(raw)["ok"]


async def _run(provider, *, config=None, files=None, metadata=None, work_dir="/tmp/verify"):
    return await run_verification_sandbox(
        provider=provider,
        config=config or SandboxVerificationConfig(image="verifier:test", resources={"cpu": 4}),
        files=files or {"sandbox_runner.py": _RUNNER, "lib/helper.py": "X = 1\n"},
        request={"x": 21},
        exec_timeout_s=120,
        parse_result=_parse,
        metadata=metadata or {"benchmark": "toy"},
        work_dir=work_dir,
    )


def _call_dir(provider: _VerificationProvider) -> str:
    (call_dir,) = {path.rsplit("/", 1)[0] for path in provider.uploaded if path.endswith("/request.json")}
    assert call_dir.startswith("/tmp/verify/") and "/" not in call_dir.removeprefix("/tmp/verify/")
    return call_dir


async def test_builds_spec_runs_runner_and_parses_result() -> None:
    provider = _VerificationProvider()
    config = SandboxVerificationConfig(
        image="verifier:test", resources={"cpu": 4}, env={"EXTRA": "1"}, metadata={"benchmark": "override"}
    )

    assert await _run(provider, config=config, metadata={"benchmark": "toy", "run": "r1"}) == 42

    (spec,) = provider.created_specs
    assert spec.image == "verifier:test"
    assert spec.env["OMP_NUM_THREADS"] == "4" and spec.env["EXTRA"] == "1"
    assert spec.metadata == {"benchmark": "override", "run": "r1"}
    assert spec.ttl_s > spec.ready_timeout_s + 120
    call_dir = _call_dir(provider)
    assert provider.uploaded[f"{call_dir}/lib/helper.py"] == "X = 1\n"
    assert json.loads(provider.uploaded[f"{call_dir}/request.json"]) == {"x": 21}
    (runner_call,) = provider.exec_calls
    assert runner_call["command"] == (
        f"cd {call_dir} && python3 sandbox_runner.py request.json result.json > runner.log 2>&1"
    )
    assert runner_call["timeout_s"] == 120
    assert provider.download_calls[0][1] == f"{call_dir}/result.json"
    assert len(provider.closed) == 1


async def test_each_call_gets_its_own_work_dir() -> None:
    # Providers that share /tmp between sandboxes (local, Apptainer) would otherwise mix up concurrent calls.
    first, second = _VerificationProvider(), _VerificationProvider()
    await _run(first, work_dir="/tmp/verify/")
    await _run(second)
    assert _call_dir(first) != _call_dir(second)


async def test_create_failure_is_provider_unavailable() -> None:
    provider = _VerificationProvider(create_error=SandboxCreateError("no capacity"))
    with pytest.raises(SandboxVerificationError) as info:
        await _run(provider)
    assert info.value.failure_kind == PROVIDER_UNAVAILABLE
    assert "no capacity" in str(info.value)


async def test_missing_provider_sdk_fails_loudly() -> None:
    # A venv without nemo-gym[sandbox] must not turn every sample into a masked provider_unavailable.
    provider = _VerificationProvider(create_error=ModuleNotFoundError("OpenSandbox SDK is required"))
    with pytest.raises(ModuleNotFoundError, match="OpenSandbox SDK is required"):
        await _run(provider)


@pytest.mark.parametrize(
    "runner_result",
    [
        # ECS-style: a timeout reported only as exit code 124.
        SandboxExecResult(stdout="", stderr="timed out after 120s", return_code=124),
        SandboxExecResult(stdout="", stderr="", return_code=None, error_type="timeout"),
    ],
)
async def test_runner_failure_is_verifier_error_with_log_tail(runner_result: SandboxExecResult) -> None:
    provider = _VerificationProvider(runner_result=runner_result)
    with pytest.raises(SandboxVerificationError) as info:
        await _run(provider)
    assert info.value.failure_kind == VERIFIER_ERROR
    assert "runner failed" in str(info.value) and "Traceback: boom" in str(info.value)
    assert len(provider.closed) == 1


async def test_result_the_parser_rejects_is_verifier_error() -> None:
    provider = _VerificationProvider(result_json='{"not_ok": 1}')
    with pytest.raises(SandboxVerificationError) as info:
        await _run(provider)
    assert info.value.failure_kind == VERIFIER_ERROR
    assert "KeyError" in str(info.value)
    assert len(provider.closed) == 1


async def test_stop_failure_does_not_mask_a_valid_result(caplog: pytest.LogCaptureFixture) -> None:
    provider = _VerificationProvider()

    async def failing_close(handle: SandboxHandle) -> None:
        raise RuntimeError("provider gone")

    provider.close = failing_close
    assert await _run(provider) == 42
    assert "Failed to stop verification sandbox" in caplog.text


async def test_files_without_runner_are_rejected() -> None:
    with pytest.raises(ValueError, match="sandbox_runner.py"):
        await _run(_VerificationProvider(), files={"lib/helper.py": ""})


def test_unknown_resource_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown sandbox resource keys: memory"):
        SandboxVerificationConfig(image="x", resources={"memory": 4096})


async def test_runs_the_runner_through_the_local_provider(tmp_path: Path) -> None:
    result = await _run(
        {"local": {}},
        config=SandboxVerificationConfig(python=sys.executable),
        work_dir=str(tmp_path / "work"),
    )
    assert result == 42
