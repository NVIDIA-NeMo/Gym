# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end verify() wire contract: an agent POSTs a standard ``BaseVerifyRequest``
(its response carries the normalized patch) and the verifier returns a non-nullable
``reward`` plus the eval-side fields, masking via reward=0.0."""

from __future__ import annotations

import asyncio

from nemo_gym.base_resources_server import BaseVerifyRequest
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxStatus, register_provider
from resources_servers.swe_env.app import SweEnvVerifier, SweEnvVerifierConfig


class _FakeProvider:
    """In-memory sandbox provider that replays canned pytest output or fails create."""

    name = "fake-http"

    def __init__(self, *, test_output="", create_error=False, **_):
        """Store the canned test output and whether sandbox creation should fail.

        Args:
            test_output: Stdout returned for any command containing ``pytest``.
            create_error: When True, ``create`` raises ``SandboxCreateError``.
        """
        self._test_output = test_output
        self._create_error = create_error

    async def create(self, spec):
        """Create a fake sandbox handle, or raise to simulate an infra failure.

        Args:
            spec: Sandbox spec whose workdir is echoed back in the handle.

        Returns:
            SandboxHandle: A handle naming this provider and the spec workdir.

        Raises:
            SandboxCreateError: When this provider was configured with create_error.
        """
        if self._create_error:
            from nemo_gym.sandbox import SandboxCreateError

            raise SandboxCreateError("boom")
        return SandboxHandle(sandbox_id="h", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        """Replay the canned output for pytest commands; return success otherwise.

        Args:
            handle: Sandbox handle (unused).
            command: Command string; pytest commands replay the canned output.
            cwd: Working directory (unused).
            env: Environment variables (unused).
            timeout_s: Execution timeout (unused).
            user: User to run as (unused).

        Returns:
            SandboxExecResult: The canned pytest result, or an empty success result.
        """
        if "pytest" in command:
            return SandboxExecResult(stdout=self._test_output, stderr="", return_code=0)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, *a, **k):
        """Accept an upload request and do nothing."""
        return None

    async def download_file(self, *a, **k):
        """Accept a download request and do nothing."""
        return None

    async def status(self, handle):
        """Report the sandbox as running.

        Args:
            handle: Sandbox handle (unused).

        Returns:
            SandboxStatus: Always ``RUNNING``.
        """
        return SandboxStatus.RUNNING

    async def close(self, handle):
        """Close the given sandbox handle and do nothing.

        Args:
            handle: Sandbox handle (unused).
        """
        return None

    async def aclose(self):
        """Close the provider and do nothing."""
        return None


register_provider("fake-http", _FakeProvider, override=True)

_PATCH = "diff --git a/x b/x\n"
_METADATA = {
    "instance_id": "http-e2e",
    "image": "img:tag",
    "base_commit": "HEAD",
    "test_command": "python -m pytest -rA -q",
    "test_framework": "pytest",
    "fail_to_pass": '["test_calc.py::test_add"]',
    "benchmark": "swe-bench-ext",
}


def _request(patch: str) -> BaseVerifyRequest:
    """Build a verify request whose response metadata carries the given patch.

    Args:
        patch: The model patch placed in the response's ``model_patch`` metadata.

    Returns:
        BaseVerifyRequest: A request pairing the standard task metadata with the patch.
    """
    params = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "fix the bug"}], metadata=dict(_METADATA)
    )
    response = NeMoGymResponse(
        id="resp-1",
        created_at=0,
        model="m",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        metadata={"model_patch": patch},
    )
    return BaseVerifyRequest(responses_create_params=params, response=response)


def _verifier(provider_cfg) -> SweEnvVerifier:
    """Build a verifier wired to the given sandbox provider configuration.

    Args:
        provider_cfg: The sandbox provider config (provider name to kwargs).

    Returns:
        SweEnvVerifier: A verifier reading the patch from the ``model_patch`` field.
    """
    cfg = SweEnvVerifierConfig.model_construct(sandbox_provider=provider_cfg, model_patch_field="model_patch")
    return SweEnvVerifier.model_construct(config=cfg)


def test_verify_returns_reward_for_resolving_patch():
    """Check that a resolving patch yields reward 1.0 and the eval-side fields."""
    # Trailing-status pytest text is the format the lighthouse parser recognizes
    # (the ``.py`` path normalizes to the F2P id in _METADATA).
    verifier = _verifier({"fake-http": {"test_output": "test_calc.py::test_add PASSED\n"}})
    out = asyncio.run(verifier.verify(_request(_PATCH)))
    assert isinstance(out.reward, float)
    assert out.reward == 1.0
    assert out.resolved is True
    assert out.mask_sample is False
    assert out.instance_id == "http-e2e"


def test_verify_masks_infra_error_as_zero_not_none():
    """Check that an infra failure masks the sample with reward 0.0 rather than None."""
    verifier = _verifier({"fake-http": {"create_error": True}})
    out = asyncio.run(verifier.verify(_request(_PATCH)))
    assert out.reward == 0.0  # never None (non-nullable wire field)
    assert out.eval_error is True
    assert out.mask_sample is True


def test_verify_empty_patch_unresolved():
    """Check that an empty patch yields reward 0.0 and patch_exists False."""
    verifier = _verifier({"fake-http": {}})
    out = asyncio.run(verifier.verify(_request("")))
    assert out.reward == 0.0
    assert out.patch_exists is False
