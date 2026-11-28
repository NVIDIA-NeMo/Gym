# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest

from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
from nemo_gym.sandbox.providers.base import SandboxHandle
from responses_api_agents.kernel_gym import app
from responses_api_agents.kernel_gym.tests.test_app import sandbox_run  # noqa: F401


@pytest.mark.parametrize("failure", [None, "upload", "agent_cancel", "verifier_cancel"])
async def test_run_kernel_uses_real_sandbox_lifecycle(sandbox_run, monkeypatch, failure):
    cfg, _, _, _, agent_response = sandbox_run
    providers = []
    events = []

    class Provider:
        name = "test"

        def __init__(self):
            self.index = len(providers)
            self.files = {}
            providers.append(self)

        async def create(self, spec):
            events.append((self.index, "create"))
            return SandboxHandle(str(self.index), self.name, None)

        async def upload_file(self, handle, source, target):
            if failure == "upload":
                raise OSError("upload failed")
            self.files[target] = source.read_text()

        async def exec(self, handle, command, **kwargs):
            events.append((self.index, "exec"))
            assert self.files["/workspace/reference.py"] == "original reference"
            if self.index == 0:
                if failure == "agent_cancel":
                    raise asyncio.CancelledError()
                self.files["/workspace/solution.py"] = "submitted solution"
                self.files["/trajectories_mount/response.json"] = json.dumps(agent_response)
            else:
                assert events.index((0, "close")) < events.index((1, "create"))
                assert self.files["/workspace/solution.py"] == "submitted solution"
                assert self.files["/tests/verify.py"] == "trusted verifier"
                if failure == "verifier_cancel":
                    raise asyncio.CancelledError()
                self.files.update(
                    {
                        "/logs/verifier/result.json": json.dumps(
                            {"evaluation_completed": True, "correctness": True, "speedup": 2}
                        ),
                        "/logs/verifier/reward.txt": "1",
                        "/logs/verifier/test-stdout.txt": "passed",
                    }
                )
            return SandboxExecResult("", "", 0)

        async def download_file(self, handle, source, target):
            target.write_text(self.files[source])

        async def close(self, handle):
            events.append((self.index, "close"))

        async def aclose(self):
            events.append((self.index, "aclose"))

    # Keep the actual wrapper: __aenter__, start/files, exec, and stop must all run.
    monkeypatch.setattr(app, "AsyncSandbox", lambda config, spec: AsyncSandbox(Provider(), spec))
    if failure in ("agent_cancel", "verifier_cancel"):
        with pytest.raises(asyncio.CancelledError):
            await app.KernelGymAgent._run_kernel(None, cfg)
    else:
        assert await app.KernelGymAgent._run_kernel(None, cfg) is (failure is None)
        metrics = json.loads(cfg.metrics_fpath.read_text())
        assert metrics["evaluation_completed"] is (failure is None)
        assert metrics["mask_sample"] is (failure is not None)

    assert len(providers) == (1 if failure in ("upload", "agent_cancel") else 2)
    for provider in providers:
        assert events.count((provider.index, "create")) == 1
        assert events.count((provider.index, "close")) == 1
        assert events.count((provider.index, "aclose")) == 1
    if failure == "upload":
        assert not any(event == "exec" for _, event in events)
