# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolation checks depend on renderer results, not scheduler membership."""

import argparse
import json
from unittest.mock import AsyncMock

import pytest

from resources_servers.vision2code import smoke
from resources_servers.vision2code.render import RenderResult


@pytest.mark.parametrize("slurm_job_id", [None, "123"])
@pytest.mark.parametrize("status", ["ok", "infrastructure_failure"])
async def test_smoke_on_local_and_slurm_workers(tmp_path, monkeypatch, slurm_job_id, status):
    if slurm_job_id is None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_ID", slurm_job_id)
    renderer = AsyncMock(return_value=RenderResult(status=status))
    monkeypatch.setattr(smoke, "render", renderer)
    args = argparse.Namespace(image=tmp_path / "renderer.sqsh", artifacts=tmp_path, forbid_path=["/private/reference"])
    if status == "ok":
        await smoke.main_async(args)
    else:
        with pytest.raises(SystemExit, match="Isolation smoke did not pass"):
            await smoke.main_async(args)
    report = json.loads((tmp_path / "isolation.json").read_text())
    assert report["status"] == status
    assert report["slurm_job_id"] == slurm_job_id
    assert "/private/reference" in renderer.call_args.kwargs["code"]
