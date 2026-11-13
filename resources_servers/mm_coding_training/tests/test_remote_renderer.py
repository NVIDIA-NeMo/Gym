# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider receipt, cleanup and candidate/infra classification regressions."""

import io
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from nemo_gym.sandbox import SandboxExecResult, SandboxSpec
from resources_servers.mm_coding_training.remote_renderer import RemoteRenderer, image_uri
from resources_servers.mm_coding_training.tests.test_app import png, request
from resources_servers.mm_coding_training.tests.test_app import server as server
from resources_servers.mm_coding_training.tests.test_sessions import http_request


def renderer() -> RemoteRenderer:
    return RemoteRenderer(
        {"local": {}},
        SandboxSpec(image="test@sha256:" + "a" * 64, ttl_s=600),
        execution_timeout=420,
        ready_timeout=600,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate_exit", [1, 2, 124])
async def test_candidate_exit_is_zero_and_always_stops(candidate_exit: int) -> None:
    box = AsyncMock()
    box.exec.side_effect = [SandboxExecResult("", "", 0), SandboxExecResult("", "", candidate_exit)]
    with patch("resources_servers.mm_coding_training.remote_renderer.AsyncSandbox", return_value=box):
        assert await renderer().render("mmsvg_text2svg", "<svg/>") is None
    box.stop.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [SandboxExecResult("", "", code) for code in (137, 139, 143, -9, -11, -15, None)]
    + [SandboxExecResult("", "signal: killed", -1, None), SandboxExecResult("", "", -1, "timeout")],
)
async def test_inconclusive_exec_is_infra(failure: SandboxExecResult) -> None:
    box = AsyncMock()
    box.exec.side_effect = [SandboxExecResult("", "", 0), failure]
    with patch("resources_servers.mm_coding_training.remote_renderer.AsyncSandbox", return_value=box):
        with pytest.raises(RuntimeError):
            await renderer().render("chart2code", "raise SystemExit(137)")
    box.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_dependencies_are_checked_before_candidate_execution() -> None:
    box = AsyncMock()
    box.exec.return_value = SandboxExecResult("", "missing latex", 1)
    with patch("resources_servers.mm_coding_training.remote_renderer.AsyncSandbox", return_value=box):
        with pytest.raises(RuntimeError):
            await renderer().render("image2struct_latex", "x")
    assert box.exec.await_count == 1
    box.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_success_uploads_no_judge_keys_or_reference_and_downloads_bytes() -> None:
    box = AsyncMock()
    box.exec.return_value = SandboxExecResult("", "", 0)

    async def download(remote_path: str, local_path: Path) -> None:
        assert remote_path == "/workspace/render.png"
        local_path.write_bytes(png())

    box.download.side_effect = download
    with patch("resources_servers.mm_coding_training.remote_renderer.AsyncSandbox", return_value=box) as factory:
        assert await renderer().render("mmsvg_text2svg", "<svg/>") == png()
    spec = factory.call_args.args[1]
    assert set(spec.files) == {"/workspace/renderer.py", "/workspace/answer.svg"}
    assert set(spec.env) == {"MPLBACKEND", "OMP_NUM_THREADS"}
    box.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_failure_also_closes_provider() -> None:
    box = AsyncMock()
    box.start.side_effect = RuntimeError("test unavailable")
    with patch("resources_servers.mm_coding_training.remote_renderer.AsyncSandbox", return_value=box):
        with pytest.raises(RuntimeError):
            await renderer().render("chart2code", "x")
    box.stop.assert_awaited_once()


def test_empty_or_unsupported_image_is_rejected() -> None:
    with pytest.raises(ValueError):
        image_uri(b"")
    import io

    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (4, 4)).save(stream, format="BMP")
    with pytest.raises(ValueError):
        image_uri(stream.getvalue())


@pytest.mark.asyncio
async def test_cancelled_render_still_stops_the_sandbox() -> None:
    import asyncio

    import anyio

    box = AsyncMock()
    running, stopped = asyncio.Event(), []

    async def hang(*args, **kwargs):
        running.set()
        await asyncio.Event().wait()

    async def stop():
        await asyncio.sleep(0)
        stopped.append(True)

    box.exec.side_effect, box.stop.side_effect = hang, stop
    with patch("resources_servers.mm_coding_training.remote_renderer.AsyncSandbox", return_value=box):
        # A client disconnect cancels the verify request through an anyio task group.
        async with anyio.create_task_group() as requests:
            requests.start_soon(renderer().render, "chart2code", "x")
            await running.wait()
            requests.cancel_scope.cancel()
    assert stopped == [True]


@pytest.mark.asyncio
async def test_successfully_rendered_candidate_over_pixel_limit_is_unmasked_zero(server):
    data = io.BytesIO()
    Image.new("RGB", (1280, 25100)).save(data, format="PNG")
    box = AsyncMock()
    box.exec.return_value = SandboxExecResult("", "", 0)

    async def download(remote_path, local_path):
        local_path.write_bytes(data.getvalue())

    box.download.side_effect = download
    with (
        patch("resources_servers.mm_coding_training.remote_renderer.AsyncSandbox", return_value=box) as factory,
        patch.object(type(server), "_renderer", return_value=renderer()),
    ):
        result = await server.verify(http_request(), request("<html>tall page</html>", benchmark="design2code_full"))
    assert result.reward == 0 and result.evaluation_completed and not result.mask_sample
    assert result.failure_kind == "mm_coding_training:execution_failed"
    assert factory.call_count == 1
    box.stop.assert_awaited_once()
