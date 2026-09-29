# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.swebench_pro.app import SWEBenchProInstanceRequest, SWEBenchProSeedSessionRequest
from resources_servers.swebench_pro.tests.test_app import make_server, request_body
from resources_servers.swebench_pro.verification import VerificationResult, inconclusive_reason


@pytest.mark.parametrize("failure", [OSError("stop failed"), TimeoutError("stop timed out")])
async def test_failed_seed_preserves_error_and_retains_cleanup_handle(failure):
    server = make_server(golden=False)
    sandbox = SimpleNamespace(
        upload=AsyncMock(side_effect=ValueError("seed failed")), stop=AsyncMock(side_effect=failure)
    )
    server._create_sandbox = AsyncMock(return_value=sandbox)
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})
    with pytest.raises(ValueError, match="seed failed"):
        await server.seed_session(request, SWEBenchProSeedSessionRequest.model_validate(request_body()))
    assert server._session_id_to_sandbox["session"] is sandbox
    sandbox.stop.side_effect = None
    await server.shutdown()
    assert sandbox.stop.await_count == 2
    assert not server._session_id_to_sandbox


async def test_close_and_reseed_do_not_forget_failed_cleanup():
    server = make_server(golden=False)
    sandbox = SimpleNamespace(stop=AsyncMock(side_effect=OSError("stop failed")))
    server._session_id_to_sandbox["session"] = sandbox
    server._create_sandbox = AsyncMock()
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})
    for operation in (
        lambda: server.close_session(request),
        lambda: server.seed_session(request, SWEBenchProSeedSessionRequest.model_validate(request_body())),
    ):
        with pytest.raises(OSError, match="stop failed"):
            await operation()
        assert server._session_id_to_sandbox["session"] is sandbox
        server._create_sandbox.assert_not_awaited()
    sandbox.stop.side_effect = None
    assert await server.close_session(request) == {"closed": True}
    assert not server._session_id_to_sandbox


async def test_cancelled_seed_still_tracks_failed_cleanup():
    server = make_server(golden=False)
    started = asyncio.Event()

    async def upload(*args):
        started.set()
        await asyncio.Event().wait()

    sandbox = SimpleNamespace(upload=upload, stop=AsyncMock(side_effect=OSError("stop failed")))
    server._create_sandbox = AsyncMock(return_value=sandbox)
    task = asyncio.create_task(
        server.seed_session(
            SimpleNamespace(session={SESSION_ID_KEY: "session"}),
            SWEBenchProSeedSessionRequest.model_validate(request_body()),
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert server._session_id_to_sandbox["session"] is sandbox


def test_image_validation_reuses_hash_but_rechecks_changed_file_or_manifest(tmp_path, monkeypatch):
    from resources_servers.swebench_pro import image_cache

    image = tmp_path / "task.sif"
    image.write_bytes(b"task image")
    digest = "sha256:" + "a" * 64
    manifest = image.with_suffix(".sif.json")
    provenance = {
        "source_uri": f"docker://docker.io/jefzda/sweap-images@{digest}",
        "sif_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
    }
    manifest.write_text(json.dumps(provenance))
    checksum = Mock(wraps=image_cache.sif_checksum)
    monkeypatch.setattr(image_cache, "sif_checksum", checksum)
    server = make_server(golden=False, image_template=str(image))
    body = SWEBenchProInstanceRequest.model_validate(request_body() | {"image_digest": digest})
    first = server._image_info(body)
    assert server._image_info(body) == first
    checksum.assert_called_once()
    # Metadata changes cannot make an old successful hash validate a new file.
    image.write_bytes(b"modified task image")
    with pytest.raises(ValueError, match="checksum mismatch"):
        server._image_info(body)
    manifest.write_text(json.dumps(provenance | {"source_uri": "wrong"}))
    with pytest.raises(ValueError, match="source does not match"):
        server._image_info(body)


@pytest.mark.parametrize(
    "error",
    [
        "npm error code EAI_AGAIN\nnpm error request to https://registry.npmjs.org failed: getaddrinfo EAI_AGAIN",
        "curl: (6) Could not resolve host: registry.npmjs.org",
    ],
)
def test_empty_report_after_dependency_network_failure_remains_inconclusive(error):
    result = VerificationResult(
        completed=True,
        resolved=False,
        patch_applied=True,
        test_results={"tests": []},
        test_output=error,
    )
    assert inconclusive_reason(result) is not None


def test_empty_report_after_model_compile_error_keeps_its_zero_score():
    result = VerificationResult(
        completed=True,
        resolved=False,
        patch_applied=True,
        test_results={"tests": []},
        test_output="src/app.go:12: unknown field clientFunc in struct literal [build failed]",
    )
    assert inconclusive_reason(result) is None
