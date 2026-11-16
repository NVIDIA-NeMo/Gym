# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agentic setting's lent sandbox (sandbox.py), on the local sandbox provider in a temporary directory.

The local provider runs every command on this host, so the image's /opt/video-mcqa is a temporary copy of
sandbox_image, the media root is a temporary /testbed, and the tools' own check is replaced by a stand-in program
(the image runs `python3 -I -m video_tools.cli --check`, which needs FFmpeg; test_video_tools.py covers it).
"""

import hashlib
import json
import shutil
import stat
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from resources_servers.video_mcqa import sandbox as video_sandbox
from resources_servers.video_mcqa.app import VideoMCQAResourcesServerConfig
from resources_servers.video_mcqa.prepare import prepare_dataset
from resources_servers.video_mcqa.sandbox import SANDBOX_IMAGE_DIR, TOOL_FILES
from resources_servers.video_mcqa.tests.helpers import REVISION, media, v1_record
from resources_servers.videomme_v1.app import VideoMMEV1ResourcesServer


CLIP = Path(__file__).resolve().parents[1] / "data" / "example_videos" / "synthetic-1.mp4"
CLIP_SHA256 = hashlib.sha256(CLIP.read_bytes()).hexdigest()


@pytest.fixture
def host(tmp_path, monkeypatch) -> Path:
    """A host directory that stands in for the sandbox's filesystem: the image's tools, its media root, the videos."""
    monkeypatch.setattr(
        "nemo_gym.server_runtime.sandbox_lending.get_global_config_dict", lambda: {"sandbox": {"local": {}}}
    )
    # The local provider cannot be reattached from another process, so it has no descriptor to hand out.
    monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(return_value={"sandbox_id": "local"}))
    # It also runs every command as this host's user: the transfers' and the check's root commands run as that user.
    real_exec = AsyncSandbox.exec

    async def exec_as_host_user(self, command, **kwargs):
        kwargs.pop("user", None)
        return await real_exec(self, command, **kwargs)

    monkeypatch.setattr(AsyncSandbox, "exec", exec_as_host_user)
    tools = tmp_path / "opt" / "video-mcqa"
    for name in TOOL_FILES:
        (tools / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SANDBOX_IMAGE_DIR / name, tools / name)
    monkeypatch.setattr(video_sandbox, "TOOLS_DIR", tools)
    (tmp_path / "testbed" / "work").mkdir(parents=True)
    (tmp_path / "videos").mkdir()
    shutil.copy(CLIP, tmp_path / "videos" / "shapes.mp4")
    return tmp_path


@pytest.fixture
def stopped(monkeypatch) -> list[AsyncSandbox]:
    stopped: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        stopped.append(self)
        await real_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    return stopped


def _cohort(path: Path) -> tuple[Path, dict]:
    rows, cohort = prepare_dataset(
        [v1_record()],
        benchmark="videomme_v1",
        dataset_revision=REVISION,
        media=media(source=str(path / "videos" / "shapes.mp4"), sandbox_path="videos/shapes.mp4", sha256=CLIP_SHA256),
        setting="agentic",
        sandbox_root=str(path / "testbed"),
    )
    manifest = path / "cohort.json"
    manifest.write_text(cohort.model_dump_json())
    return manifest, rows[0]


def _server(host: Path, *, check: str = "import sys; sys.exit(0)", **fields) -> VideoMMEV1ResourcesServer:
    manifest, _ = _cohort(host)
    config = VideoMCQAResourcesServerConfig(
        name="videomme_v1",
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        cohort_manifest=manifest,
        **{
            "sandbox_provider": "sandbox",
            "sandbox_config": {"image": "unused-by-the-local-provider"},
            "sandbox_media_root": str(host / "testbed"),
            "sandbox_workdir": str(host / "testbed" / "work"),
            "video_roots": [str(host / "videos")],
            "video_tools": {"command": [sys.executable, "-c", check]},
            **fields,
        },
    )
    return VideoMMEV1ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _seed(host: Path, session_id: str = "session", **binding: object) -> ResourcesSeedSessionRequest:
    _, row = _cohort(host)
    row = {**row, "video_binding": {**row["video_binding"], **binding}}
    task_data = {key: value for key, value in row.items() if key != "responses_create_params"}
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=f"rollout-{session_id}"),
        task_id=TaskId(taskset="videomme_v1:agentic", task_id=row["task_id"]),
        task_data=task_data,
    )


def test_seed_puts_the_checked_video_and_the_tool_settings_into_the_sandbox(host, stopped) -> None:
    server = _server(host)
    client = TestClient(server.setup_webserver())

    seeded = client.post("/seed_session", json=_seed(host).model_dump(mode="json"))

    assert seeded.status_code == 200, seeded.text
    assert seeded.json()["sandbox_access"]["workdir"] == str(host / "testbed" / "work")
    video = host / "testbed" / "videos" / "shapes.mp4"
    assert video.read_bytes() == CLIP.read_bytes()
    # Read-only for everyone once checked.
    assert stat.S_IMODE(video.stat().st_mode) == 0o444
    tools = json.loads((video_sandbox.TOOLS_DIR / "tools.json").read_text())
    assert tools["workspace_root"] == str(host / "testbed") and tools["cache_dir"] == "/tmp/video-mcqa/index"
    assert tools["profile"]["patch_size"] == 14 and tools["profile"]["downsample_ratio"] == 0.5
    assert (video_sandbox.TOOLS_DIR / "video.sha256").read_text() == f"{CLIP_SHA256}  {video}\n"
    # The row's prompt names the video where the seed put it.
    _, row = _cohort(host)
    assert f"Video: {video}" in row["responses_create_params"]["input"][0]["content"][0]["text"]

    # A repeated seed returns the same sandbox and uploads nothing again.
    video.chmod(0o644)
    video.write_bytes(b"left alone")
    again = client.post("/seed_session", json=_seed(host).model_dump(mode="json"))
    assert again.status_code == 200 and again.json() == seeded.json()
    assert video.read_bytes() == b"left alone"

    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=_seed(host).episode_id)
    assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200
    assert len(stopped) == 1 and server._lent_sandboxes.get("session") is None
    assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200
    assert len(stopped) == 1


def _refused(server, seed: ResourcesSeedSessionRequest) -> str:
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    response = client.post("/seed_session", json=seed.model_dump(mode="json"))
    assert response.status_code == 422, response.text
    assert server._lent_sandboxes.get(seed.resources_session_id) is None
    return response.text


@pytest.mark.parametrize(
    ("binding", "message"),
    [
        ({"source": "/etc/hostname"}, "video_roots"),
        ({"source": "missing.mp4"}, "Cannot read"),
        ({"sandbox_path": "../outside.mp4"}, "valid video_binding"),
    ],
)
def test_a_row_whose_video_the_server_cannot_serve_is_refused_before_a_sandbox_starts(
    host, stopped, binding, message
) -> None:
    assert message in _refused(_server(host), _seed(host, **binding))
    assert stopped == []


def test_a_row_without_a_video_binding_is_refused(host, stopped) -> None:
    seed = _seed(host)
    seed = seed.model_copy(
        update={"task_data": {key: value for key, value in seed.task_data.items() if key != "video_binding"}}
    )
    assert "video_binding" in _refused(_server(host), seed)
    assert stopped == []


def test_a_video_without_the_rows_digest_is_refused_and_its_sandbox_stopped(host, stopped) -> None:
    assert "video_binding.sha256" in _refused(_server(host), _seed(host, sha256="0" * 64))
    assert len(stopped) == 1


def test_an_image_whose_tools_differ_from_this_repository_is_refused(host, stopped) -> None:
    with (video_sandbox.TOOLS_DIR / "video_tools" / "media.py").open("a") as stream:
        stream.write("# a stale image\n")
    assert "build the image" in _refused(_server(host), _seed(host))
    assert len(stopped) == 1


def test_tools_that_cannot_run_are_refused(host, stopped) -> None:
    server = _server(host, check="import sys; print('ffmpeg is not on PATH', file=sys.stderr); sys.exit(1)")
    assert "ffmpeg is not on PATH" in _refused(server, _seed(host))
    assert len(stopped) == 1


@pytest.mark.parametrize(
    "result",
    [
        SandboxExecResult(stdout=None, stderr=None, return_code=-1),
        SandboxExecResult(stdout=None, stderr=None, return_code=137),
    ],
    ids=["signal_sentinel", "killed"],
)
def test_a_check_that_did_not_finish_is_retried_and_its_sandbox_stopped(host, stopped, monkeypatch, result) -> None:
    monkeypatch.setattr(video_sandbox, "exec_as_root", AsyncMock(return_value=result))
    server = _server(host)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    response = client.post("/seed_session", json=_seed(host).model_dump(mode="json"))
    assert response.status_code >= 500
    assert len(stopped) == 1 and server._lent_sandboxes.get("session") is None
    # The seed did not bind the session, so the Environment Server's retry seeds it again.
    assert "session" not in server._sessions


def test_a_check_the_provider_could_not_run_is_retried(host, stopped, monkeypatch) -> None:
    real_exec = AsyncSandbox.exec

    async def exec_(self, command, **kwargs):
        if "sha256sum" in command:
            return SandboxExecResult(stdout=None, stderr="lost", return_code=124, error_type="timeout")
        return await real_exec(self, command, **kwargs)

    monkeypatch.setattr(AsyncSandbox, "exec", exec_)
    server = _server(host)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=_seed(host).model_dump(mode="json")).status_code >= 500
    assert len(stopped) == 1


def test_a_lending_server_lends_only_to_a_typed_episode_seed(host, stopped) -> None:
    client = TestClient(_server(host).setup_webserver())
    cookie_seed = {"task_id": "videomme_v1:one", "setting": "agentic"}
    assert client.post("/seed_session", json=cookie_seed).status_code == 422
    assert stopped == []


def test_lent_sandbox_follows_the_session_contract(host, stopped) -> None:
    server = _server(host)
    check_resources_session_contract(
        server.setup_webserver(),
        _seed(host, "contract"),
        keeps_state=True,
        holds_state=lambda resources_session_id: server._lent_sandboxes.get(resources_session_id) is not None,
    )


def test_shutdown_stops_every_sandbox_still_lent(host, stopped) -> None:
    server = _server(host)
    with TestClient(server.setup_webserver()) as client:
        for session in ("a", "b"):
            assert client.post("/seed_session", json=_seed(host, session).model_dump(mode="json")).status_code == 200
        assert stopped == []
    assert len(stopped) == 2


def test_without_a_sandbox_provider_the_server_stays_stateless_and_scales(host) -> None:
    manifest, _ = _cohort(host)
    config = VideoMCQAResourcesServerConfig(
        name="videomme_v1", host="0.0.0.0", port=8080, entrypoint="", cohort_manifest=manifest
    )
    server = VideoMMEV1ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    seeded = TestClient(server.setup_webserver()).post("/seed_session", json=_seed(host).model_dump(mode="json"))
    assert seeded.status_code == 200 and seeded.json()["sandbox_access"] is None
    assert server.multiple_workers_unsupported_reason() is None
    assert _server(host).multiple_workers_unsupported_reason() == "lent sandboxes live in one worker's memory"


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"sandbox_config": {}}, "Set sandbox_config.image"),
        ({"video_roots": []}, "Set video_roots"),
        ({"sandbox_workdir": "/workspace"}, "inside sandbox_media_root"),
        ({"sandbox_media_root": "relative"}, "absolute sandbox directory"),
        ({"sandbox_provider": None, "sandbox_network_access": "model_only"}, "set sandbox_provider"),
        ({"video_tools": {"profile": {"downsample_ratio": 0.3}}}, "reciprocal"),
    ],
)
def test_sandbox_settings_that_cannot_work_are_refused_at_startup(host, fields, message) -> None:
    with pytest.raises(ValueError, match=message):
        _server(host, **fields)
