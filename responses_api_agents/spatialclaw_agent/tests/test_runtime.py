# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.spatialclaw_agent import app
from responses_api_agents.spatialclaw_agent.tests.test_observability import agent_config


def agent(**overrides):
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {
        "model": {"responses_api_models": {"openai_model": {"host": "localhost", "port": 99}}}
    }
    client._build_server_base_url.return_value = "http://localhost:99"
    return app.SpatialClawAgent(config=agent_config(**overrides), server_client=client)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "origin"
    (root / "spatial_agent").mkdir(parents=True)
    (root / "spatial_agent/workflow.py").write_text("# Native workflow boundary\n")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    git("init", "-q")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "fixture",
    )
    return root, git("rev-parse", "HEAD")


def test_checkout_enforces_full_pin_and_recovers_cached_revision(source, tmp_path):
    root, commit = source
    with pytest.raises(ValueError, match="full 40"):
        app.ensure_spatialclaw_checkout(str(root), commit[:8], tmp_path / "cache")
    cached = app.ensure_spatialclaw_checkout(str(root), commit, tmp_path / "cache")
    assert app._validate_spatialclaw_checkout(cached, commit) == cached
    subprocess.run(
        [
            "git",
            "-C",
            str(cached),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-qm",
            "wrong revision",
        ],
        check=True,
    )
    with pytest.raises(RuntimeError, match="expected pinned"):
        app._validate_spatialclaw_checkout(cached, commit)
    assert app.ensure_spatialclaw_checkout(str(root), commit, tmp_path / "cache") == cached
    assert subprocess.check_output(["git", "-C", str(cached), "rev-parse", "HEAD"], text=True).strip() == commit
    with pytest.raises(RuntimeError, match="Invalid SpatialClaw"):
        app._validate_spatialclaw_checkout(tmp_path, commit)
    untracked = tmp_path / "not-a-repository"
    (untracked / "spatial_agent").mkdir(parents=True)
    (untracked / "spatial_agent/workflow.py").touch()
    with pytest.raises(RuntimeError, match="not a Git worktree"):
        app._validate_spatialclaw_checkout(untracked, commit)


async def test_resolve_local_and_cached_runtime_without_mutating_other_sources(source, tmp_path, monkeypatch):
    root, commit = source
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setenv("PYTHONPATH", "")
    local = agent(spatialclaw_root=str(root), spatialclaw_commit=commit)
    assert await local._resolve_spatialclaw_root() == root
    assert os.environ["PYTHONPATH"] == str(root)
    assert await local._resolve_spatialclaw_root() == root
    remote = agent(spatialclaw_url=str(root), spatialclaw_commit=commit, source_cache_root=str(tmp_path / "cache"))
    cached = await remote._resolve_spatialclaw_root()
    assert cached != root and (cached / "spatial_agent/workflow.py").exists()


def test_gpu_registry_link_is_idempotent_and_rejects_replacement(tmp_path):
    registry = tmp_path / "live.json"
    app._configure_gpu_registry(tmp_path / "client", None)
    with pytest.raises(FileNotFoundError):
        app._configure_gpu_registry(tmp_path / "client", str(registry))
    registry.write_text("{}")
    root = tmp_path / "client"
    app._configure_gpu_registry(root, str(registry))
    app._configure_gpu_registry(root, str(registry))
    assert (root / "spatial_agent/logs/gpu_server.json").resolve() == registry
    other = tmp_path / "other.json"
    other.write_text("{}")
    with pytest.raises(RuntimeError, match="refusing to replace"):
        app._configure_gpu_registry(root, str(other))
    assert (root / "spatial_agent/logs/gpu_server.json").resolve() == registry


@pytest.mark.parametrize(
    "key,value",
    [("video_paths", "42"), ("image_paths", "[]oops"), ("image_paths", "[42]"), ("image_groups", '["flat"]')],
)
async def test_rejects_malformed_media_metadata(tmp_path, key, value):
    body = NeMoGymResponseCreateParamsNonStreaming(input="question", metadata={key: value})
    with pytest.raises(ValueError, match="must"):
        await agent()._materialize_inputs(body, tmp_path, SimpleNamespace())


@pytest.mark.parametrize("metadata", [{}, {"image_paths": '["a.png"]', "image_groups": '[["a.png"]]'}])
async def test_rejects_missing_or_ambiguous_media(tmp_path, metadata):
    with pytest.raises(ValueError):
        await agent()._materialize_inputs(
            NeMoGymResponseCreateParamsNonStreaming(input="question", metadata=metadata), tmp_path, SimpleNamespace()
        )
    with pytest.raises(ValueError, match="no text"):
        await agent()._materialize_inputs(
            NeMoGymResponseCreateParamsNonStreaming(input=[]), tmp_path, SimpleNamespace()
        )


async def test_materializes_inline_local_and_downloaded_media(tmp_path, monkeypatch):
    output = tmp_path / "media/file"
    assert await app._materialize_url("data:image/png;base64,YWJj", output) == str(output)
    assert output.read_bytes() == b"abc"
    await app._materialize_url("data:text/plain,a%20b", output)
    assert output.read_bytes() == b"a b"
    assert await app._materialize_url(output.as_uri(), output) == str(output)
    with pytest.raises(FileNotFoundError):
        await app._materialize_url((tmp_path / "missing").as_uri(), output)
    with pytest.raises(FileNotFoundError):
        await app._materialize_url(str(tmp_path / "missing"), output)
    response = SimpleNamespace(read=AsyncMock(return_value=b"downloaded"))
    request = AsyncMock(return_value=response)
    monkeypatch.setattr(app, "http_request", request)
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    await app._materialize_url("https://example.com/image.jpg", output)
    assert output.read_bytes() == b"downloaded"
    request.assert_awaited_once_with(method="GET", url="https://example.com/image.jpg")
    assert app._suffix_for_url("data:image/png;base64,AA==", ".jpg") == ".png"
    assert app._suffix_for_url("https://example.com/image", ".jpg") == ".jpg"
    assert app._resolve_media_path(str(output), str(tmp_path), "image") == str(output)
    assert app._resolve_media_path(output.as_uri(), str(tmp_path), "image") == output.as_uri()
    with pytest.raises(FileNotFoundError):
        app._resolve_media_path("missing", str(tmp_path), "image")


@pytest.mark.parametrize("videos", [1, 2])
async def test_native_frame_extraction_preserves_video_groups_and_timing(tmp_path, monkeypatch, videos):
    module = ModuleType("spatial_agent.evals.base")

    def extract(path, directory, fps, resize):
        assert fps == 2 and resize is None
        index = int(Path(path).stem[-1])
        return [f"frame-{index}"], [index * 2], 2.0, 20

    module.extract_video_frames = extract
    monkeypatch.setitem(sys.modules, "spatial_agent.evals.base", module)
    files = [tmp_path / f"video-{index}.mp4" for index in range(videos)]
    for path in files:
        path.write_bytes(b"video")
    runner = agent(frame_cache_root=str(tmp_path / "cache"))
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="question", metadata={"video_paths": json.dumps([path.as_uri() for path in files])}
    )
    instruction, images, metadata = await runner._materialize_inputs(
        body, tmp_path / "work", SimpleNamespace(video_max_fps=2, video_frame_resize_short_edge=None)
    )
    assert images == [f"frame-{index}" for index in range(videos)]
    assert metadata["video_sources_per_video"] == [str(path) for path in files]
    if videos == 1:
        assert (metadata["fps"], metadata["duration_sec"], metadata["frame_indices"]) == (2.0, 10.0, [0])
    else:
        assert metadata["image_groups"] == [["frame-0"], ["frame-1"]]
        assert metadata["frame_indices_groups"] == [[0], [2]]
        assert metadata["duration_per_video"] == [10.0, 10.0]
    module.extract_video_frames = lambda *args: ([], [], 2, 20)
    with pytest.raises(RuntimeError, match="no frames"):
        await runner._materialize_inputs(
            body, tmp_path / "empty", SimpleNamespace(video_max_fps=2, video_frame_resize_short_edge=None)
        )


async def test_image_paths_and_input_images_keep_supplied_timing(tmp_path):
    image = tmp_path / "image.png"
    image.touch()
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "question"},
                    {"type": "input_image", "image_url": image.as_uri(), "detail": "auto"},
                ],
            }
        ],
        metadata={"image_paths": json.dumps([str(image)]), "fps": "2", "video_source": "source.mp4"},
    )
    _, images, metadata = await agent()._materialize_inputs(body, tmp_path / "work", SimpleNamespace())
    assert images == [str(image), str(image)]
    assert metadata["fps"] == "2" and metadata["video_source"] == "source.mp4"
    body.metadata = {"image_groups": json.dumps([[str(image)]])}
    with pytest.raises(ValueError, match="input image blocks"):
        await agent()._materialize_inputs(body, tmp_path / "work", SimpleNamespace())


def test_config_applies_native_roles_and_enforces_output_budget(tmp_path, monkeypatch):
    module = ModuleType("spatial_agent.config")

    class Config:
        def __init__(self):
            self.video_max_fps = 1
            for role in [
                "main_params",
                "planning_params",
                "general_params",
                "vlm_params",
                "vlm_grounding_params",
                "reflection_params",
            ]:
                setattr(self, role, SimpleNamespace(temperature=1, top_p=1, max_tokens=32000))

        def _load_from_envs(self):
            pass

        def update_from_dataset_json(self, path):
            self.dataset = path

        def update_from_model_json(self, path):
            self.model_path = path

    module.SpatialAgentConfig = Config
    monkeypatch.setitem(sys.modules, "spatial_agent.config", module)
    (tmp_path / "dataset.json").write_text("{}")
    (tmp_path / "model.json").write_text("{}")
    runner = agent(
        model_config_path=str(tmp_path / "model.json"),
        config_overrides={"video_max_fps": 2, "unused": None},
        max_output_tokens=16384,
    )
    runner.config.dataset_config = str(tmp_path / "dataset.json")
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="question", temperature=0.6, top_p=0.95, max_output_tokens=8192
    )
    cfg = runner._build_spatialclaw_config(tmp_path, tmp_path / "work", body)
    assert cfg.llm_base_url == "http://localhost:99/v1"
    assert cfg.video_max_fps == 2 and cfg.concurrency == 1 and not cfg.generate_report
    assert cfg.dataset == str(tmp_path / "dataset.json") and cfg.model_path == str(tmp_path / "model.json")
    for name in [
        "main_params",
        "planning_params",
        "general_params",
        "vlm_params",
        "vlm_grounding_params",
        "reflection_params",
    ]:
        assert vars(getattr(cfg, name)) == {"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192}
    runner.config.config_overrides = {"typo_setting": 4}
    with pytest.raises(ValueError, match="Unknown SpatialClaw"):
        runner._build_spatialclaw_config(tmp_path, tmp_path / "work", body)
    assert app._config_path(tmp_path, None, "model") is None
    with pytest.raises(FileNotFoundError):
        app._config_path(tmp_path, "missing", "model")
