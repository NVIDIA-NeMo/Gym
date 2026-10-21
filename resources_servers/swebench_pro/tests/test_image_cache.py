# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from resources_servers.swebench_pro.app import SWEBenchProInstanceRequest
from resources_servers.swebench_pro.image_cache import prepare_image, verify_local_image
from resources_servers.swebench_pro.tests.test_app import make_server, request_body


def test_cache_reuse_requires_original_source_and_unchanged_sif(tmp_path, monkeypatch):
    calls = []

    def pull(argv, **kwargs):
        calls.append(argv)
        Path(argv[-2]).write_bytes(b"converted image")

    monkeypatch.setattr("resources_servers.swebench_pro.image_cache.subprocess.run", pull)
    digest = "sha256:" + "a" * 64
    info = prepare_image(tmp_path, "docker.io/example/images", digest)
    assert calls[0][-1] == f"docker://docker.io/example/images@{digest}"
    assert prepare_image(tmp_path, "docker.io/example/images", digest) == info
    assert len(calls) == 1
    path = Path(info["image"])
    with pytest.raises(ValueError, match="source does not match"):
        verify_local_image(path, "docker://docker.io/example/images@sha256:" + "b" * 64)
    path.write_bytes(b"stale or replaced image")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_image(tmp_path, "docker.io/example/images", digest)
    assert len(calls) == 1  # Never silently accept or overwrite an untrusted cache entry.


def test_pro_server_rejects_local_image_without_provenance(tmp_path):
    path = tmp_path / "image.sif"
    path.write_bytes(b"untracked cache")
    body = SWEBenchProInstanceRequest.model_validate(request_body() | {"image_digest": "sha256:" + "a" * 64})
    server = make_server(golden=False, image_template=str(path))
    with pytest.raises(ValueError, match="Missing or invalid image provenance"):
        server._image_info(body)
    path.with_suffix(".sif.json").write_text(json.dumps({"source_uri": "wrong source", "sif_sha256": "unused"}))
    with pytest.raises(ValueError, match="source does not match"):
        server._image_info(body)


def test_concurrent_preparations_share_one_pull(tmp_path, monkeypatch):
    started, release = Event(), Event()
    calls = []

    def pull(argv, **kwargs):
        calls.append(argv)
        started.set()
        assert release.wait(5)
        Path(argv[-2]).write_bytes(b"converted image")

    monkeypatch.setattr("resources_servers.swebench_pro.image_cache.subprocess.run", pull)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(prepare_image, tmp_path, "example/images", "sha256:" + "a" * 64)
        assert started.wait(5)
        second = pool.submit(prepare_image, tmp_path, "example/images", "sha256:" + "a" * 64)
        release.set()
        assert first.result() == second.result()
    assert len(calls) == 1
