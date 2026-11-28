# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MUCS scoring maps are downloaded from the pinned archive on first use and verified.

The scorer that applies them is tested in ``resources_servers/audio_eval/tests/test_mucs_assets.py``.
"""

import hashlib
import io
import tarfile

import pytest

from resources_servers.audio_eval import mucs_assets


FILES = {
    "punct.mapping.txt": b"slash /\n",
    "hin2engwords.txt": b"\xe0\xa4\x95\xe0\xa4\xae cam\nx y\nx z\n",
    "ben2engwords.txt": b"b ben\n",
}


def archive(files, directory="subtask2_blindtest_wReadme/inference"):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        decoy = tarfile.TarInfo("subtask2_blindtest_wReadme/audio/a.wav")
        decoy.size = 3
        tar.addfile(decoy, io.BytesIO(b"abc"))
        for name, data in files.items():
            info = tarfile.TarInfo(f"{directory}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture
def downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(mucs_assets, "ASSET_SHA256", {n: hashlib.sha256(d).hexdigest() for n, d in FILES.items()})
    monkeypatch.setattr(mucs_assets, "assets_dir", lambda: tmp_path / "cache")
    mucs_assets.load_official_mapping.cache_clear()
    calls = []

    def serve(payload):
        def urlopen(request, timeout):
            calls.append(request.full_url)
            return io.BytesIO(payload)

        monkeypatch.setattr(mucs_assets.urllib.request, "urlopen", urlopen)

    serve(archive(FILES))
    yield calls, serve, tmp_path / "cache"
    mucs_assets.load_official_mapping.cache_clear()


def test_assets_are_downloaded_once_and_cached(downloads):
    calls, _, cache = downloads
    assert mucs_assets.load_official_mapping("hin2engwords.txt") == {"कम": "cam", "x": "y"}
    assert sorted(p.name for p in cache.iterdir()) == sorted(FILES)
    assert calls == [mucs_assets.ARCHIVE_URL]
    mucs_assets.fetch_assets(cache)
    assert len(calls) == 1


def test_a_corrupted_cached_file_is_fetched_again(downloads):
    calls, _, cache = downloads
    mucs_assets.fetch_assets(cache)
    (cache / "punct.mapping.txt").write_bytes(b"tampered\n")
    assert mucs_assets.load_official_mapping("punct.mapping.txt") == {"slash": "/"}
    assert len(calls) == 2


def test_a_member_with_the_wrong_digest_is_rejected(downloads):
    _, serve, cache = downloads
    serve(archive(FILES | {"punct.mapping.txt": b"evil /\n"}))
    with pytest.raises(ValueError, match="pinned SHA-256"):
        mucs_assets.fetch_assets(cache)


def test_missing_members_and_unknown_assets_are_errors(downloads):
    _, serve, cache = downloads
    serve(archive({"punct.mapping.txt": FILES["punct.mapping.txt"]}))
    with pytest.raises(ValueError, match="not found"):
        mucs_assets.fetch_assets(cache)
    with pytest.raises(ValueError, match="Unknown MUCS scoring asset"):
        mucs_assets.load_official_mapping("other.txt")


def test_blank_mapping_rows_are_rejected(tmp_path):
    path = tmp_path / "map.txt"
    path.write_text("a b\n\n")
    with pytest.raises(ValueError, match="Blank mapping row"):
        mucs_assets.read_mapping(path)


def test_assets_dir_is_under_the_gym_cache():
    assert mucs_assets.assets_dir().parts[-2:] == ("audio_eval", "mucs2021")
