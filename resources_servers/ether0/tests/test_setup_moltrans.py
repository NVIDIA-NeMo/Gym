# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
from urllib.error import URLError

import pytest

from resources_servers.ether0 import setup_moltrans


@pytest.fixture
def download(monkeypatch, tmp_path):
    content = b"pinned checkpoint test payload"
    monkeypatch.delenv(setup_moltrans.MOLTRANS_PATH_ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(setup_moltrans, "MOLTRANS_SHA256", hashlib.sha256(content).hexdigest())
    fetch = MagicMock(side_effect=lambda *args, **kwargs: io.BytesIO(content))
    monkeypatch.setattr(setup_moltrans.urllib.request, "urlopen", fetch)
    return content, fetch


def test_first_use_downloads_and_second_use_is_offline(download, tmp_path):
    content, fetch = download
    path = setup_moltrans.ensure_moltrans()
    assert path == tmp_path / "nemo-gym" / "ether0" / setup_moltrans.MOLTRANS_FILENAME
    assert path.read_bytes() == content
    fetch.side_effect = AssertionError("cached startup must not use the network")
    assert setup_moltrans.ensure_moltrans() == path
    assert fetch.call_count == 1


def test_configured_path_overrides_environment(download, monkeypatch, tmp_path):
    content, _ = download
    env_path = tmp_path / "env" / "model.pt"
    monkeypatch.setenv(setup_moltrans.MOLTRANS_PATH_ENV_VAR, str(env_path))
    explicit = tmp_path / "configured" / "model.pt"
    assert setup_moltrans.ensure_moltrans(str(explicit)) == explicit
    assert explicit.read_bytes() == content
    assert not env_path.exists()


def test_environment_path_downloads_when_missing(download, monkeypatch, tmp_path):
    content, _ = download
    path = tmp_path / "shared" / "model.pt"
    monkeypatch.setenv(setup_moltrans.MOLTRANS_PATH_ENV_VAR, str(path))
    assert setup_moltrans.ensure_moltrans() == path
    assert path.read_bytes() == content


def test_default_cache_uses_home(download, monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setattr(setup_moltrans.Path, "home", lambda: tmp_path)
    assert setup_moltrans.ensure_moltrans() == (
        tmp_path / ".cache" / "nemo-gym" / "ether0" / setup_moltrans.MOLTRANS_FILENAME
    )


def test_existing_corrupt_file_is_rejected_without_overwriting(download, tmp_path):
    _, fetch = download
    path = tmp_path / "model.pt"
    path.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="not the pinned MolTrans checkpoint"):
        setup_moltrans.ensure_moltrans(str(path))
    assert path.read_bytes() == b"wrong checkpoint"
    fetch.assert_not_called()


def test_bad_download_is_not_published_and_can_be_retried(download, tmp_path):
    content, fetch = download
    path = tmp_path / "model.pt"
    fetch.side_effect = lambda *args, **kwargs: io.BytesIO(b"HTML error page")
    with pytest.raises(ValueError, match="not the pinned MolTrans checkpoint"):
        setup_moltrans.ensure_moltrans(str(path))
    assert not path.exists()
    assert not list(tmp_path.glob("*.partial"))
    fetch.side_effect = lambda *args, **kwargs: io.BytesIO(content)
    assert setup_moltrans.ensure_moltrans(str(path)).read_bytes() == content


def test_interrupted_download_cleans_up_partial_file(download, tmp_path):
    _, fetch = download
    path = tmp_path / "model.pt"

    class InterruptedResponse(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise URLError("connection lost")
            return super().read(4)

    fetch.side_effect = lambda *args, **kwargs: InterruptedResponse(b"some partial data")
    with pytest.raises(RuntimeError, match="For offline startup"):
        setup_moltrans.ensure_moltrans(str(path))
    assert not path.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_concurrent_workers_share_one_download(download, tmp_path):
    content, fetch = download
    barrier = threading.Barrier(4)
    path = tmp_path / "shared" / "model.pt"

    def start(_):
        barrier.wait(timeout=10)
        return setup_moltrans.ensure_moltrans(str(path))

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(start, range(4))) == [path] * 4
    assert path.read_bytes() == content
    assert fetch.call_count == 1
