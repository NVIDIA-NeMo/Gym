# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from resources_servers.gdpval.reference_files import download_reference_files


class TestDownloadReferenceFilesLocal:
    """The local-path / ``file://`` branch of ``download_reference_files``."""

    def test_absolute_path_is_copied(self, tmp_path: Path) -> None:
        src = tmp_path / "src" / "input.xlsx"
        src.parent.mkdir()
        src.write_bytes(b"hello")

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        downloaded = download_reference_files(["input.xlsx"], [str(src)], dest_dir)

        assert downloaded == ["input.xlsx"]
        assert (dest_dir / "input.xlsx").read_bytes() == b"hello"

    def test_file_url_is_copied(self, tmp_path: Path) -> None:
        src = tmp_path / "src" / "input.txt"
        src.parent.mkdir()
        src.write_bytes(b"world")

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        downloaded = download_reference_files(["input.txt"], [f"file://{src}"], dest_dir)

        assert downloaded == ["input.txt"]
        assert (dest_dir / "input.txt").read_bytes() == b"world"

    def test_missing_local_file_does_not_raise(self, tmp_path: Path) -> None:
        """A bad local path should be logged but not abort the whole batch."""
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        downloaded = download_reference_files(["missing.bin"], [str(tmp_path / "does-not-exist.bin")], dest_dir)

        assert downloaded == []
        assert not (dest_dir / "missing.bin").exists()

    def test_nested_dest_path_is_created(self, tmp_path: Path) -> None:
        """``reference_files`` may contain nested paths; parent dirs are created."""
        src = tmp_path / "src.bin"
        src.write_bytes(b"data")

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        downloaded = download_reference_files(["sub/dir/file.bin"], [str(src)], dest_dir)

        assert downloaded == ["sub/dir/file.bin"]
        assert (dest_dir / "sub" / "dir" / "file.bin").read_bytes() == b"data"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://hf.test/f", code, "error", {}, None)


@pytest.fixture
def network(monkeypatch):
    """Patch the network boundary: queued outcomes per urlopen call, recorded requests and sleeps."""
    outcomes: list = []
    requests: list[urllib.request.Request] = []
    sleeps: list[float] = []

    def urlopen(req, timeout=None):
        requests.append(req)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    return SimpleNamespace(outcomes=outcomes, requests=requests, sleeps=sleeps)


class TestDownloadReferenceFilesHttp:
    def test_the_hf_token_is_sent_as_a_bearer_token(self, tmp_path, network, monkeypatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_secret")
        network.outcomes.append(b"data")

        assert download_reference_files(["f.bin"], ["https://hf.test/f"], tmp_path) == ["f.bin"]

        assert network.requests[0].get_header("Authorization") == "Bearer hf_secret"
        assert (tmp_path / "f.bin").read_bytes() == b"data"

    def test_no_authorization_header_without_a_token(self, tmp_path, network, monkeypatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        network.outcomes.append(b"data")

        download_reference_files(["f.bin"], ["https://hf.test/f"], tmp_path)

        assert network.requests[0].get_header("Authorization") is None

    @pytest.mark.parametrize("code", [429, 500, 503])
    def test_retryable_statuses_are_retried_with_backoff(self, tmp_path, network, code) -> None:
        network.outcomes.extend([_http_error(code), _http_error(code), b"data"])

        assert download_reference_files(["f.bin"], ["https://hf.test/f"], tmp_path) == ["f.bin"]

        assert network.sleeps == [1, 2]
        assert (tmp_path / "f.bin").read_bytes() == b"data"

    def test_a_connection_error_is_retried(self, tmp_path, network) -> None:
        network.outcomes.extend([urllib.error.URLError("reset"), b"data"])

        assert download_reference_files(["f.bin"], ["https://hf.test/f"], tmp_path) == ["f.bin"]

    @pytest.mark.parametrize("code", [403, 404])
    def test_other_statuses_are_not_retried(self, tmp_path, network, code) -> None:
        network.outcomes.append(_http_error(code))

        assert download_reference_files(["f.bin"], ["https://hf.test/f"], tmp_path) == []

        assert len(network.requests) == 1
        assert network.sleeps == []

    def test_exhausted_retries_drop_only_the_failing_file(self, tmp_path, network) -> None:
        network.outcomes.extend([_http_error(429)] * 6 + [b"ok"])

        downloaded = download_reference_files(
            ["bad.bin", "good.bin"], ["https://hf.test/bad", "https://hf.test/good"], tmp_path
        )

        assert downloaded == ["good.bin"]
        assert network.sleeps == [1, 2, 4, 8, 16]

    def test_a_leading_slash_is_stripped(self, tmp_path, network) -> None:
        network.outcomes.append(b"data")

        downloaded = download_reference_files(["/reference_files/abc/f.bin"], ["https://hf.test/f"], tmp_path)

        assert downloaded == ["reference_files/abc/f.bin"]
        assert (tmp_path / "reference_files" / "abc" / "f.bin").read_bytes() == b"data"


class TestMismatchedLists:
    def test_extra_names_without_urls_are_not_downloaded(self, tmp_path) -> None:
        src = tmp_path / "src.bin"
        src.write_bytes(b"data")
        dest = tmp_path / "dest"
        dest.mkdir()

        assert download_reference_files(["a.bin", "b.bin"], [str(src)], dest) == ["a.bin"]
