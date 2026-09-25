# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import pytest

from nemo_gym.jsonl_io import compress_jsonl, open_jsonl, zstd


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.zst"])
@pytest.mark.parametrize("binary", [False, True])
def test_streaming_roundtrip_append_flush(tmp_path, suffix, binary):
    path = tmp_path / ("records" + suffix)
    first = '{"text":"café 😀"}\n'
    second = '{"unknown":{"nested":[1,null]}}\n'
    mode = "b" if binary else "t"
    with open_jsonl(path, "w" + mode) as writer:
        writer.write(first.encode() if binary else first)
        writer.flush()
        os.fsync(writer.fileno())
        # A reader sees a complete frame even while the writer stays open.
        with open_jsonl(path, "rt") as reader:
            assert list(reader) == [first]
        writer.write(second.encode() if binary else second)
        writer.flush()
    with open_jsonl(path, "a" + mode) as writer:
        writer.write(first.encode() if binary else first)
    with open_jsonl(path, "rt") as reader:
        assert list(reader) == [first, second, first]


def test_compression_verifies_bytes_and_is_restartable(tmp_path):
    path = tmp_path / "records.jsonl"
    original = b' {"x": 1} \n\n' * 1000
    path.write_bytes(original)
    destination = compress_jsonl(path, remove_source=False)
    assert path.read_bytes() == original
    assert destination.stat().st_size < len(original) / 10
    # Simulate interruption after destination publication but before source removal.
    assert compress_jsonl(path) == destination
    assert not path.exists()
    with open_jsonl(destination, "rb") as reader:
        assert reader.read() == original
    assert list(tmp_path.glob(".*.zst")) == []


def test_compression_never_overwrites_conflicting_destination(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_bytes(b"new\n")
    destination = tmp_path / "records.jsonl.zst"
    with open_jsonl(destination, "wb") as writer:
        writer.write(b"old\n")
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        compress_jsonl(path)
    assert path.read_bytes() == b"new\n"
    assert destination.read_bytes() == before


def test_failed_publication_keeps_source_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"x":1}\n')

    def fail(*args):
        raise OSError("publication failed")

    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(OSError, match="publication failed"):
        compress_jsonl(path)
    assert path.read_bytes() == b'{"x":1}\n'
    assert list(tmp_path.iterdir()) == [path]


def test_truncated_frame_raises_instead_of_silent_eof(tmp_path):
    path = tmp_path / "records.jsonl.zst"
    path.write_bytes(zstd.compress(b'{"x":1}\n')[:-1])
    with open_jsonl(path, "rb") as reader, pytest.raises(EOFError):
        list(reader)


def test_empty_frame_and_invalid_mode(tmp_path):
    path = tmp_path / "empty.jsonl.zst"
    with open_jsonl(path, "wb"):
        pass
    with open_jsonl(path, "rb") as reader:
        assert reader.read() == b""
    with pytest.raises(ValueError, match="Unsupported JSONL mode"):
        open_jsonl(path, "r+")
    with pytest.raises(ValueError, match="uncompressed JSONL"):
        compress_jsonl(path)


def test_failed_byte_verification_keeps_original(tmp_path, monkeypatch):
    import nemo_gym.jsonl_io as jsonl_io

    path = tmp_path / "records.jsonl"
    original = b'{"evidence":"keep"}\n'
    path.write_bytes(original)
    monkeypatch.setattr(jsonl_io, "same_jsonl_bytes", lambda *args: False)
    with pytest.raises(OSError, match="verification failed"):
        compress_jsonl(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("source_mode", [0o640, 0o644])
def test_compression_preserves_shared_reader_permissions(tmp_path, source_mode):
    path = tmp_path / "records.jsonl"
    original = b'{"evidence":"shared"}\n'
    path.write_bytes(original)
    path.chmod(source_mode)

    destination = compress_jsonl(path)

    assert not path.exists()
    assert destination.stat().st_mode & 0o7777 == source_mode
    with open_jsonl(destination, "rb") as reader:
        assert reader.read() == original
