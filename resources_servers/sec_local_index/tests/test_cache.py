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
"""Cache files and directories stay usable by other users sharing the cache."""

from __future__ import annotations

import os
import stat

import pytest

from resources_servers.sec_local_index.cache import (
    SHARED_DIR_MODE,
    SHARED_FILE_MODE,
    ToolCache,
    atomic_write,
    make_shared_dirs,
)


@pytest.fixture
def restrictive_umask():
    previous = os.umask(0o077)
    yield
    os.umask(previous)


def test_atomic_write_creates_group_usable_files_and_dirs(tmp_path, restrictive_umask) -> None:
    path = tmp_path / "filings" / "0000320193" / "000032019324000123" / "doc.htm.txt"

    atomic_write(path, "body")

    assert path.read_text() == "body"
    assert stat.S_IMODE(path.stat().st_mode) == SHARED_FILE_MODE
    for directory in (path.parent, path.parent.parent, path.parent.parent.parent):
        assert stat.S_IMODE(directory.stat().st_mode) == SHARED_DIR_MODE
    assert list(path.parent.iterdir()) == [path]


def test_overwrite_keeps_shared_mode(tmp_path, restrictive_umask) -> None:
    path = tmp_path / "doc.txt"
    atomic_write(path, "old")
    atomic_write(path, "new")

    assert path.read_text() == "new"
    assert stat.S_IMODE(path.stat().st_mode) == SHARED_FILE_MODE


def test_existing_dirs_are_left_alone(tmp_path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir(mode=0o700)
    os.chmod(existing, 0o700)

    make_shared_dirs(existing / "new")

    assert stat.S_IMODE(existing.stat().st_mode) == 0o700
    assert stat.S_IMODE((existing / "new").stat().st_mode) == SHARED_DIR_MODE


def test_tool_cache_writes_use_shared_mode(tmp_path, restrictive_umask) -> None:
    cache = ToolCache(tmp_path)
    path = cache.path("filings", "0000320193", "doc.htm.txt")

    cache.write_text(path, "body")

    assert stat.S_IMODE(path.stat().st_mode) == SHARED_FILE_MODE
    assert stat.S_IMODE(path.parent.stat().st_mode) == SHARED_DIR_MODE
