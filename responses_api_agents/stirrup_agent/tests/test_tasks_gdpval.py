# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path

from responses_api_agents.stirrup_agent.tasks.gdpval import _download_reference_files


class TestDownloadReferenceFilesLocal:
    """The local-path / ``file://`` branch of ``_download_reference_files``."""

    def test_absolute_path_is_copied(self, tmp_path: Path) -> None:
        src = tmp_path / "src" / "input.xlsx"
        src.parent.mkdir()
        src.write_bytes(b"hello")

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        downloaded = _download_reference_files(["input.xlsx"], [str(src)], dest_dir)

        assert downloaded == ["input.xlsx"]
        assert (dest_dir / "input.xlsx").read_bytes() == b"hello"

    def test_file_url_is_copied(self, tmp_path: Path) -> None:
        src = tmp_path / "src" / "input.txt"
        src.parent.mkdir()
        src.write_bytes(b"world")

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        downloaded = _download_reference_files(["input.txt"], [f"file://{src}"], dest_dir)

        assert downloaded == ["input.txt"]
        assert (dest_dir / "input.txt").read_bytes() == b"world"

    def test_missing_local_file_does_not_raise(self, tmp_path: Path) -> None:
        """A bad local path should be logged but not abort the whole batch."""
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        downloaded = _download_reference_files(["missing.bin"], [str(tmp_path / "does-not-exist.bin")], dest_dir)

        assert downloaded == []
        assert not (dest_dir / "missing.bin").exists()

    def test_nested_dest_path_is_created(self, tmp_path: Path) -> None:
        """``reference_files`` may contain nested paths; parent dirs are created."""
        src = tmp_path / "src.bin"
        src.write_bytes(b"data")

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        downloaded = _download_reference_files(["sub/dir/file.bin"], [str(src)], dest_dir)

        assert downloaded == ["sub/dir/file.bin"]
        assert (dest_dir / "sub" / "dir" / "file.bin").read_bytes() == b"data"
