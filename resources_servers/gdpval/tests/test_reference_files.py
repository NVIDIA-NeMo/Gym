# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

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
