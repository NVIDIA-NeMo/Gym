# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from resources_servers.mimo_music import setup_abc2midi as setup


def test_existing_tool():
    with patch.object(setup.shutil, "which", return_value="/bin/abc2midi"):
        assert setup.ensure_abc2midi() == "/bin/abc2midi"


def test_cached_install(tmp_path, monkeypatch):
    binary = tmp_path / "bin" / "abc2midi"
    binary.parent.mkdir()
    binary.write_text("cached")
    monkeypatch.setattr(setup, "PREFIX", tmp_path)
    monkeypatch.setattr(setup.shutil, "which", lambda _: None)
    monkeypatch.setenv("PATH", "/usr/bin")
    assert setup.ensure_abc2midi() == str(binary)
    assert setup.os.environ["PATH"].startswith(str(binary.parent))


@pytest.mark.parametrize("found", [True, False])
def test_macos(tmp_path, monkeypatch, found):
    monkeypatch.setattr(setup, "PREFIX", tmp_path)
    monkeypatch.setattr(setup.sys, "platform", "darwin")
    with (
        patch.object(setup.shutil, "which", side_effect=[None, "/bin/abc2midi" if found else None]),
        patch.object(setup.subprocess, "run") as run,
    ):
        if found:
            assert setup.ensure_abc2midi() == "/bin/abc2midi"
        else:
            with pytest.raises(RuntimeError, match="not on PATH"):
                setup.ensure_abc2midi()
        run.assert_called_once_with(["brew", "install", "abcmidi"], check=True, timeout=600)


@pytest.mark.parametrize(
    "platform,tool,error",
    [("win32", None, "supports Linux"), ("linux", "make", "Install make"), ("linux", "cc", "Install cc")],
)
def test_install_prerequisites(tmp_path, monkeypatch, platform, tool, error):
    monkeypatch.setattr(setup, "PREFIX", tmp_path)
    monkeypatch.setattr(setup.sys, "platform", platform)
    monkeypatch.setattr(setup.shutil, "which", lambda name: None if name in ("abc2midi", tool) else "/usr/bin/" + name)
    with pytest.raises(RuntimeError, match=error):
        setup.ensure_abc2midi()


@pytest.mark.parametrize("valid_checksum", [False, True])
def test_linux_install(tmp_path, monkeypatch, valid_checksum):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        info = tarfile.TarInfo(f"abcmidi-{setup.REVISION}/abc2midi")
        info.size = 4
        info.mode = 0o755
        archive.addfile(info, io.BytesIO(b"test"))
    data = buf.getvalue()
    monkeypatch.setattr(setup, "PREFIX", tmp_path)
    monkeypatch.setattr(setup.sys, "platform", "linux")
    monkeypatch.setattr(setup.shutil, "which", lambda name: None if name == "abc2midi" else "/usr/bin/" + name)
    monkeypatch.setattr(setup, "SHA256", hashlib.sha256(data).hexdigest() if valid_checksum else "incorrect")
    with patch.object(setup, "urlopen", return_value=io.BytesIO(data)), patch.object(setup.subprocess, "run") as run:
        if valid_checksum:
            binary = Path(setup.ensure_abc2midi())
            assert binary.read_bytes() == b"test"
            assert run.call_args_list[0].args[0] == ["make", "CC=cc", "abc2midi"]
            assert run.call_args_list[1].args[0][-1] == "-ver"
            assert not list(tmp_path.glob("tmp*"))
        else:
            with pytest.raises(RuntimeError, match="checksum mismatch"):
                setup.ensure_abc2midi()
            run.assert_not_called()
