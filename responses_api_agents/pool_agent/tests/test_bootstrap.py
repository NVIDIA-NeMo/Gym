# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the POSIX installer with isolated commands; never install packages or access the network."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("manager", ["apt-get", "apk"])
@pytest.mark.parametrize("missing", [(), ("bash",), ("python3", "bash")])
@pytest.mark.parametrize("failure", [None, "nonroot", "package_install", "download"])
def test_bootstrap_without_bash(tmp_path, manager, missing, failure):
    bindir = tmp_path / "bin"
    bindir.mkdir()

    def executable(name, source):
        path = bindir / name
        path.write_text("#!/bin/sh\nset -eu\n" + source)
        path.chmod(0o755)

    for name in ("mkdir", "cat", "chmod", "tar", "sh", "bash"):
        if name not in missing:
            source = shutil.which(name)
            if source is None:
                pytest.skip(f"Bootstrap test requires {name}")
            (bindir / name).symlink_to(source)
    if "python3" not in missing:
        (bindir / "python3").symlink_to(sys.executable)
    executable("id", f"echo {1000 if failure == 'nonroot' else 0}\n")
    executable(
        manager,
        'printf "%s\\n" "$*" >> "$TEST_ROOT/packages.log"\n'
        + ("exit 100\n" if failure == "package_install" else "")
        + (f"/bin/ln -sf {sys.executable} {bindir / 'python3'}\n" if "python3" in missing else ""),
    )
    executable(
        "curl",
        ("exit 22\n" if failure == "download" else "")
        + """while [ "$1" != "-o" ]; do shift; done
shift
cat > "$1" <<'INSTALL'
#!/bin/sh
cat > "$POOL_INSTALL_DIR/pool" <<'BINARY'
#!/bin/sh
echo 1.0.16
BINARY
chmod +x "$POOL_INSTALL_DIR/pool"
INSTALL
""",
    )
    root = tmp_path / "session with spaces"
    script = Path(__file__).parents[1] / "install_pool.sh"
    result = subprocess.run(
        ["/bin/sh", str(script), str(root), "v1.0.16"],
        env=os.environ | {"PATH": str(bindir), "TEST_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    if missing and failure == "nonroot":
        expected = 1
    elif missing and failure == "package_install":
        expected = 100
    elif failure == "download":
        expected = 22
    else:
        expected = 0
    assert result.returncode == expected, result.stderr
    if expected:
        assert not (root / "bin/pool").exists()
    else:
        assert (root / "bin/pool").exists()
    packages = tmp_path / "packages.log"
    if missing and failure != "nonroot":
        # Dependencies are ordered by the installer, independently of the fixture order.
        names = " ".join(name for name in ("python3", "bash") if name in missing)
        expected_commands = (
            [f"add --no-cache {names}"]
            if manager == "apk"
            else ["update", f"install -y --no-install-recommends {names}"]
        )
        if failure == "package_install":
            expected_commands = expected_commands[:1]
        assert packages.read_text().splitlines() == expected_commands
    else:
        assert not packages.exists()


def test_old_python_fails_before_downloading_pool(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("mkdir", "cat", "chmod", "tar", "sh", "bash"):
        source = shutil.which(name)
        if source is None:
            pytest.skip(f"Bootstrap test requires {name}")
        (bindir / name).symlink_to(source)
    curl = bindir / "curl"
    curl.write_text("#!/bin/sh\necho unexpected download >&2\nexit 42\n")
    curl.chmod(0o755)
    python = bindir / "python3"
    python.write_text(f"#!{sys.executable}\nimport sys\nsys.version_info = (3, 7, 0)\nexec(sys.argv[2])\n")
    python.chmod(0o755)
    root = tmp_path / "session"
    result = subprocess.run(
        ["/bin/sh", str(Path(__file__).parents[1] / "install_pool.sh"), str(root), "v1.0.16"],
        env=os.environ | {"PATH": str(bindir)},
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    assert result.returncode != 0
    assert "Pool requires Python >=3.8" in result.stderr
    assert not root.exists()
