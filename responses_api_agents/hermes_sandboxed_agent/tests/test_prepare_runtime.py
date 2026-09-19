# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("bash", "git", "flock")), reason="Requires Linux build tools"
)
def test_requested_version_controls_checkout_and_cache(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    git("init", "-q")
    commits = []
    for version in ("first", "second"):
        (source / "version").write_text(version)
        git("add", "version")
        git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", version)
        commits.append(git("rev-parse", "HEAD"))
    git("tag", "experiment", commits[1])

    scripts = tmp_path / "scripts" / "hermes_sandboxed_agent"
    scripts.mkdir(parents=True)
    for name in ("prepare_runtime.sh", "runtime_python.sh"):
        shutil.copyfile(Path(__file__).parents[1] / name, scripts / name)
    helper = scripts.parent / "anyswe_agent" / "setup_scripts" / "_portable_python.sh"
    helper.parent.mkdir(parents=True)
    # Exercise real Git selection and cache decisions without downloading Python or dependencies.
    helper.write_text("""
PYTHON_VERSION=3.13.14
ARCH=x86_64-unknown-linux-gnu
portable_python_can_run() { return 0; }
install_portable_python() {
    printf '#!/bin/sh\nexit 0\n' > "$DEPS_DIR/bin/python3"
    chmod +x "$DEPS_DIR/bin/python3"
}
install_python_packages() {
    local site="$DEPS_DIR/lib/python3.13/site-packages"
    mkdir -p "$site"
    printf '%s\n' "$DEPS_DIR/hermes-src" > "$site/__editable__hermes.pth"
    echo installed >> "$DEPS_DIR/builds"
}
""")
    runtime = tmp_path / "runtime"
    ripgrep = runtime / "tools" / "bin" / "rg"
    ripgrep.parent.mkdir(parents=True)
    ripgrep.write_text("#!/bin/sh\nexit 0\n")
    ripgrep.chmod(0o755)
    env = os.environ | {"DEPS_DIR": str(runtime), "HERMES_REPO_URL": str(source)}
    for requested, expected, builds in (
        (commits[0], commits[0], 1),
        ("experiment", commits[1], 2),
        ("experiment", commits[1], 2),
    ):
        result = subprocess.run(
            ["bash", str(scripts / "prepare_runtime.sh")],
            env=env | {"HERMES_VERSION": requested},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads((runtime / "hermes-runtime.json").read_text())["hermes_commit"] == expected
        actual = subprocess.check_output(["git", "-C", str(runtime / "hermes-src"), "rev-parse", "HEAD"], text=True)
        assert actual.strip() == expected
        assert len((runtime / "builds").read_text().splitlines()) == builds
