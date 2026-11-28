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
"""Install a local Lean 4 + Mathlib workspace for checking Verina solutions.

The workspace holds elan (the Lean toolchain manager), the pinned Lean toolchain, and a Lake project that depends on
Mathlib. Mathlib's prebuilt .olean files come from its cache (`lake exe cache get`), so nothing is compiled from
source. The install is idempotent: a sentinel file marks a finished workspace.
"""

import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path


# Verina's release pins Lean v4.18.0 and the Mathlib tag of the same name.
LEAN_TOOLCHAIN = "leanprover/lean4:v4.18.0"
MATHLIB_REV = "v4.18.0"
ELAN_INIT_URL = "https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh"

_LAKEFILE = f"""name = "verina_workspace"

[[require]]
name = "mathlib"
git = "https://github.com/leanprover-community/mathlib4.git"
rev = "{MATHLIB_REV}"
"""

_SMOKE_TEST = "import Mathlib\n\nexample : (2 : ℕ) + 2 = 4 := by norm_num\n"


def lean_env(workspace: Path) -> dict[str, str]:
    """Return a process environment whose PATH finds this workspace's elan, lean, and lake."""
    env = dict(os.environ)
    elan_home = workspace / "elan"
    env["ELAN_HOME"] = str(elan_home)
    env["PATH"] = f"{elan_home / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    return env


def project_dir(workspace: Path) -> Path:
    """Return the Lake project directory inside the workspace."""
    return workspace / "project"


def lean_search_path(workspace: Path) -> str:
    """Return the LEAN_PATH that lets `lean` find Mathlib's .olean files without going through `lake env`."""
    env = lean_env(workspace)
    lake = shutil.which("lake", path=env["PATH"])
    if lake is None:
        raise FileNotFoundError(f"no lake under {workspace / 'elan'}; run ensure_lean_workspace first")
    result = subprocess.run(
        [lake, "env", "printenv", "LEAN_PATH"],
        cwd=project_dir(workspace),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def ensure_lean_workspace(workspace: Path) -> Path:
    """Create the Lean workspace if needed and return its Lake project directory.

    Raises subprocess.CalledProcessError when a download or the smoke test fails.
    """
    workspace = Path(workspace).resolve()
    project = project_dir(workspace)
    sentinel = workspace / ".installed"
    if sentinel.exists():
        return project

    workspace.mkdir(parents=True, exist_ok=True)
    env = lean_env(workspace)

    if not (workspace / "elan" / "bin" / "elan").exists():
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "elan-init.sh"
            urllib.request.urlretrieve(ELAN_INIT_URL, script)
            subprocess.check_call(
                ["sh", str(script), "-y", "--no-modify-path", "--default-toolchain", "none"], env=env
            )

    project.mkdir(exist_ok=True)
    (project / "lean-toolchain").write_text(LEAN_TOOLCHAIN + "\n")
    (project / "lakefile.toml").write_text(_LAKEFILE)
    lake = shutil.which("lake", path=env["PATH"])
    if lake is None:
        raise FileNotFoundError(f"elan did not install lake under {workspace / 'elan'}")
    subprocess.check_call([lake, "update"], cwd=project, env=env)
    subprocess.check_call([lake, "exe", "cache", "get"], cwd=project, env=env)

    smoke = project / "Smoke.lean"
    smoke.write_text(_SMOKE_TEST)
    try:
        subprocess.check_call([lake, "env", "lean", str(smoke)], cwd=project, env=env)
    finally:
        smoke.unlink()

    sentinel.touch()
    return project


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).parent / ".lean_workspace")
    args = parser.parse_args()
    print(f"Lean workspace ready at {ensure_lean_workspace(args.workspace)}")
