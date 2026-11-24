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
"""Install the pinned official ARC Prize harness (arc-agi-3-benchmarking, MIT; not vendored) and its own venv.

``official_harness_manifest.json`` pins the commit and the sha256 of every file of the repository at that commit. An
install is ``<dest>/src`` (exactly those files) and ``<dest>/venv`` (built by ``uv sync --frozen --no-dev`` from the
harness's own ``uv.lock``; its openai / anthropic pins conflict with Gym's, so it cannot share the agent's venv).

Sources:

- ``git`` (the default): clone the repository and check out the pinned commit;
- a directory with the repository files (for example a ``git archive`` of the commit), or a ``.tar.gz`` of one (a
  single top-level directory is allowed).

    python responses_api_agents/arc_agi_3_official_agent/setup_official_harness.py [source] [--dest DIR]
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "official_harness_manifest.json"
DEFAULT_DEST = HERE / ".arc3_official_harness"
IGNORED_PARTS = ("__pycache__", ".git", ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache")
IGNORED_SUFFIXES = (".pyc",)


def load_manifest(path: Path | None = None) -> dict:
    return json.loads((path or MANIFEST_PATH).read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_files(root: Path) -> list[str]:
    files = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if path.is_dir() or any(p in IGNORED_PARTS for p in rel.parts) or path.suffix in IGNORED_SUFFIXES:
            continue
        files.append(rel.as_posix())
    return files


def verify_tree(root: Path, manifest: dict | None = None) -> None:
    """Raise unless ``root`` holds exactly the pinned repository files."""
    manifest = manifest or load_manifest()
    pinned = manifest["files"]
    problems = []
    for rel, digest in pinned.items():
        path = root / rel
        if not path.is_file():
            problems.append(f"missing {rel}")
        elif _sha256(path) != digest:
            problems.append(f"sha256 differs: {rel}")
    problems += [f"unpinned file {rel}" for rel in sorted(set(_tree_files(root)) - set(pinned))]
    if problems:
        commit = manifest["provenance"]["commit"]
        raise ValueError(f"{root} is not arc-agi-3-benchmarking @ {commit}: {'; '.join(problems[:10])}")


def venv_python(dest: Path) -> Path:
    return dest / "venv" / "bin" / "python"


def verify_install(dest: Path, manifest: dict | None = None) -> None:
    """Raise unless ``dest`` has the pinned source tree and a venv that imports the harness."""
    verify_tree(dest / "src", manifest)
    python = venv_python(dest)
    if not python.is_file():
        raise FileNotFoundError(f"official harness install {dest} has no venv ({python})")
    probe = subprocess.run(
        [str(python), "-c", "import arcengine, benchmarking.agent, openai"],
        cwd=dest / "src",
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"the official harness venv at {dest} cannot import the harness: {probe.stderr[-2000:]}")


def _single_root(directory: Path) -> Path:
    if (directory / "benchmarking").is_dir():
        return directory
    children = [p for p in directory.iterdir() if p.is_dir()]
    if len(children) == 1 and (children[0] / "benchmarking").is_dir():
        return children[0]
    raise FileNotFoundError(f"{directory} has no benchmarking/ directory (expected the arc-agi-3-benchmarking tree)")


def _stage_source(source: str, workdir: Path, manifest: dict) -> Path:
    if source == "git":
        provenance = manifest["provenance"]
        target = workdir / "git"
        subprocess.run(["git", "clone", "--quiet", provenance["repository"], str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--quiet", provenance["commit"]], check=True)
        head = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if head != provenance["commit"]:
            raise ValueError(f"checked out {head}, not the pinned {provenance['commit']}")
        shutil.rmtree(target / ".git")
        return target
    path = Path(source).expanduser()
    if path.is_dir():
        return _single_root(path)
    if path.is_file() and path.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path) as archive:
            archive.extractall(workdir / "tar", filter="data")
        return _single_root(workdir / "tar")
    raise FileNotFoundError(f"official harness source {source!r} is not 'git', a directory or a tarball")


def _build_venv(dest: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv is needed to build the official harness venv (uv sync --frozen)")
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(dest / "venv"))
    subprocess.run(
        [uv, "sync", "--frozen", "--no-dev", "--python", sys.executable, "--project", str(dest / "src")],
        check=True,
        env=env,
    )


@contextlib.contextmanager
def install_lock(dest: Path) -> Iterator[None]:
    """Hold ``<dest>.lock`` (an exclusive ``flock``): concurrent installs, also from other processes, run one at a time,
    so the next one finds the finished install. The kernel releases the lock if its holder dies."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest.with_name(dest.name + ".lock"), "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def ensure_official_harness(source: str | None, dest: Path = DEFAULT_DEST) -> Path:
    """Return a verified install at ``dest``, installing it from ``source`` (default ``git``) when absent.

    Concurrent calls (several servers on one checkout) share one install: the first installs under ``install_lock``
    and the others then verify its result. The source tree is staged in a directory of its own next to ``dest`` and
    renamed into place; the venv is then built there (a venv records absolute paths), still under the lock. A failed
    install removes what it built, so a later call retries."""
    manifest = load_manifest()
    with install_lock(dest):
        if dest.exists():
            verify_install(dest, manifest)
            return dest
        with tempfile.TemporaryDirectory(prefix="arc3_official_") as tmp:
            staged = _stage_source(source or "git", Path(tmp), manifest)
            verify_tree(staged, manifest)
            partial = Path(tempfile.mkdtemp(prefix=dest.name + ".partial-", dir=dest.parent))
            try:
                shutil.copytree(staged, partial / "src", ignore=shutil.ignore_patterns(*IGNORED_PARTS))
                verify_tree(partial / "src", manifest)
                partial.rename(dest)
            except BaseException:
                shutil.rmtree(partial, ignore_errors=True)
                raise
        try:
            _build_venv(dest)
            verify_install(dest, manifest)
        except BaseException:
            shutil.rmtree(dest, ignore_errors=True)
            raise
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default="git", help="git (default), a directory or a .tar.gz")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args(argv)
    root = ensure_official_harness(args.source, args.dest)
    print(f"official ARC-AGI-3 harness verified at {root} ({load_manifest()['provenance']['commit']})")


if __name__ == "__main__":
    main()
