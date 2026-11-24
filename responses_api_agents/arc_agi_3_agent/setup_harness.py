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
"""Install the pinned ARC-AGI-3 harness (not vendored: it is third-party code) and check every byte of it.

An install is a directory with ``harness/`` (the Kaggle ``...-harness`` dataset tree) and
``config/harness_env.toml``. ``harness_manifest.json`` pins the sha256 of every file. Sources:

- a directory with that layout, or a ``.tar.gz`` of one (a single top-level directory is allowed);
- ``kaggle:<owner>/<prefix>``: downloads the ``<prefix>-harness`` and ``<prefix>-config`` Kaggle datasets. Public
  datasets (the default, ``eladsarafian/arc3-c237-j30vfh-20261102``) download anonymously; only when Kaggle refuses an
  anonymous download is a token in ``KAGGLE_API_TOKEN`` or ``~/.kaggle/access_token`` sent (private datasets need one
  with access), to the Kaggle API only: never to the storage a download redirects to.

    python responses_api_agents/arc_agi_3_agent/setup_harness.py <source> [--dest DIR]
"""

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "harness_manifest.json"
DEFAULT_DEST = HERE / ".arc3_harness"
KAGGLE_API = "https://www.kaggle.com/api/v1"
# Kaggle answers 404 to urllib's default User-Agent (Python-urllib/x.y), even with a valid token.
KAGGLE_USER_AGENT = "nemo-gym-arc-agi-3"
# Files Python creates when the harness is imported; not part of the pinned tree.
IGNORED_PARTS = ("__pycache__",)
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


def verify_install(root: Path, manifest: dict | None = None) -> None:
    """Raise unless ``root`` holds exactly the pinned harness files and harness_env.toml."""
    manifest = manifest or load_manifest()
    problems = []
    for subdir, pinned in (("harness", manifest["harness_files"]), ("config", manifest["config_files"])):
        base = root / subdir
        if not base.is_dir():
            raise FileNotFoundError(f"ARC-AGI-3 harness install {root} has no {subdir}/ directory")
        present = set(_tree_files(base)) - {"MANIFEST.json", "HARNESS-PROVENANCE.json"}
        if subdir == "config":
            present &= set(pinned)  # the config dataset's other files (profile.args, ...) are engine-side
        for rel, digest in pinned.items():
            path = base / rel
            if not path.is_file():
                problems.append(f"missing {subdir}/{rel}")
            elif _sha256(path) != digest:
                problems.append(f"sha256 differs: {subdir}/{rel}")
        problems += [f"unpinned file {subdir}/{rel}" for rel in sorted(present - set(pinned))]
    if problems:
        shown = "; ".join(problems[:10])
        raise ValueError(f"ARC-AGI-3 harness install {root} does not match harness_manifest.json: {shown}")


def _single_root(directory: Path) -> Path:
    if (directory / "harness").is_dir():
        return directory
    children = [p for p in directory.iterdir() if p.is_dir()]
    if len(children) == 1 and (children[0] / "harness").is_dir():
        return children[0]
    raise FileNotFoundError(f"{directory} has no harness/ directory (expected harness/ and config/)")


def _kaggle_token() -> str | None:
    """The Kaggle token when one is configured; None otherwise (public datasets download without one)."""
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        return token
    path = Path.home() / ".kaggle" / "access_token"
    if not path.is_file():
        return None
    return path.read_text().strip() or None


def _kaggle_request(url: str, token: str | None) -> urllib.request.Request:
    """A GET of the Kaggle API; with ``token``, its bearer credential goes to the Kaggle API only. It is an unredirected
    header: urllib does not copy it to a redirect's target (Kaggle answers a download with a redirect to cloud storage,
    whose signed URL needs no credential)."""
    request = urllib.request.Request(url, headers={"User-Agent": KAGGLE_USER_AGENT})
    if token:
        if not url.startswith(KAGGLE_API + "/"):
            raise ValueError(f"refusing to send the Kaggle token to {url}")
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    return request


def _kaggle_get(url: str) -> bytes:
    """Download ``url`` anonymously (public datasets); only when Kaggle refuses that, again with the token when one is
    configured (private datasets)."""
    try:
        with urllib.request.urlopen(_kaggle_request(url, None), timeout=300) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        token = _kaggle_token()
        if exc.code not in (401, 403, 404) or not token:
            raise
    with urllib.request.urlopen(_kaggle_request(url, token), timeout=300) as response:
        return response.read()


def _download_kaggle(ref: str, workdir: Path) -> Path:
    owner, prefix = ref.split("/", 1)
    root = workdir / "kaggle"
    for kind in ("harness", "config"):
        data = _kaggle_get(f"{KAGGLE_API}/datasets/download/{owner}/{prefix}-{kind}")
        zipfile.ZipFile(io.BytesIO(data)).extractall(root / kind)
    return root


def _stage_source(source: str, workdir: Path) -> Path:
    if source.startswith("kaggle:"):
        return _download_kaggle(source[len("kaggle:") :], workdir)
    path = Path(source).expanduser()
    if path.is_dir():
        return _single_root(path)
    if path.is_file() and path.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path) as archive:
            archive.extractall(workdir / "tar", filter="data")
        return _single_root(workdir / "tar")
    raise FileNotFoundError(f"ARC-AGI-3 harness source {source!r} is not a directory, a tarball or kaggle:<ref>")


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


def ensure_harness(source: str | None, dest: Path = DEFAULT_DEST) -> Path:
    """Return a verified harness install at ``dest``, installing it from ``source`` when ``dest`` is absent.

    Concurrent calls (several servers on one checkout) share one install: the first installs under ``install_lock``
    and the others then verify its result. The install is built in a directory of its own next to ``dest`` and renamed
    into place when complete; a failed install leaves nothing at ``dest``, so a later call retries."""
    manifest = load_manifest()
    with install_lock(dest):
        if dest.exists():
            verify_install(dest, manifest)
            return dest
        if not source:
            raise FileNotFoundError(
                f"The ARC-AGI-3 harness is not installed at {dest}; set harness_source (directory, tarball or "
                "kaggle:<ref>)"
            )
        with tempfile.TemporaryDirectory(prefix="arc3_harness_") as tmp:
            staged = _stage_source(source, Path(tmp))
            verify_install(staged, manifest)
            partial = Path(tempfile.mkdtemp(prefix=dest.name + ".partial-", dir=dest.parent))
            try:
                shutil.copytree(staged / "harness", partial / "harness")
                (partial / "config").mkdir()
                for rel in manifest["config_files"]:
                    shutil.copy2(staged / "config" / rel, partial / "config" / rel)
                verify_install(partial, manifest)
                partial.rename(dest)
            except BaseException:
                shutil.rmtree(partial, ignore_errors=True)
                raise
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", help="directory, .tar.gz, or kaggle:<owner>/<prefix>")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args(argv)
    root = ensure_harness(args.source, args.dest)
    print(f"ARC-AGI-3 harness verified at {root}")


if __name__ == "__main__":
    main()
