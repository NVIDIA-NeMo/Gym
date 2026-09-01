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

"""Prepare EnterpriseOps-Gym tool-schema snapshots from Hugging Face.

The seven per-domain `tools/list` snapshots are build-time inputs: `convert_tasks.py`
bakes them into dataset rows at prepare time, and nothing reads them at run time. They
are hosted rather than committed because they are ~30k lines of generated JSON that
`snapshot_tools.py` can re-capture from the upstream MCP gym containers at any time.

A refresh is a revision bump, not a commit -- see the "Refreshing the snapshots"
section of this server's README.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPO_ID = "nvidia/NeMo-Gym-EnterpriseOps-Assets"
DEFAULT_REVISION = "8918dc64b8575d5ff476e62e1cc3687523ab59c2"  # pragma: allowlist secret
REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE_COMMAND = "python -m resources_servers.enterpriseops_gym.prepare"

EOG_REPO_REVISION = "de22905d21a080b83bf4a54258afe4250ee2dd55"  # pragma: allowlist secret
GYM_DBS_ARCHIVE_URL = f"https://github.com/ServiceNow/EnterpriseOps-Gym/raw/{EOG_REPO_REVISION}/gym_dbs.zip"
GYM_DBS_DIR = REPO_ROOT / "resources_servers/enterpriseops_gym/data/gym_dbs"

TOOLS_DIR = REPO_ROOT / "resources_servers/enterpriseops_gym/data/tools"
REMOTE_TOOLS_DIR = Path("enterpriseops_gym/tools")

# One snapshot per upstream MCP gym container.
SNAPSHOT_FILENAMES = (
    "calendar.json",
    "csm.json",
    "drive.json",
    "email.json",
    "hr.json",
    "itsm.json",
    "teams.json",
)
# Pin of the exact bytes this integration was built and validated against. Recompute
# with `python -m resources_servers.enterpriseops_gym.prepare --print-hash <dir>`.
TOOLS_FILE_COUNT = 7
TOOLS_TREE_SHA256 = "d9ee1a279ec85985ba3fc59f2f9502a9c470301062800cab4d07c87b123354b3"  # pragma: allowlist secret

# Point at a directory holding the seven snapshots to skip the download entirely
# (air-gapped machines; see the README).
TOOLS_DIR_ENV_VAR = "NEMO_GYM_EOG_TOOLS_DIR"

SnapshotDownload = Callable[..., str]


@dataclass(frozen=True)
class AssetBundle:
    remote_dir: Path
    local_dir: Path
    file_count: int
    tree_sha256: str
    filenames: tuple[str, ...] | None = None


def _tools_bundle(repo_root: Path) -> AssetBundle:
    return AssetBundle(
        remote_dir=REMOTE_TOOLS_DIR,
        local_dir=repo_root / "resources_servers/enterpriseops_gym/data/tools",
        filenames=SNAPSHOT_FILENAMES,
        file_count=TOOLS_FILE_COUNT,
        tree_sha256=TOOLS_TREE_SHA256,
    )


def _runtime_bundles(repo_root: Path) -> tuple[AssetBundle, ...]:
    return (_tools_bundle(repo_root),)


def tree_hash(directory: Path, filenames: Iterable[str] | None = None) -> tuple[int, str]:
    """Return a stable filename-and-content hash for one flat asset directory.

    With `filenames`, hash only those entries. The unfiltered form validates a freshly
    downloaded snapshot (where an unexpected file means the wrong revision); the
    filtered form checks an already-materialized directory, which callers may have
    added unrelated files to.
    """
    digest = hashlib.sha256()
    if filenames is None:
        paths = sorted(path for path in directory.iterdir() if path.is_file())
    else:
        paths = [directory / name for name in sorted(filenames)]
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return len(paths), digest.hexdigest()


def _snapshot_download() -> SnapshotDownload:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("Preparing EnterpriseOps-Gym tool snapshots requires `huggingface-hub`.") from exc
    return snapshot_download


def _validate_bundle(snapshot: Path, bundle: AssetBundle) -> Path:
    source = snapshot / bundle.remote_dir
    if not source.is_dir():
        raise ValueError(f"Missing asset directory in snapshot: {bundle.remote_dir}")
    actual_names = tuple(sorted(path.name for path in source.iterdir() if path.is_file()))
    if bundle.filenames is not None and actual_names != tuple(sorted(bundle.filenames)):
        raise ValueError(
            f"Invalid filenames for {bundle.remote_dir}: expected={sorted(bundle.filenames)}, actual={actual_names}"
        )
    actual_hash = tree_hash(source)
    expected_hash = (bundle.file_count, bundle.tree_sha256)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Asset checksum mismatch for {bundle.remote_dir}: expected={expected_hash}, actual={actual_hash}"
        )
    return source


def _materialize_bundle(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".enterpriseops-gym-prepare-", dir=destination) as temp_dir:
        staging = Path(temp_dir)
        for source_file in source.iterdir():
            if source_file.is_file():
                shutil.copy2(source_file, staging / source_file.name)
        for staged_file in staging.iterdir():
            staged_file.replace(destination / staged_file.name)


def _is_current(bundle: AssetBundle) -> bool:
    """True when the destination already holds exactly the pinned snapshot bytes."""
    if not bundle.local_dir.is_dir():
        return False
    filenames = bundle.filenames or ()
    if not all((bundle.local_dir / name).is_file() for name in filenames):
        return False
    return tree_hash(bundle.local_dir, filenames) == (bundle.file_count, bundle.tree_sha256)


def _validate_override_dir(directory: Path, bundle: AssetBundle) -> Path:
    """Validate a user-supplied snapshot directory named by `TOOLS_DIR_ENV_VAR`."""
    if not directory.is_dir():
        raise ValueError(f"{TOOLS_DIR_ENV_VAR}={directory} is not a directory.")
    filenames = bundle.filenames or ()
    missing = sorted(name for name in filenames if not (directory / name).is_file())
    if missing:
        raise ValueError(f"{TOOLS_DIR_ENV_VAR}={directory} is missing snapshots: {missing}")
    actual_hash = tree_hash(directory, filenames)
    expected_hash = (bundle.file_count, bundle.tree_sha256)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Asset checksum mismatch for {TOOLS_DIR_ENV_VAR}={directory}: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    return directory


def prepare(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    repo_root: Path = REPO_ROOT,
    snapshot_download: SnapshotDownload | None = None,
) -> tuple[Path, ...]:
    """Download, validate, and materialize the tool-schema snapshots."""
    bundles = _runtime_bundles(repo_root)
    download = snapshot_download or _snapshot_download()
    snapshot = Path(
        download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{bundle.remote_dir.as_posix()}/*" for bundle in bundles],
        )
    )

    sources = tuple(_validate_bundle(snapshot, bundle) for bundle in bundles)
    for source, bundle in zip(sources, bundles, strict=True):
        _materialize_bundle(source, bundle.local_dir)
    return tuple(bundle.local_dir for bundle in bundles)


def ensure_gym_dbs(dest: Path = GYM_DBS_DIR) -> Path:
    """Download and extract gym_dbs.zip from the pinned EOG release. No-op if dest is non-empty."""
    if dest.is_dir() and any(dest.iterdir()):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(GYM_DBS_ARCHIVE_URL) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest)
    return dest


def ensure_tool_snapshots(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    repo_root: Path = REPO_ROOT,
    snapshot_download: SnapshotDownload | None = None,
) -> Path:
    """Return a directory holding the seven snapshots, downloading them if needed.

    Idempotent: a directory that already matches the pin is used as-is, with no
    network call. `TOOLS_DIR_ENV_VAR` bypasses the download for air-gapped machines.
    """
    bundle = _tools_bundle(repo_root)
    override = os.getenv(TOOLS_DIR_ENV_VAR)
    if override:
        return _validate_override_dir(Path(override), bundle)
    if _is_current(bundle):
        return bundle.local_dir
    prepare(repo_id=repo_id, revision=revision, repo_root=repo_root, snapshot_download=snapshot_download)
    return bundle.local_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--print-hash",
        nargs="?",
        const=TOOLS_DIR,
        type=Path,
        metavar="DIR",
        help="Print the (file_count, tree_sha256) pin for DIR and exit; used when republishing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.print_hash is not None:
        file_count, digest = tree_hash(args.print_hash)
        print(f"file_count = {file_count}")
        print(f"tree_sha256 = {digest}")
        return
    destinations = prepare(
        repo_id=args.repo_id,
        revision=args.revision,
        repo_root=args.repo_root,
    )
    for destination in destinations:
        print(f"Prepared EnterpriseOps-Gym tool snapshots in {destination}")
    gym_dbs = ensure_gym_dbs()
    print(f"Prepared EnterpriseOps-Gym seed SQL archive in {gym_dbs}")


if __name__ == "__main__":
    main()
