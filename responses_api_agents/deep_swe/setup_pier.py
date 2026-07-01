# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Install Pier into an isolated runtime to preserve Gym's OpenAI dependency pin."""

import asyncio
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from responses_api_agents.deep_swe.secure_paths import ensure_private_directory


PIER_VERSION = "0.3.0"
PIER_SOURCE_URL = "https://github.com/datacurve-ai/pier.git"
PIER_SOURCE_COMMIT = "e69a20e4e0ac073ec71fde0274bab3d9f40bac87"  # pragma: allowlist secret
PIER_REQUIREMENT = f"datacurve-pier @ git+{PIER_SOURCE_URL}@{PIER_SOURCE_COMMIT}"
PIER_RUNTIME_MODAL_VERSION = "1.5.1"
PIER_RUNTIME_LAYOUT_VERSION = 3
PIER_CONSTRAINTS_PATH = Path(__file__).with_name("pier-0.3.0-constraints.txt")
_INSTALL_LOCK = asyncio.Lock()
_VALIDATED_EXECUTABLES: set[tuple[Path, int, int, int]] = set()


def pier_constraints_sha256() -> str:
    """Hash only resolved requirement lines, not replay documentation comments."""
    requirements = [
        line.strip()
        for line in PIER_CONSTRAINTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return hashlib.sha256(("\n".join(requirements) + "\n").encode()).hexdigest()


def _expected_pier_direct_url() -> dict[str, object]:
    return {
        "url": PIER_SOURCE_URL,
        "vcs_info": {
            "commit_id": PIER_SOURCE_COMMIT,
            "requested_revision": PIER_SOURCE_COMMIT,
            "vcs": "git",
        },
    }


def pier_direct_url_sha256() -> str:
    canonical = json.dumps(_expected_pier_direct_url(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _pier_direct_url_path(runtime_dir: Path) -> Path | None:
    site_packages = runtime_dir / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    matches = list(site_packages.glob(f"datacurve_pier-{PIER_VERSION}.dist-info/direct_url.json"))
    return matches[0] if len(matches) == 1 else None


def _pier_direct_url_is_valid(runtime_dir: Path) -> bool:
    direct_url_path = _pier_direct_url_path(runtime_dir)
    if direct_url_path is None or not _owned_not_group_writable(direct_url_path):
        return False
    try:
        value = json.loads(direct_url_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value == _expected_pier_direct_url()


def _run(command: list[str], *, timeout: int = 1800) -> None:
    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if process.returncode != 0:
        output = "\n".join(part for part in (process.stdout, process.stderr) if part)
        raise RuntimeError(
            f"Pier runtime setup command failed with code {process.returncode}: "
            f"{command[0]} {command[1] if len(command) > 1 else ''}\n{output[-4000:]}"
        )


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise RuntimeError(f"Insecure Pier runtime lock file: {path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _owned_not_group_writable(path: Path, *, require_directory: bool = False) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    expected_kind = stat.S_ISDIR if require_directory else stat.S_ISREG
    return (
        expected_kind(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and not stat.S_IMODE(metadata.st_mode) & 0o022
    )


def _executable_is_valid(pier: Path) -> bool:
    if not _owned_not_group_writable(pier):
        return False
    metadata = pier.lstat()
    if not os.access(pier, os.X_OK):
        return False
    cache_key = (pier.resolve(), metadata.st_ino, metadata.st_mtime_ns, metadata.st_size)
    if cache_key in _VALIDATED_EXECUTABLES:
        return True
    try:
        process = subprocess.run(
            [str(pier), "--version"],
            check=False,
            capture_output=True,
            text=True,
            # A cold Pier import loads its full agent/environment registry and
            # can exceed 30 seconds on shared filesystems and fresh macOS caches.
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if process.returncode != 0 or process.stdout.strip() != PIER_VERSION:
        return False
    _VALIDATED_EXECUTABLES.add(cache_key)
    return True


def _runtime_is_valid(runtime_dir: Path, gym_root: Path) -> bool:
    marker = runtime_dir / "runtime.json"
    pier = runtime_dir / "bin" / "pier"
    python = runtime_dir / "bin" / "python"
    import_path = (
        runtime_dir
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nemo_gym_current_install.pth"
    )
    if (
        not _owned_not_group_writable(runtime_dir, require_directory=True)
        or not _owned_not_group_writable(marker)
        or not _owned_not_group_writable(pier)
        or not _owned_not_group_writable(import_path)
        or not python.is_file()
        or not _pier_direct_url_is_valid(runtime_dir)
    ):
        return False
    try:
        if import_path.read_text(encoding="utf-8") != str(gym_root.resolve()) + "\n":
            return False
    except OSError:
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    constraints_sha256 = pier_constraints_sha256()
    if metadata != {
        "pier_version": PIER_VERSION,
        "pier_source_url": PIER_SOURCE_URL,
        "pier_source_commit": PIER_SOURCE_COMMIT,
        "pier_direct_url_sha256": pier_direct_url_sha256(),
        "modal_version": PIER_RUNTIME_MODAL_VERSION,
        "runtime_layout_version": PIER_RUNTIME_LAYOUT_VERSION,
        "constraints_sha256": constraints_sha256,
        "gym_root": str(gym_root.resolve()),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "relocatable": True,
    }:
        return False
    return _executable_is_valid(pier)


def _install_runtime(runtime_dir: Path, gym_root: Path) -> Path:
    runtime_dir = Path(os.path.abspath(os.path.expanduser(runtime_dir)))
    ensure_private_directory(runtime_dir.parent)
    lock_path = runtime_dir.parent / f".{runtime_dir.name}.lock"
    with _exclusive_lock(lock_path):
        if _runtime_is_valid(runtime_dir, gym_root):
            return runtime_dir / "bin" / "pier"
        if runtime_dir.exists():
            raise RuntimeError(
                f"Pier runtime exists but does not match the requested pins: {runtime_dir}. "
                "Use a new pier_runtime_dir or remove this cache explicitly."
            )
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("DeepSWE requires the 'uv' executable to install its isolated Pier runtime")

        with tempfile.TemporaryDirectory(prefix="pier-runtime-", dir=runtime_dir.parent) as tmp:
            candidate = Path(tmp) / "venv"
            _run([uv, "venv", "--python", sys.executable, "--seed", "--relocatable", str(candidate)])
            python = candidate / "bin" / "python"
            _run(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--constraint",
                    str(PIER_CONSTRAINTS_PATH),
                    PIER_REQUIREMENT,
                ]
            )
            site_packages = (
                candidate / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
            )
            site_packages.mkdir(parents=True, exist_ok=True)
            import_path = site_packages / "nemo_gym_current_install.pth"
            import_path.write_text(str(gym_root.resolve()) + "\n", encoding="utf-8")
            os.chmod(import_path, 0o444)
            _run(
                [
                    str(python),
                    "-c",
                    "import nemo_gym.sandbox, pier; "
                    "from importlib.metadata import version; "
                    "from nemo_gym.sandbox.providers.modal import ModalProvider; "
                    "from responses_api_agents.deep_swe.pier_claude_code import ClaudeCodeNpmInstall; "
                    "from responses_api_agents.deep_swe.pier_sandbox_environment import PierSandboxEnvironment; "
                    f"assert version('datacurve-pier') == '{PIER_VERSION}'; "
                    f"assert version('modal') == '{PIER_RUNTIME_MODAL_VERSION}'; "
                    "assert ModalProvider.name == 'modal'; "
                    "assert ClaudeCodeNpmInstall.name() == 'claude-code'; "
                    "assert PierSandboxEnvironment.type() == 'nemo-gym-sandbox'",
                ]
            )
            marker = {
                "pier_version": PIER_VERSION,
                "pier_source_url": PIER_SOURCE_URL,
                "pier_source_commit": PIER_SOURCE_COMMIT,
                "pier_direct_url_sha256": pier_direct_url_sha256(),
                "modal_version": PIER_RUNTIME_MODAL_VERSION,
                "runtime_layout_version": PIER_RUNTIME_LAYOUT_VERSION,
                "constraints_sha256": pier_constraints_sha256(),
                "gym_root": str(gym_root.resolve()),
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                "relocatable": True,
            }
            (candidate / "runtime.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
            os.chmod(candidate / "runtime.json", 0o600)
            if not _runtime_is_valid(candidate, gym_root):
                raise RuntimeError("New Pier runtime failed executable validation before publication")
            candidate.rename(runtime_dir)

        if not _runtime_is_valid(runtime_dir, gym_root):
            shutil.rmtree(runtime_dir)
            raise RuntimeError("Published Pier runtime failed executable validation")
        return runtime_dir / "bin" / "pier"


async def ensure_pier_runtime(runtime_dir: Path, gym_root: Path) -> Path:
    async with _INSTALL_LOCK:
        return await asyncio.to_thread(_install_runtime, runtime_dir, gym_root)


__all__ = [
    "PIER_CONSTRAINTS_PATH",
    "PIER_RUNTIME_MODAL_VERSION",
    "PIER_RUNTIME_LAYOUT_VERSION",
    "PIER_SOURCE_URL",
    "PIER_SOURCE_COMMIT",
    "PIER_VERSION",
    "ensure_pier_runtime",
    "pier_constraints_sha256",
    "pier_direct_url_sha256",
]
