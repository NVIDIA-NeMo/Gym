# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a target-native NOOA runtime without changing the task environment."""

import hashlib
import io
import json
import re
import shlex
import tarfile
import tempfile
import tomllib
from pathlib import Path

from aiohttp import ClientTimeout

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_utils import raise_for_status, request


_RUNTIME_ROOT = "/opt/nemo-gym-nooa"
_PYTHON_VERSION = "3.13.14"
_PYTHON_RELEASE = "20260805"
_IMPORT_CHECK = (
    "import sys; assert sys.version_info >= (3,13,14); "
    "import nooa; import responses_api_agents.nooa_agent.sandbox_entrypoint; "
    "from responses_api_agents.nooa_agent.resource_tools import ResourceToolDispatcher; "
    "from responses_api_agents.nooa_agent.observability import GymTraceHooks"
)


def _runtime_archive(root: Path, *, requirements_path: Path | None = None) -> tuple[bytes, str]:
    """Include current source, never repository metadata, caches, or user artifacts."""
    files = {name: (root / name).read_bytes() for name in ("README.md", "LICENSE")}
    template = (root / "responses_api_agents/nooa_agent/runtime/pyproject.toml").read_text()
    runtime_config = tomllib.loads(template)["tool"]["nemo-gym-runtime"]
    core_dependencies = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    core_by_name = {
        re.split(r"[<>=!~\[ ;]", dependency, maxsplit=1)[0]: dependency for dependency in core_dependencies
    }
    dependencies = [core_by_name[name] for name in runtime_config["core-dependencies"]]
    dependencies.extend(runtime_config["additional-dependencies"])
    files["pyproject.toml"] = template.replace(
        "dependencies = []", f"dependencies = {json.dumps(dependencies)}", 1
    ).encode()
    for path in (root / "nemo_gym").rglob("*.py"):
        if not path.is_symlink() and not any(part.startswith(".") for part in path.relative_to(root).parts):
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    agent_dir = root / "responses_api_agents/nooa_agent"
    for path in agent_dir.glob("*.py"):
        if not path.is_symlink():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    selected_requirements = requirements_path or agent_dir / "requirements.txt"
    if not selected_requirements.is_absolute():
        selected_requirements = root / selected_requirements
    requirements = selected_requirements.read_text()
    runtime_requirements = []
    for line in requirements.splitlines():
        if not line.strip() or line.startswith(("#", "-e ")):
            continue
        # Same immutable source pin, without a dependency on the task image's git.
        line = re.sub(
            r"git\+https://github\.com/([^@ ]+)\.git@([0-9a-f]{40})$",
            r"https://github.com/\1/archive/\2.tar.gz",
            line,
        )
        runtime_requirements.append(line)
    files["runtime-requirements.txt"] = ("\n".join(runtime_requirements) + "\n").encode()
    digest = hashlib.sha256(f"{_PYTHON_VERSION}:{_PYTHON_RELEASE}:{_IMPORT_CHECK}".encode())
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, contents in sorted(files.items()):
            digest.update(name.encode() + b"\0" + contents + b"\0")
            entry = tarfile.TarInfo(name)
            entry.size = len(contents)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(contents))
    return output.getvalue(), digest.hexdigest()


async def _stage_python_if_needed(sandbox: AsyncSandbox, runtime: str, directory: Path) -> None:
    """Upload portable Python when the unchanged task image has no downloader."""
    target = await sandbox.exec(
        "if command -v curl >/dev/null || command -v wget >/dev/null || command -v python3 >/dev/null; "
        "then exit 0; fi; "
        '[ "$(uname -s)" = Linux ] || exit 1; '
        "libc=gnu; for loader in /lib/ld-musl-*.so.1; do "
        '[ ! -e "$loader" ] || libc=musl; done; echo "$(uname -m)/$libc"',
        cwd="/",
        timeout_s=30,
    )
    if target.return_code != 0:
        raise RuntimeError("Cannot detect portable Python target in task sandbox")
    target_name = (target.stdout or "").strip()
    if not target_name:
        return
    if target_name not in {"x86_64/gnu", "x86_64/musl", "aarch64/gnu", "aarch64/musl"}:
        raise RuntimeError(f"Unsupported NOOA Python target: {target_name!r}")
    arch, libc = target_name.split("/")
    url = (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{_PYTHON_RELEASE}/cpython-{_PYTHON_VERSION}+{_PYTHON_RELEASE}-"
        f"{arch}-unknown-linux-{libc}-install_only.tar.gz"
    )
    response = await request("GET", url, timeout=ClientTimeout(total=300))
    try:
        await raise_for_status(response)
        archive = directory / "python.tar.gz"
        with archive.open("wb") as output:
            async for chunk in response.content.iter_chunked(1024 * 1024):
                output.write(chunk)
    finally:
        response.release()
    await sandbox.upload(archive, f"{runtime}/python.tar.gz")


async def prepare_nooa_runtime(sandbox: AsyncSandbox, *, requirements_path: Path | None = None) -> str:
    """Return an isolated Python executable after staging and checking current source.

    Requires Linux, a supported portable CPython target, writable /opt, tar, and outbound HTTPS for package installation.
    Portable Python is uploaded by the controller when the image lacks a downloader. Task Python
    and its packages are never changed. Unsupported images fail during seed.
    """
    root = Path(__file__).resolve().parents[2]
    source, fingerprint = (
        _runtime_archive(root, requirements_path=requirements_path)
        if requirements_path is not None
        else _runtime_archive(root)
    )
    runtime = f"{_RUNTIME_ROOT}/{fingerprint}"
    python = f"{runtime}/python/bin/python3"
    check = f"{shlex.quote(python)} -I -c {shlex.quote(_IMPORT_CHECK)}"
    cached = await sandbox.exec(f"test -f {runtime}/ready && {check}", cwd="/", timeout_s=60)
    if cached.return_code == 0:
        return python
    # The upload directory is content-addressed; it lives outside the task repo.
    prepared = await sandbox.exec(f"mkdir -p {runtime}", cwd="/", timeout_s=30)
    if prepared.return_code != 0:
        raise RuntimeError("NOOA runtime preparation requires writable /opt/nemo-gym-nooa")
    with tempfile.TemporaryDirectory(prefix="nooa-runtime-") as directory:
        archive = Path(directory) / "source.tar.gz"
        archive.write_bytes(source)
        await sandbox.upload(archive, f"{runtime}/source.tar.gz")
        await _stage_python_if_needed(sandbox, runtime, Path(directory))
    command = f"""set -eu
root={shlex.quote(runtime)}
[ "$(uname -s)" = Linux ] || {{ echo 'NOOA runtime requires Linux' >&2; exit 1; }}
case "$(uname -m)" in
  x86_64|aarch64) arch=$(uname -m) ;;
  *) echo 'Unsupported NOOA runtime architecture' >&2; exit 1 ;;
esac
libc=gnu
for loader in /lib/ld-musl-*.so.1; do
  if [ -e "$loader" ]; then libc=musl; fi
done
command -v tar >/dev/null || {{ echo 'NOOA runtime requires tar' >&2; exit 1; }}
fetch() {{
  if command -v curl >/dev/null; then curl -fLsS --retry 3 -o "$2" "$1"
  elif command -v wget >/dev/null; then wget -q -O "$2" "$1"
  elif command -v python3 >/dev/null; then python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$1" "$2"
  else echo 'NOOA runtime requires curl, wget or python3' >&2; return 1; fi
}}
# A failed preparation cannot leave a reusable completion marker.
rm -f "$root/ready"
mkdir -p "$root/python" "$root/source/cache"
url="https://github.com/astral-sh/python-build-standalone/releases/download/{_PYTHON_RELEASE}/cpython-{_PYTHON_VERSION}+{_PYTHON_RELEASE}-$arch-unknown-linux-$libc-install_only.tar.gz"
[ -s "$root/python.tar.gz" ] || fetch "$url" "$root/python.tar.gz" || {{ echo "Unsupported/unavailable NOOA Python target: $arch/$libc" >&2; exit 1; }}
tar xzf "$root/python.tar.gz" -C "$root/python" --strip-components=1
tar xzf "$root/source.tar.gz" -C "$root/source"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
"$root/python/bin/python3" -I -m pip --isolated install --force-reinstall --no-cache-dir "$root/source" -r "$root/source/runtime-requirements.txt"
{check}
touch "$root/ready"
"""
    result = await sandbox.exec(command, cwd="/", timeout_s=1800)
    if result.return_code != 0:
        raise RuntimeError(
            f"NOOA runtime preparation failed: stdout={(result.stdout or '')[-2000:]} stderr={(result.stderr or '')[-2000:]}"
        )
    return python
