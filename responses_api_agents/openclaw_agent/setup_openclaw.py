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

"""Provision the ``openclaw`` CLI for the OpenClaw responses-API agent.

The module guarantees that ``openclaw`` is resolvable through ``PATH``. When it
is missing it is installed with ``npm install -g``; when ``npm`` itself is
missing a private Node.js toolchain is unpacked next to this file first.

The toolchain is downloaded from nodejs.org for the running platform: Linux,
macOS and Windows, on x64 and arm64. Anything else raises, since nodejs.org
publishes no build for it.

Versions are pinned for reproducibility and can be overridden per process with
environment variables:

* ``OPENCLAW_VERSION`` - npm version spec of the ``openclaw`` package.
* ``OPENCLAW_NODE_VERSION`` - Node.js version downloaded when ``npm`` is absent.

Installed runtimes are validated before reuse: an ``openclaw`` already on
``PATH`` is accepted only when it reports the requested version, and a Node
toolchain (local cache or system ``npm``) is accepted only when its ``node``
satisfies the ``engines.node`` range of the ``openclaw`` release being pinned.
Otherwise the incompatible artifact is replaced.

Examples:
    Install the pinned default and make it importable by the agent::

        >>> from responses_api_agents.openclaw_agent.setup_openclaw import ensure_openclaw
        >>> ensure_openclaw()

    Pin a version explicitly (a config value, typically)::

        >>> ensure_openclaw("2026.9.4")

    Override both versions from the environment::

        $ OPENCLAW_VERSION=2026.8.1 OPENCLAW_NODE_VERSION=26.9.0 python -m my_runner

    Inspect the resolved versions without touching the filesystem::

        >>> resolve_openclaw_version(None), resolve_node_version()
        ('2026.9.4', '24.21.0')
"""

import ctypes
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from ctypes import wintypes
from pathlib import Path


LOG = logging.getLogger(__name__)

_OPENCLAW_PKG = "openclaw"

#: Newest published ``openclaw`` release; used when no override is supplied.
DEFAULT_OPENCLAW_VERSION = "2026.9.4"

#: Newest release of the Node.js 24 LTS line, which is the line OpenClaw's own
#: installer provisions on Linux (``NODE_LINUX_DEFAULT_MAJOR=24``). OpenClaw
#: declares ``engines.node = ">=24.16.0 <25 || >=26.1.0"``, so 24.21.0 is the
#: latest *stable* runtime that satisfies it.
DEFAULT_NODE_VERSION = "24.21.0"

OPENCLAW_VERSION_ENV = "OPENCLAW_VERSION"
NODE_VERSION_ENV = "OPENCLAW_NODE_VERSION"

_NPM_INSTALL_ATTEMPTS = 3
_LOCAL_PREFIX = Path(__file__).parent / ".openclaw_node"
_USER_LOCAL_BIN = Path.home() / ".local" / "bin"

# `openclaw` declares this `engines.node` range (e.g. 2026.6.11 declared
# ">=22.19.0"). It encodes every engine constraint the installer must honour, so
# when upstream changes it this constant — not a parsed node version — decides
# whether an existing runtime is reusable.
OPENCLAW_ENGINES_NODE = ">=24.16.0 <25 || >=26.1.0"

#: ``sys.platform`` value -> the OS token nodejs.org uses in its archive names.
_NODE_OS = {"linux": "linux", "darwin": "darwin", "win32": "win", "cygwin": "win"}

#: Lowercased ``platform.machine()`` spelling -> the arch token nodejs.org uses.
_NODE_ARCH = {"x86_64": "x64", "amd64": "x64", "x64": "x64", "aarch64": "arm64", "arm64": "arm64"}


def resolve_openclaw_version(version: str | None = None) -> str:
    """Return the ``openclaw`` version to install.

    ``OPENCLAW_VERSION`` wins over *version* so an operator can override a
    pinned config without editing it; the module default is the last resort.
    """
    return os.environ.get(OPENCLAW_VERSION_ENV) or version or DEFAULT_OPENCLAW_VERSION


def resolve_node_version() -> str:
    """Return the Node.js version to download, honouring ``OPENCLAW_NODE_VERSION``."""
    return os.environ.get(NODE_VERSION_ENV) or DEFAULT_NODE_VERSION


def _parse_version(version: str) -> tuple[int, ...]:
    """Return the numeric ``(major[, minor[, patch]])`` prefix of *version*.

    Accepts partial versions (``24``, ``24.16``) and ignores prerelease/build
    suffixes (``24.21.0-rc.1``); npm range logic only needs the numeric prefix.
    """
    match = re.match(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", version.strip().lstrip("v"))
    if match is None:
        raise ValueError(f"unparseable version {version!r}")
    return tuple(int(part) for part in match.groups(default="0") if part is not None)


def _range_comparator(op: str, operand: str, actual: tuple[int, int, int]) -> bool:
    """Evaluate one npm range primitive (``>=X.Y.Z``, ``<X``, …) against *actual*."""
    wanted = _parse_version(operand)
    return {
        ">=": actual >= wanted,
        ">": actual > wanted,
        "<=": actual <= wanted,
        "<": actual < wanted,
        "=": actual == wanted,
    }[op]


#: npm range primitives, longest operator first so ``>=`` is not read as ``>``.
_RANGE_PRIMITIVE = re.compile(r"^(>=|<=|>|<|=|\^|~)?v?(\d+(?:\.\d+){0,2})")


def _satisfies_range(node_version: str, node_range: str) -> bool:
    """Return whether *node_version* satisfies the npm *node_range*.

    Supports the constructs ``openclaw`` actually publishes — ``||``,
    whitespace-AND, and ``>=``/``<``/``=`` comparators with partial operands.
    npm pads a partial operand differently by comparison: ``>=24`` means
    ``>=24.0.0`` while ``<24`` means ``<24.0.0``, so both are padded to the full
    triple; ``^``/``~``/``x`` wildcards keep only the precision the operand
    states, which is all coarse ranges like ``>=22`` need.
    """
    actual = _parse_version(node_version)
    for alternative in node_range.split("||"):
        if all(_primitive_matches(primitive, actual) for primitive in alternative.split()):
            return True
    return False


def _primitive_matches(primitive: str, actual: tuple[int, int, int]) -> bool:
    """Evaluate one npm range primitive against the actual version triple."""
    match = _RANGE_PRIMITIVE.match(primitive)
    if match is None:
        raise ValueError(f"unsupported npm range primitive {primitive!r}")
    op, operand = match.groups()
    # Truncate at an x/X/* wildcard component; whatever remains is the
    # precision the operand actually states ("22.x" == "22").
    operand = re.split(r"[xX*]", operand, maxsplit=1)[0].rstrip(".")
    if not operand:
        return True
    parts = [int(piece) for piece in operand.split(".")]

    if op in (None, "^", "~"):
        # Wildcard-ish: only the stated precision constrains.
        return actual[: len(parts)] == tuple(parts)
    padded = (parts + [0, 0, 0])[:3]
    return _range_comparator(op, ".".join(map(str, padded)), actual)


def _node_reported_version(node_bin: str) -> str | None:
    """Return the version *node_bin* prints, or ``None`` when it fails to run.

    A corrupt or half-extracted cached toolchain (e.g. one unpacked for a
    different architecture) exits non-zero; treat that as "not usable" rather
    than crashing startup.
    """
    try:
        completed = subprocess.run([node_bin, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip().lstrip("v") or None


#: ``IsWow64Process2`` native-machine codes (IMAGE_FILE_MACHINE_*) we can map.
_IMAGE_FILE_MACHINE_ARM64 = 0xAA64
_IMAGE_FILE_MACHINE_AMD64 = 0x8664
_IMAGE_FILE_MACHINE_I386 = 0x014C


def _windows_machine() -> str:
    """Return the host CPU architecture on Windows, robust to emulation.

    The obvious primitive, ``platform.machine()``, is *not* trustworthy on
    Windows-on-ARM: it reports the architecture of the *running interpreter*,
    not of the host CPU — an x64 ``python.exe`` running under emulation on
    ARM64 silicon gets ``AMD64``.

    Worse, CPython 3.13+ makes the value *nondeterministic on one machine*:
    ``platform.machine()`` consults WMI first (``Win32_Processor.Architecture``,
    which reports the true host CPU) and silently falls back to the
    ``PROCESSOR_ARCHITECTURE`` environment variable (which reports the emulated
    image, ``AMD64`` here) whenever the WMI service is slow or times out. We
    observed both values alternating between processes on the same ARM64 host,
    which made the toolchain download flip between the win-arm64 and win-x64
    zips from run to run.

    ``IsWow64Process2`` (Windows 10+) closes the trap: it reports the *native*
    process machine directly from the kernel, independent of how this
    interpreter was built and with no WMI involved. Any failure (old Windows,
    unexpected API shape) falls back to ``platform.machine()`` — imperfect, but
    no worse than before.

    Returns:
        A ``platform.machine()``-style token such as ``"ARM64"`` or ``"AMD64"``
        ("" when the native machine is not one we recognise).
    """
    try:
        kernel32 = ctypes.windll.kernel32
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = wintypes.HANDLE

        is_wow64_process2 = kernel32.IsWow64Process2
        is_wow64_process2.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.USHORT),
            ctypes.POINTER(wintypes.USHORT),
        ]
        is_wow64_process2.restype = wintypes.BOOL

        process_machine = wintypes.USHORT()
        native_machine = wintypes.USHORT()
        ok = is_wow64_process2(get_current_process(), ctypes.byref(process_machine), ctypes.byref(native_machine))
        if ok:
            return {
                _IMAGE_FILE_MACHINE_ARM64: "ARM64",
                _IMAGE_FILE_MACHINE_AMD64: "AMD64",
                _IMAGE_FILE_MACHINE_I386: "x86",
            }.get(native_machine.value, "")
    except AttributeError:
        # Pre-Windows-10 kernel: the API does not exist; use the legacy source.
        pass
    return platform.machine()


def _node_platform() -> tuple[str, str]:
    """Return the ``(os, arch)`` tokens nodejs.org uses for the running interpreter.

    Raises:
        RuntimeError: the OS or CPU architecture has no published Node.js build.
    """
    node_os = _NODE_OS.get(sys.platform)
    if node_os is None:
        raise RuntimeError(
            f"no Node.js build is published for platform {sys.platform!r}; install Node.js manually "
            f"and put 'npm' on PATH"
        )
    machine = (_windows_machine() if node_os == "win" else platform.machine()).lower()
    node_arch = _NODE_ARCH.get(machine)
    if node_arch is None:
        raise RuntimeError(
            f"no Node.js build is published for architecture {machine!r}; install Node.js "
            f"manually and put 'npm' on PATH"
        )
    return node_os, node_arch


def _node_dist_url(node_version: str) -> str:
    """Return the nodejs.org download URL of *node_version* for this platform.

    Windows builds ship as ``.zip``; every other platform ships ``.tar.xz``.
    """
    node_os, node_arch = _node_platform()
    suffix = ".zip" if node_os == "win" else ".tar.xz"
    return f"https://nodejs.org/dist/v{node_version}/node-v{node_version}-{node_os}-{node_arch}{suffix}"


def _node_bin_dir(prefix: Path) -> Path:
    """Return the directory under *prefix* that holds the ``node``/``npm`` launchers.

    Windows distributions place them at the root of the tree; every other
    platform uses a ``bin/`` subdirectory.
    """
    return prefix if _node_platform()[0] == "win" else prefix / "bin"


def _prepend_path(bin_dir: Path | str) -> None:
    """Put *bin_dir* at the front of ``PATH`` for this process."""
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _openclaw_on_path() -> str | None:
    return shutil.which(_OPENCLAW_PKG)


def _adopt_user_local_bin() -> bool:
    """Add ``~/.local/bin`` to ``PATH`` when it already holds ``openclaw``."""
    if not shutil.which(_OPENCLAW_PKG, path=str(_USER_LOCAL_BIN)):
        return False
    _prepend_path(_USER_LOCAL_BIN)
    return True


def _adopt_npm_global_bin(npm_bin: str) -> bool:
    """Add npm's global bin directory to ``PATH`` when it exists."""
    completed = subprocess.run([npm_bin, "prefix", "-g"], capture_output=True, text=True)
    prefix = completed.stdout.strip()
    if not prefix:
        return False
    global_bin = _node_bin_dir(Path(prefix))
    if not global_bin.is_dir():
        return False
    _prepend_path(global_bin)
    return True


def _npm_install(npm_bin: str, version: str) -> None:
    """Run ``npm install -g openclaw@version``, retrying transient failures."""
    pkg = f"{_OPENCLAW_PKG}@{version}"
    for attempt in range(1, _NPM_INSTALL_ATTEMPTS + 1):
        try:
            subprocess.run([npm_bin, "install", "-g", pkg], check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == _NPM_INSTALL_ATTEMPTS:
                raise
            LOG.warning("npm install %s failed (attempt %d/%d), retrying", pkg, attempt, _NPM_INSTALL_ATTEMPTS)
            time.sleep(2 * attempt)


def _download_node_archive(url: str, dest: Path) -> None:
    LOG.info("downloading %s", url)
    urllib.request.urlretrieve(url, dest)  # noqa: S310


def _extract_node_archive(archive: Path, prefix: Path) -> None:
    """Unpack *archive* into *prefix*.

    ``zipfile`` drops the executable bit, but only Windows ships a zip and there
    the launchers are ``.exe``/``.cmd``, so the mode does not matter.
    """
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(prefix)
    else:
        with tarfile.open(archive, "r:xz") as tf:
            tf.extractall(prefix, filter="data")


def _flatten_extracted_node(prefix: Path) -> None:
    """Hoist the ``node-vX.Y.Z-<os>-<arch>/`` payload directly into *prefix*."""
    nested = next((p for p in prefix.iterdir() if p.is_dir() and p.name.startswith("node-")), None)
    if nested is None:
        raise RuntimeError(f"Node.js archive did not contain a 'node-*' directory under {prefix}")
    for item in nested.iterdir():
        item.rename(prefix / item.name)
    nested.rmdir()


def _install_node_locally(node_version: str) -> Path:
    """Unpack a private Node.js toolchain and return the directory holding ``node``.

    A cached toolchain is only reused when its ``node`` actually runs and
    reports *node_version*; anything else (a leftover from an older pin or a
    corrupt partial extraction) is wiped and re-provisioned so the runtime
    always matches the resolved request.
    """
    bin_dir = _node_bin_dir(_LOCAL_PREFIX)
    cached = shutil.which("node", path=str(bin_dir))
    if cached is not None:
        reported = _node_reported_version(cached)
        if reported == node_version:
            LOG.info("reusing cached Node.js %s in %s", node_version, _LOCAL_PREFIX)
            return bin_dir
        LOG.info(
            "cached Node.js toolchain reports %s but %s is requested; re-provisioning %s",
            reported or "nothing (corrupt)",
            node_version,
            _LOCAL_PREFIX,
        )
        shutil.rmtree(_LOCAL_PREFIX)

    _LOCAL_PREFIX.mkdir(parents=True, exist_ok=True)
    url = _node_dist_url(node_version)
    archive = _LOCAL_PREFIX / url.rsplit("/", 1)[-1]
    try:
        _download_node_archive(url, archive)
        _extract_node_archive(archive, _LOCAL_PREFIX)
    finally:
        archive.unlink(missing_ok=True)

    _flatten_extracted_node(_LOCAL_PREFIX)
    return bin_dir


def _ensure_npm() -> str:
    """Return a usable ``npm`` whose ``node`` satisfies OpenClaw's engine range.

    A system ``npm`` is only reused when the ``node`` beside it satisfies
    ``OPENCLAW_ENGINES_NODE``; anything outside the range (an older runtime,
    or a newer major OpenClaw does not support yet) is ignored and a private
    toolchain is provisioned instead, so the ``npm install -g`` below cannot
    silently run against a runtime OpenClaw refuses to start on.
    """
    npm = shutil.which("npm")
    if npm:
        node_version = _node_on_path()
        if node_version is None or _satisfies_range(node_version, OPENCLAW_ENGINES_NODE):
            LOG.info("using system npm (%s), node %s", npm, node_version or "unknown")
            return npm
        LOG.info(
            "system npm (%s) runs node %s, which does not satisfy openclaw's engines.node %r; "
            "provisioning a private Node.js toolchain",
            npm,
            node_version,
            OPENCLAW_ENGINES_NODE,
        )

    node_version = resolve_node_version()
    LOG.info("npm not found; installing local Node.js %s", node_version)
    bin_dir = _install_node_locally(node_version)
    _prepend_path(bin_dir)

    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError(f"npm not found after local Node.js install in {bin_dir}")
    return npm


def _node_on_path() -> str | None:
    """Return the reported version of the ``node`` first on ``PATH``, if any."""
    found = shutil.which("node")
    return _node_reported_version(found) if found else None


def _openclaw_reported_version(openclaw_bin: str) -> str | None:
    """Return the version *openclaw_bin* prints, or ``None`` when it fails.

    The launcher may be a shim whose ``node`` is unusable (the exact failure
    this module exists to fix), so a non-zero exit is treated the same as an
    unknown version: not acceptable evidence of a compatible install.
    """
    try:
        completed = subprocess.run([openclaw_bin, "--version"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    match = re.search(r"(\d+(?:\.\d+)+)", completed.stdout)
    return match.group(1) if match else None


def _installed_openclaw_matches(openclaw_bin: str, requested: str) -> bool:
    """Return whether *openclaw_bin* reports the *requested* ``openclaw`` version.

    npm specs (``^2026.9.0``, ``2026.9``) are compared at the precision the
    requested spec states, mirroring npm's own resolution semantics.
    """
    reported = _openclaw_reported_version(openclaw_bin)
    if reported is None:
        return False
    requested_match = re.match(r"[\^~>=< ]*v?(\d+(?:\.\d+){0,2})", requested)
    if requested_match is None:
        return False
    precision = len(requested_match.group(1).split("."))
    try:
        reported_parts = _parse_version(reported)
        requested_parts = _parse_version(requested_match.group(1))
    except ValueError:
        return False
    return reported_parts[:precision] == requested_parts[:precision]


def _expose_installed_openclaw(npm_bin: str) -> str | None:
    """Locate ``openclaw`` after a successful install, extending ``PATH`` as needed.

    ``npm install -g`` may target a prefix that is not on ``PATH`` yet, and some
    setups link the launcher into ``~/.local/bin`` instead.
    """
    for adopt in (lambda: True, lambda: _adopt_npm_global_bin(npm_bin), _adopt_user_local_bin):
        if adopt() and (found := _openclaw_on_path()):
            return found
    return None


def ensure_openclaw(version: str | None = None) -> None:
    """Ensure the requested ``openclaw`` version is on ``PATH``.

    An existing install is only accepted when ``openclaw --version`` reports
    the resolved version, so changing ``OPENCLAW_VERSION`` (or the config pin)
    takes effect instead of silently keeping whatever was installed earlier.
    When the version differs — or the existing launcher fails to report one at
    all — the requested release is installed over it via npm.

    Args:
        version: npm version spec to pin. Overridden by ``OPENCLAW_VERSION`` and
            defaulted to :data:`DEFAULT_OPENCLAW_VERSION`.

    Raises:
        RuntimeError: the install reported success but ``openclaw`` is still not
            resolvable, or no ``npm`` could be provisioned.
    """
    requested = resolve_openclaw_version(version)
    existing = _openclaw_on_path()
    if existing is None and _adopt_user_local_bin():
        existing = _openclaw_on_path()
    if existing and _installed_openclaw_matches(existing, requested):
        LOG.info("openclaw %s already installed at %s", requested, existing)
        return
    if existing:
        LOG.info("openclaw at %s does not report the requested version %s; reinstalling", existing, requested)

    npm = _ensure_npm()
    _npm_install(npm, requested)

    found = _expose_installed_openclaw(npm)
    if not found:
        raise RuntimeError("openclaw install appeared to succeed but 'openclaw' is still not on PATH")

    LOG.info("openclaw is ready at %s", found)
