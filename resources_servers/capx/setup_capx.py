# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install the pinned CaP-X checkout and run its perception and IK API servers.

CaP-X needs Python 3.10 and its own simulator stack, so it gets its own uv venv next to this module rather than
dependencies in the Gym server venv. Its lockfile only syncs with uv 0.9.x; newer uv re-resolves it and fails on an
optional flash-attn pin, so the sync runs through ``uvx uv@0.9.5``.
"""

import fcntl
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


_LOGGER = logging.getLogger(__name__)

CAPX_URL = "https://github.com/capgym/cap-x"
CAPX_COMMIT = "53e9966d7a8e2fa7494676772bccc35280f5c0ed"
CAPX_UV_VERSION = "0.9.5"
# Submodules the Robosuite tasks need; the BEHAVIOR (b1k) stack is not installed.
CAPX_SUBMODULES = [
    "LIBERO-PRO",
    "verl",
    "robosuite",
    "contact_graspnet_pytorch",
    "libero_dependencies/robosuite",
    "sam3",
    "curobo",
]
DEFAULT_ROOT = Path(__file__).parent / ".capx"
# CaP-X's `full` launch profile: the union of the API servers the Robosuite text-tier configs declare.
API_SERVER_PORTS = {"sam2": 8113, "sam3": 8114, "graspnet": 8115, "pyroki": 8116, "owlvit": 8117}


def _run(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def ensure_capx(root: str | os.PathLike | None = None) -> Path:
    """Return a CaP-X checkout with a synced ``.venv``, installing the pinned commit into ``DEFAULT_ROOT`` if needed.

    ``root`` (or ``CAPX_ROOT``) points at an existing, already-set-up checkout and is used as-is.
    """
    root = root or os.environ.get("CAPX_ROOT")
    if root:
        path = Path(root).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"CaP-X checkout {path} does not exist")
        return path

    uvx = shutil.which("uvx")
    if uvx is None:
        raise FileNotFoundError("uvx is required to install CaP-X (https://docs.astral.sh/uv/)")
    DEFAULT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_ROOT.parent / ".capx.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = DEFAULT_ROOT / ".installed"
        if marker.exists() and marker.read_text().strip() == CAPX_COMMIT:
            return DEFAULT_ROOT
        _LOGGER.info(
            "Installing CaP-X %s into %s (first start only; takes several minutes)", CAPX_COMMIT, DEFAULT_ROOT
        )
        if not (DEFAULT_ROOT / ".git").exists():
            _run(["git", "clone", "--quiet", CAPX_URL, str(DEFAULT_ROOT)], cwd=DEFAULT_ROOT.parent)
        _run(["git", "checkout", "--quiet", CAPX_COMMIT], cwd=DEFAULT_ROOT)
        for submodule in CAPX_SUBMODULES:
            _run(
                ["git", "submodule", "update", "--quiet", "--init", f"capx/third_party/{submodule}"], cwd=DEFAULT_ROOT
            )
        uv = [uvx, f"uv@{CAPX_UV_VERSION}"]
        # A system Python 3.10 may lack the headers that building robosuite's evdev dependency needs.
        _run(
            [*uv, "venv", "--quiet", "--allow-existing", "--python-preference", "only-managed", "-p", "3.10"],
            cwd=DEFAULT_ROOT,
        )
        _run(
            [*uv, "sync", "--quiet", "--frozen", "--extra", "robosuite", "--extra", "contactgraspnet"],
            cwd=DEFAULT_ROOT,
        )
        marker.write_text(CAPX_COMMIT)
    return DEFAULT_ROOT


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def missing_api_servers() -> list[str]:
    return [name for name, port in API_SERVER_PORTS.items() if not _port_open(port)]


def start_api_servers(
    capx_root: Path, *, gpus: str | None, log_dir: Path, timeout_s: float
) -> subprocess.Popen | None:
    """Launch CaP-X's API servers unless they already run; wait until every port accepts connections.

    CaP-X's launcher reports readiness even when one server failed, so readiness is checked on the ports.
    Returns the launcher process (to stop at shutdown), or None if the servers were already running.
    """
    if not missing_api_servers():
        _LOGGER.info("CaP-X API servers already running on ports %s", sorted(API_SERVER_PORTS.values()))
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    args = [str(capx_root / ".venv" / "bin" / "python"), "capx/serving/launch_servers.py", "--profile", "full"]
    args += ["--log-dir", str(log_dir), "--timeout", str(int(timeout_s))]
    if gpus is not None:
        args += ["--gpus", gpus]
    launcher_log_path = log_dir / "launch_servers.log"
    launcher_log = open(launcher_log_path, "ab")
    start_offset = launcher_log.tell()
    process = subprocess.Popen(
        args, cwd=capx_root, stdout=launcher_log, stderr=subprocess.STDOUT, start_new_session=True
    )
    launcher_log.close()
    deadline = time.monotonic() + timeout_s
    while missing := missing_api_servers():
        # The launcher keeps running after a server dies and only logs it, so its log is the failure signal.
        failed = re.findall(
            r"\[FAILED\]\s+(\w+)", launcher_log_path.read_bytes()[start_offset:].decode(errors="replace")
        )
        if failed or process.poll() is not None or time.monotonic() > deadline:
            _stop(process)
            details = "".join(
                _log_tail(log_dir / f"{name}_{API_SERVER_PORTS[name]}.log") for name in sorted(set(failed))
            )
            raise RuntimeError(
                f"CaP-X API servers {sorted(set(failed)) or missing} did not start (logs in {log_dir}){details}"
            )
        time.sleep(5)
    _LOGGER.info("CaP-X API servers ready; logs in %s", log_dir)
    return process


def _log_tail(path: Path, lines: int = 15) -> str:
    if not path.exists():
        return ""
    tail = path.read_text(errors="replace").splitlines()[-lines:]
    return f"\n--- {path.name} ---\n" + "\n".join(tail)


def _stop(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    process.wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(ensure_capx(sys.argv[1] if len(sys.argv) > 1 else None))
