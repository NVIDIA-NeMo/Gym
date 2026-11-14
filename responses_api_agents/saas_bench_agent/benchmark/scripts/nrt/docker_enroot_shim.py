#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal ``docker exec`` / ``docker cp`` / start-time inspection compatibility for Enroot verifies.

Task verify scripts invoke ``docker exec`` to inspect files or databases.
On GPU nodes the Enroot backend records PIDs under ``$SAAS_ENROOT_STATE_DIR``.

Exec recipe that works with util-linux 2.37 + Enroot switchroot:

  nsenter -t KEEPER -U --preserve-credentials -n \\
    nsenter -t APP -m --wd=/ \\
      /bin/sh -c 'cd / && PATH=<container> LC_ALL=C exec "$@"' sh ARGS...

For ordinary tools retain the legacy namespace recipe for PostgreSQL's
``/proc/self/exe`` lookup. SQLCMD instead requires target root and PID membership
to load its resource files; it uses an explicit command-scoped exception.
Do not inherit the host PATH: after ``-m`` those
directories are gone and ``psql`` is found via ``/usr/bin`` / ``/usr/lib/postgresql/*/bin``.
"""

from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path


_CONTAINER_PATH = (
    "/saas-bin:/usr/lib/postgresql/16/bin:/usr/lib/postgresql/15/bin:"
    "/usr/lib/postgresql/14/bin:/usr/lib/postgresql/13/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)


def die(message: str, code: int = 2) -> int:
    print(f"docker-enroot-shim: {message}", file=sys.stderr)
    return code


def container_path(spec: str) -> tuple[str, str] | None:
    if ":" not in spec:
        return None
    container, path = spec.split(":", 1)
    if not container or not path.startswith("/"):
        return None
    return container, path


def rootfs_path(container: str, path: str) -> Path:
    base = Path(
        os.environ.get(
            "ENROOT_DATA_PATH", f"/raid/containers/data/user-{os.getuid()}"
        )
    ) / container
    target = (base / path.lstrip("/")).resolve()
    if base.resolve() not in target.parents and target != base.resolve():
        raise ValueError(f"path escapes container rootfs: {path}")
    return target


def copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        if destination.exists() and destination.is_dir():
            destination = destination / source.name
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        if destination.exists() and destination.is_dir():
            destination = destination / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def docker_cp(args: list[str]) -> int:
    if len(args) != 2:
        return die("usage: docker cp CONTAINER:SRC DEST (or SRC CONTAINER:DEST)")
    src_spec = container_path(args[0])
    dst_spec = container_path(args[1])
    if bool(src_spec) == bool(dst_spec):
        return die("docker cp requires exactly one container path")
    try:
        if src_spec:
            container, path = src_spec
            copy_path(rootfs_path(container, path), Path(args[1]))
        else:
            assert dst_spec is not None
            container, path = dst_spec
            copy_path(Path(args[0]), rootfs_path(container, path))
    except (OSError, ValueError) as exc:
        return die(f"cp failed: {exc}", 1)
    return 0


def _read_pid(path: Path) -> int | None:
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _proc_children() -> dict[int, list[int]]:
    tree: dict[int, list[int]] = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return tree
    for name in names:
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as handle:
                stat = handle.read()
        except OSError:
            continue
        rparen = stat.rfind(")")
        if rparen < 0:
            continue
        fields = stat[rparen + 2 :].split()
        if len(fields) < 2:
            continue
        try:
            ppid = int(fields[1])
        except ValueError:
            continue
        tree.setdefault(ppid, []).append(pid)
    return tree


def _descendants(root_pid: int) -> list[int]:
    tree = _proc_children()
    found: list[int] = []
    stack = [root_pid]
    seen = {root_pid}
    while stack:
        pid = stack.pop()
        for child in tree.get(pid, []):
            if child not in seen:
                seen.add(child)
                found.append(child)
                stack.append(child)
    return found


def _private_mnt_pid(wrap_pid: int) -> int | None:
    try:
        wrap_mnt = os.readlink(f"/proc/{wrap_pid}/ns/mnt")
    except OSError:
        return None
    for pid in _descendants(wrap_pid):
        try:
            mnt = os.readlink(f"/proc/{pid}/ns/mnt")
        except OSError:
            continue
        if mnt != wrap_mnt:
            return pid
    return None


def _keeper_pid(container: str, state_dir: Path) -> int | None:
    pid = _read_pid(state_dir / f"{container}.enroot.pid")
    if pid:
        return pid
    name = container
    while "-" in name:
        name, _, _ = name.rpartition("-")
        pid = _read_pid(state_dir / f"{name}.enroot.pid")
        if pid:
            return pid
    return None


def resolve_exec_pids(container: str) -> tuple[int, int] | str:
    state_dir = Path(
        os.environ.get("SAAS_ENROOT_STATE_DIR", "/tmp/saas_enroot_state")
    )
    keeper = _keeper_pid(container, state_dir)
    app = _read_pid(state_dir / f"{container}.app.pid")
    wrap = _read_pid(state_dir / f"{container}.wrap.pid")
    if wrap and (not app or app == keeper):
        scanned = _private_mnt_pid(wrap)
        if scanned:
            app = scanned
    if not _alive(keeper):
        return f"no live Enroot keeper for {container} under {state_dir}"
    if not _alive(app):
        return f"no live Enroot app PID for {container} under {state_dir}"
    return keeper, app


def docker_inspect(args: list[str]) -> int:
    """Expose only the live service start time required by stock audits."""
    if len(args) != 3 or args[0] not in {"--format", "-f"} or args[1].replace(" ", "") != "{{.State.StartedAt}}":
        return die("supported inspect usage: docker inspect --format '{{.State.StartedAt}}' CONTAINER")
    resolved = resolve_exec_pids(args[2])
    if isinstance(resolved, str):
        return die(resolved)
    _, app_pid = resolved
    try:
        # Kernel process birth is stable across inspections. Never substitute
        # the audit time: the verifier needs the task-start stock boundary.
        stat = Path(f"/proc/{app_pid}/stat").read_text()
        ticks = int(stat[stat.rfind(")") + 2 :].split()[19])
        boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
        started = datetime.datetime.fromtimestamp(boot + ticks / os.sysconf("SC_CLK_TCK"), datetime.timezone.utc)
    except (OSError, ValueError, IndexError, StopIteration) as exc:
        return die(f"cannot read service start time: {exc}", 1)
    print(started.isoformat(timespec="microseconds").replace("+00:00", "Z"))
    return 0


def docker_exec(args: list[str]) -> int:
    env_values: list[str] = []
    interactive = False
    while args and args[0].startswith("-"):
        opt = args.pop(0)
        if opt in {"-i", "--interactive"}:
            interactive = True
        elif opt in {"-t", "--tty"}:
            continue
        elif opt in {"-e", "--env"}:
            if not args:
                return die(f"{opt} requires KEY=VALUE")
            env_values.append(args.pop(0))
        elif opt.startswith("--env="):
            env_values.append(opt.split("=", 1)[1])
        elif opt in {"-u", "--user", "-w", "--workdir"}:
            if not args:
                return die(f"{opt} requires a value")
            args.pop(0)
        else:
            return die(f"unsupported docker exec option: {opt}")

    if len(args) < 2:
        return die("usage: docker exec [options] CONTAINER COMMAND [ARG...]")
    container = args.pop(0)
    resolved = resolve_exec_pids(container)
    if isinstance(resolved, str):
        return die(resolved)
    keeper_pid, app_pid = resolved

    try:
        os.chdir("/")
    except OSError:
        pass

    # Namespace entry alone does not inherit the service's UID virtualization.
    # Copy only runtime flags, never its full environment or credentials.
    runtime_env = []
    allowed = {'LD_PRELOAD', 'SAAS_FAKE_ID', 'SAAS_FORCE_UID',
               'SAAS_FAKE_OWNER_PATHS', 'SAAS_FAKE_OWNER_UID', 'SAAS_FAKE_OWNER_GID'}
    try:
        entries = Path(f'/proc/{app_pid}/environ').read_bytes().split(b'\0')
        runtime_env = [entry.decode() for entry in entries if entry.split(b'=', 1)[0].decode(errors='replace') in allowed]
    except OSError:
        pass
    # SQLCMD resolves resource files through /proc/self/exe, which requires
    # membership in the target PID namespace as well as its root filesystem.
    sqlcmd = Path(args[0]).name == "sqlcmd"
    # PHP must start inside the service root too. A host-root cwd becomes
    # unreachable after namespace entry and causes shell/PHP startup failures.
    # Resolve a directory FD to the target root before switching namespaces;
    # unlike the service cwd, this remains valid if its work directory is gone.
    rooted = sqlcmd or Path(args[0]).name == "php"
    namespace_flags = (["--pid"] if sqlcmd else []) + ["--root", f"--wd=/proc/{app_pid}/root"] if rooted else ["--wd=/"]
    payload = ["env", *runtime_env, *env_values, *args]
    inner = [
        "/bin/sh",
        "-c",
        (
            "cd / && "
            f"export PATH={_CONTAINER_PATH} LC_ALL=C LANG=C LANGUAGE=C && "
            'exec "$@"'
        ),
        "sh",
        *payload,
    ]
    cmd = [
        "nsenter",
        "-t",
        str(keeper_pid),
        "-U",
        "--preserve-credentials",
        "-n",
        "nsenter",
        "-t",
        str(app_pid),
        "-m",
        *namespace_flags,
        *inner,
    ]
    result = subprocess.run(cmd, stdin=None if interactive else sys.stdin.buffer, env={"PATH": os.environ.get("PATH", "")})
    return result.returncode


def main(argv: list[str]) -> int:
    if not argv:
        return die("only `docker exec`, `docker cp` and start-time `docker inspect` are supported")
    if argv[0] == "inspect":
        return docker_inspect(argv[1:])
    if argv[0] == "cp":
        return docker_cp(argv[1:])
    if argv[0] != "exec":
        return die(f"unsupported docker command: {argv[0]}")
    return docker_exec(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
