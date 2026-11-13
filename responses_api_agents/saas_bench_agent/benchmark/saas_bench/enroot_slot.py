# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/saas_bench/enroot_slot.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""Enroot-backed slot helpers for GPU nodes (no Podman/Docker).

Activated when ``SAAS_NRT_ENROOT=1``.

Constraints on NRT interactive GPU enroot (3.5.0 as deployed):
  - Enroot itself has no ``--net`` / ``--pid`` flags.
  - We create rootless user+network+PID namespaces with ``unshare``. This
    isolates common ports across slots and lets s6-overlay run as PID 1.
  - A user-local ``slirp4netns`` gives each app outbound network/DNS access.
  - Host ``socat`` enters that namespace with ``nsenter`` and publishes the
    slot's host port.
  - The cluster bind-mounts ``/etc/enroot/entrypoint`` over image
    ``/etc/rc.local`` and defaults to ``/bin/bash``. We override that mount with
    an empty file and disable login-shell mode so Alpine/BusyBox images use
    their original image entrypoint.
  - GPU nodes have no ``/etc/subuid``. An ``LD_PRELOAD`` shim fakes
    ``setuid``/``geteuid``/``stat`` ownership so mysql/postgres/nginx can start
    without extra UIDs.
  - Compose apps (pretix / onlyoffice / mattermost / owncloud) share one
    rootless netns; service hostnames are rewritten to 127.0.0.1.

Images: ``$SAAS_ENROOT_IMAGE_DIR/<name>.sqsh`` from
``scripts/nrt/convert_images_to_enroot.sh`` (run on cpu_short).
"""

from __future__ import annotations

import os
import json
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path


_ENROOT = os.environ.get("SAAS_NRT_ENROOT", "").lower() in {"1", "true", "yes"}
_IMAGE_DIR = Path(
    os.environ.get(
        "SAAS_ENROOT_IMAGE_DIR",
        "/saas-images",
    )
)
_STATE_DIR = Path(
    os.environ.get(
        "SAAS_ENROOT_STATE_DIR",
        os.path.join(os.environ.get("TMPDIR", "/tmp"), "saas_enroot_state"),
    )
)
_PUBLISH_BIND = os.environ.get("SAAS_NRT_PUBLISH_BIND", "127.0.0.1")
_HOST_UID = os.getuid()
_ENROOT_DATA_PATH = os.environ.get(
    "ENROOT_DATA_PATH", f"/raid/containers/data/user-{_HOST_UID}"
)
_ENROOT_CACHE_PATH = os.environ.get(
    "ENROOT_CACHE_PATH", f"/raid/containers/cache/group-{_HOST_UID}"
)
_ENROOT_RUNTIME_PATH = os.environ.get(
    "ENROOT_RUNTIME_PATH", f"/raid/containers/runtime/user-{_HOST_UID}"
)
_ENROOT_TEMP_PATH = os.environ.get("ENROOT_TEMP_PATH", "/raid/containers/tmp")
_ENROOT_RUNTIME_DIR = Path(
    os.environ.get(
        "SAAS_ENROOT_RUNTIME_DIR",
        "/saas-enroot-runtime",
    )
)
_SLIRP_BIN = _ENROOT_RUNTIME_DIR / "usr/bin/slirp4netns"
_SLIRP_LIB = _ENROOT_RUNTIME_DIR / "usr/lib/x86_64-linux-gnu"
_UID_SHIM = Path(
    os.environ.get(
        "SAAS_UID_SHIM",
        str(Path(__file__).resolve().parent / "saas-uid-shim.so"),
    )
)
_UID_SHIM_CTR = "/saas-uid-shim.so"
_UID_WRAPPERS = Path(__file__).resolve().parent / "uid_wrappers"
_UID_SHIM_SRC = Path(__file__).resolve().parent / "uid_shim.c"


def _distroless_shell() -> Path:
    """Resolve the provisioned static shell without depending on login-node packages."""
    explicit = os.environ.get("SAAS_STATIC_SHELL")
    candidates = [Path(explicit)] if explicit else [
        _ENROOT_RUNTIME_DIR / "usr/bin/busybox",
        _ENROOT_RUNTIME_DIR / "bin/busybox",
        Path("/bin/busybox"),
    ]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise RuntimeError(
        "distroless service requires executable static BusyBox; provision "
        "SAAS_ENROOT_RUNTIME_DIR/usr/bin/busybox or set SAAS_STATIC_SHELL "
        f"(checked: {', '.join(map(str, candidates))})"
    )


class EnrootPortBusy(RuntimeError):
    """The Enroot host-side publish port cannot be claimed.

    ``SlotManager`` handles this separately from an app-start failure: it
    tears down the partial slot and retries the whole slot on a fresh port
    epoch.  Keep this exception in the Enroot module to avoid a circular
    dependency on ``slot.py``.
    """


def _ensure_uid_shim() -> None:
    src, so = _UID_SHIM_SRC, _UID_SHIM
    if not src.is_file():
        return
    if so.is_file() and so.stat().st_mtime >= src.stat().st_mtime:
        return
    so.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=so.parent, suffix=".so", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        r = subprocess.run(
            ["gcc", "-shared", "-fPIC", "-nostdlib", "-o", str(temporary), str(src)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"gcc uid shim failed: {r.stderr or r.stdout}")
        # Parallel app starts can compile together. Never truncate a library
        # that another container is loading or already has mapped.
        temporary.replace(so)
    finally:
        temporary.unlink(missing_ok=True)


def enroot_enabled() -> bool:
    return _ENROOT


def _state_paths(name: str) -> dict[str, Path]:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "enroot_pid": _STATE_DIR / f"{name}.enroot.pid",
        "app_pid": _STATE_DIR / f"{name}.app.pid",
        "socat_pid": _STATE_DIR / f"{name}.socat.pid",
        "socat_log": _STATE_DIR / f"{name}.socat.log",
        "loopback_pid": _STATE_DIR / f"{name}.loopback.pid",
        "loopback_log": _STATE_DIR / f"{name}.loopback.log",
        "slirp_pid": _STATE_DIR / f"{name}.slirp.pid",
        "slirp_log": _STATE_DIR / f"{name}.slirp.log",
        "net_ready": _STATE_DIR / f"{name}.net.ready",
        "resolv": _STATE_DIR / f"{name}.resolv.conf",
        "hosts": _STATE_DIR / f"{name}.hosts",
        "svcs": _STATE_DIR / f"{name}.svcs",
        "svc_pids": _STATE_DIR / f"{name}.svc.pids",
        "log": _STATE_DIR / f"{name}.log",
    }

def _net_file_mounts(paths: dict[str, Path]) -> list[str]:
    """resolv.conf plus a hosts file that maps the node name to loopback.

    The container keeps the node's hostname but slirp has no DNS entry for it,
    so apps that resolve their own hostname (metabase's scheduler instance id)
    fail to start.
    """
    paths["resolv"].write_text("nameserver 10.0.2.3\noptions timeout:2 attempts:2\n")
    paths["hosts"].write_text(
        f"127.0.0.1\tlocalhost {socket.gethostname()}\n"
        "::1\tlocalhost ip6-localhost ip6-loopback\n"
    )
    return [
        "-m",
        f"{paths['resolv']}:/etc/resolv.conf:none:bind,ro",
        "-m",
        f"{paths['hosts']}:/etc/hosts:none:bind,ro",
    ]


def _svc_log_path(svc_name: str) -> Path:
    return _STATE_DIR / f"{svc_name}.log"


def compose_volume_root(name: str) -> Path:
    """Keep ephemeral app databases off the persistent diagnostic filesystem."""
    base = Path(os.environ.get("SAAS_ENROOT_VOLUME_DIR", str(Path(_ENROOT_DATA_PATH) / ".saas-volumes")))
    return base / name


def _populate_named_volumes(rootfs: Path, volumes: list[dict]) -> None:
    """Match Docker's initial named-volume copy-up, preserving benchmark seeds."""
    for volume in volumes:
        source = (rootfs / volume["source"].lstrip("/")).resolve()
        source.relative_to(rootfs.resolve())
        target = Path(volume["target"])
        if not source.is_dir() or not any(source.iterdir()):
            continue
        if any(target.iterdir()):
            continue
        try:
            shutil.copytree(source, target, symlinks=True, dirs_exist_ok=True)
        except BaseException:
            # A partial seed must never be mistaken for a valid resumed volume.
            shutil.rmtree(target)
            target.mkdir(parents=True)
            raise


def _ordered_compose_services(services: list[dict]) -> list[dict]:
    """Respect Compose dependency ordering before any container is launched."""
    names = {svc["name"] for svc in services}
    if len(names) != len(services):
        raise ValueError("Duplicate compose service name")
    ordered, pending = [], list(services)
    while pending:
        ready = [svc for svc in pending if all(
            dep["name"] in {item["name"] for item in ordered}
            for dep in svc.get("depends_on", [])
        )]
        if not ready:
            raise ValueError("Unknown or cyclic compose dependency")
        for svc in ready:
            for dep in svc.get("depends_on", []):
                if dep.get("condition", "service_started") not in {"service_started", "service_healthy"}:
                    raise ValueError(f"Unsupported compose dependency condition: {dep['condition']}")
            ordered.append(svc)
            pending.remove(svc)
    return ordered


def _wait_compose_dependency(svc: dict, child: subprocess.Popen, keeper: subprocess.Popen,
                             *, timeout: float, label: str = "") -> None:
    """Run the image's actual healthcheck inside its mount and network namespaces."""
    test = svc.get("healthcheck") or []
    if not test or test[0] not in {"CMD", "CMD-SHELL"}:
        raise RuntimeError(f"{label}{svc['name']}: service_healthy requires a healthcheck")
    command = ["/bin/sh", "-c", test[1]] if test[0] == "CMD-SHELL" else test[1:]
    deadline = time.monotonic() + timeout
    last_error = "container mount namespace is not ready"
    while time.monotonic() < deadline:
        if keeper.poll() is not None or child.poll() is not None:
            raise RuntimeError(f"{label}{svc['name']} exited before becoming healthy\n" +
                               compose_log_tail([svc['name']], per_svc=2000))
        mount_pid = _descendant_in_private_mnt(child.pid)
        if mount_pid:
            cmd = ["nsenter", "-t", str(keeper.pid), "-U", "--preserve-credentials", "-n",
                   "nsenter", "-t", str(mount_pid), "-m", "--wd=/", "/bin/sh", "-c",
                   'cd / && export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin LC_ALL=C && exec "$@"',
                   "saas-healthcheck", *command]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True,
                                        timeout=min(5, max(0.1, deadline-time.monotonic())))
                if result.returncode == 0:
                    return
                last_error = (result.stderr or result.stdout).strip()[-300:]
            except subprocess.TimeoutExpired:
                last_error = "healthcheck timed out"
        time.sleep(0.5)
    raise RuntimeError(f"{label}{svc['name']} did not become healthy in {timeout}s: {last_error}")


def compose_log_tail(svc_names: list[str], per_svc: int = 600) -> str:
    """Per-service log tails, so a failure names the service that broke."""
    chunks = []
    for svc_name in svc_names:
        try:
            text = _svc_log_path(svc_name).read_text(errors="replace")
        except OSError:
            continue
        chunks.append(f"--- {svc_name} ---\n{text[-per_svc:]}")
    return "\n".join(chunks)


def _read_pid(path: Path) -> int | None:
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def _kill_pid(pid: int | None, sig: int = signal.SIGTERM) -> None:
    if not pid:
        return
    try:
        os.kill(pid, sig)
    except OSError:
        pass


def _kill_process_group(pid: int | None, sig: int = signal.SIGTERM) -> None:
    """Signal a keeper and every Enroot/app process in its session."""
    if not pid:
        return
    try:
        os.killpg(pid, sig)
    except OSError:
        _kill_pid(pid, sig)


def _image_ref_from_start(start_cmd: str) -> str:
    m = re.search(r"((?:mw-[\w-]+|code-server-bundle):latest)\b", start_cmd)
    if not m:
        raise RuntimeError(f"cannot parse image from start cmd: {start_cmd[:120]}")
    return m.group(1)


def _sqsh_for_image(image_ref: str) -> Path:
    repo_tag = image_ref.split("/")[-1]
    repo, _, tag = repo_tag.partition(":")
    candidates = [_IMAGE_DIR / f"{repo}.sqsh"]
    if tag:
        safe_tag = tag.replace(":", "-")
        candidates.append(_IMAGE_DIR / f"{repo}-{safe_tag}.sqsh")
        candidates.append(_IMAGE_DIR / f"{repo}_{safe_tag}.sqsh")
    for sqsh in candidates:
        if sqsh.is_file():
            return sqsh
    raise RuntimeError(
        f"missing enroot image for {image_ref} (tried {candidates}; "
        f"run scripts/nrt/convert_images_to_enroot.sh on cpu_short; "
        f"SAAS_ENROOT_IMAGE_DIR={_IMAGE_DIR})"
    )


def _parse_env_and_mounts(start_cmd: str, fmt: dict) -> tuple[list[str], list[str], list[str]]:
    cmd = start_cmd.format(**fmt) if "{" in start_cmd else start_cmd
    envs: list[str] = []
    mounts: list[str] = []
    for m in re.finditer(r"-e\s+([^\s]+)", cmd):
        envs.extend(["-e", m.group(1)])
    for m in re.finditer(r"-v\s+([^\s]+)", cmd):
        spec = m.group(1)
        parts = spec.split(":")
        if len(parts) < 2:
            continue
        host, ctr = parts[0], parts[1]
        create = "x-create=dir" if Path(host).is_dir() else "x-create=file"
        opts = f"{create},bind"
        if len(parts) >= 3 and parts[2] == "ro":
            opts += ",ro"
        mounts.extend(["-m", f"{host}:{ctr}:none:{opts}"])
    extra: list[str] = []
    ep = re.search(r"--entrypoint\s+(\S+)\s+\S+:latest\s+(.+)$", cmd)
    if ep:
        extra = [ep.group(1), *ep.group(2).split()]
    return envs, mounts, extra


def _jvm_env_for_image(image: str, envs: list[str]) -> list[str]:
    """Avoid broken cgroup discovery in the bundled BookLore/Elasticsearch JVMs.

    Elasticsearch strips JAVA_TOOL_OPTIONS, so use JDK_JAVA_OPTIONS for its
    launcher JVMs and server. ES_JAVA_OPTS alone is applied too late for its
    memory probe. This workaround is scoped to Enroot and these two images.
    """
    repo = image.rsplit('/', 1)[-1].split(':', 1)[0]
    result = list(envs)
    if repo not in {'mw-booklore', 'oo-bundle-es'}:
        return result
    key = 'JDK_JAVA_OPTIONS' if repo == 'oo-bundle-es' else 'JAVA_TOOL_OPTIONS'
    index = next((i for i in range(len(result) - 1, -1, -1)
                  if result[i].startswith(key + '=')), None)
    options = result[index].split('=', 1)[1] if index is not None else ''
    tokens = options.split()
    container_flags = [t for t in tokens if t in {'-XX:+UseContainerSupport', '-XX:-UseContainerSupport'}]
    if not container_flags or container_flags[-1] != '-XX:-UseContainerSupport':
        options += ' -XX:-UseContainerSupport'
    # Without container ergonomics, cap BookLore's default heap calculation
    # at 8 GiB RAM (normally a 2 GiB max heap), rather than the whole HPC node.
    # An explicitly configured heap/RAM limit is retained.
    if repo == 'mw-booklore' and not any(
        token.startswith(('-Xmx', '-XX:MaxRAM=', '-XX:MaxHeapSize=')) for token in tokens
    ):
        options += ' -XX:MaxRAM=8g'
    value = key + '=' + options.strip()
    if index is None:
        result.extend(['-e', value])
    else:
        result[index] = value
    return result


def _port_listening(port: int) -> bool:
    r = subprocess.run(
        ["ss", "-Hltn", f"sport = :{port}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(r.stdout.strip())


def _socat_publish_error(
    *, name: str, host_port: int, label: str, stderr: str
) -> RuntimeError:
    """Classify a failed host forwarder without hiding genuine socat errors."""
    detail = (stderr or "").strip()
    message = f"{label}socat failed to publish :{host_port} for {name}"
    if detail:
        message = f"{message}: {detail}"
    if _port_listening(host_port) or "address already in use" in detail.lower():
        return EnrootPortBusy(message)
    return RuntimeError(message)


def _start_forwarder(command: list[str], pid_file: Path, log_file: Path) -> subprocess.Popen:
    """Keep late forwarding errors off an abandoned pipe and retain diagnostics."""
    with log_file.open("ab") as stream:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stream,
                                   start_new_session=True)
    pid_file.write_text(str(process.pid))
    return process


def _start_loopback_alias(name: str, keeper_pid: int, host_port: int, container_port: int) -> None:
    """Let an app reach its own public localhost URL from its private namespace."""
    if host_port == container_port:
        return
    paths = _state_paths(name)
    process = _start_forwarder(
        ["nsenter", "-t", str(keeper_pid), "-U", "--preserve-credentials", "-n",
         "socat", f"TCP-LISTEN:{host_port},bind=127.0.0.1,reuseaddr,fork",
         f"TCP:127.0.0.1:{container_port}"], paths["loopback_pid"], paths["loopback_log"])
    time.sleep(0.1)
    if process.poll() is not None:
        detail = paths["loopback_log"].read_text(errors="replace")[-1000:]
        stop_enroot_app(name)
        raise RuntimeError(f"{name}: public-URL loopback alias failed: {detail}")


def _netns_port_listening(pid: int, port: int) -> bool:
    """Return whether ``port`` listens in the keeper's rootless netns."""
    r = subprocess.run(
        [
            "nsenter",
            "-t",
            str(pid),
            "-U",
            "--preserve-credentials",
            "-n",
            "ss",
            "-Hltn",
            f"sport = :{port}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(r.stdout.strip())


def wait_enroot_http_json(
    name: str, port: int, path: str, *, accepted_values: tuple,
    json_key: str | None = None, timeout: float = 180,
) -> None:
    """Wait for a backend inside a compose netns, rather than its frontend port."""
    deadline = time.monotonic() + timeout
    last_error = 'not probed'
    while time.monotonic() < deadline:
        pid = _read_pid(_state_paths(name)['enroot_pid'])
        if not pid:
            raise RuntimeError(f'{name}: backend keeper exited before {port}{path} was ready')
        try:
            result = subprocess.run(
                ['nsenter', '-t', str(pid), '-U', '--preserve-credentials', '-n',
                 'curl', '-fsS', '--max-time', '3', f'http://127.0.0.1:{port}{path}'],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0:
                value = json.loads(result.stdout)
                if json_key is not None:
                    value = value.get(json_key) if isinstance(value, dict) else None
                if any(type(value) is type(expected) and value == expected for expected in accepted_values):
                    return
                last_error = f'unhealthy response: {value!r}'
            else:
                last_error = result.stderr.strip()[-300:]
        except (ValueError, subprocess.TimeoutExpired) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f'{name}: backend {port}{path} not ready in {timeout}s ({last_error})')


def _netns_interface_ready(pid: int, interface: str) -> bool:
    r = subprocess.run(
        [
            "nsenter",
            "-t",
            str(pid),
            "-U",
            "--preserve-credentials",
            "-n",
            "ip",
            "-4",
            "addr",
            "show",
            interface,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.returncode == 0 and "inet " in r.stdout


def _private_namespaces_ready(pid: int) -> bool:
    """Return once unshare has moved the keeper out of the host user/net ns."""
    try:
        return (
            os.readlink(f"/proc/{pid}/ns/user") != os.readlink("/proc/self/ns/user")
            and os.readlink(f"/proc/{pid}/ns/net") != os.readlink("/proc/self/ns/net")
        )
    except OSError:
        return False


def _child_pid(parent_pid: int) -> int | None:
    """Return the direct child of the unshare keeper, if it has started."""
    r = subprocess.run(
        ["ps", "--ppid", str(parent_pid), "-o", "pid="],
        capture_output=True,
        text=True,
        check=False,
    )
    for value in r.stdout.split():
        if value.isdigit():
            return int(value)
    return None


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


def _descendant_in_private_mnt(wrap_pid: int) -> int | None:
    """Return a descendant that switched mount namespace (Enroot switchroot)."""
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


def _wait_private_mnt_pid(wrap_pid: int, timeout: float = 25.0) -> int | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        pid = _descendant_in_private_mnt(wrap_pid)
        if pid:
            return pid
        time.sleep(0.2)
    return None


def _write_exec_pids(
    container: str,
    keeper_pid: int,
    app_pid: int | None,
    wrap_pid: int | None = None,
) -> None:
    """PID files consumed by ``scripts/nrt/docker_enroot_shim.py``."""
    paths = _state_paths(container)
    paths["enroot_pid"].write_text(str(keeper_pid))
    if app_pid:
        paths["app_pid"].write_text(str(app_pid))
    if wrap_pid:
        (_STATE_DIR / f"{container}.wrap.pid").write_text(str(wrap_pid))


def _ensure_created(name: str, sqsh: Path) -> None:
    listed = subprocess.run(
        ["enroot", "list"], capture_output=True, text=True, check=False
    )
    names = {ln.strip() for ln in listed.stdout.splitlines() if ln.strip()}
    if name not in names:
        r = subprocess.run(
            ["enroot", "create", "--name", name, str(sqsh)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0 and "already exists" not in (r.stderr + r.stdout):
            raise RuntimeError(f"enroot create {name} failed: {r.stderr or r.stdout}")
    # Docker VOLUME declarations are also baked into the imported image's
    # /etc/fstab. Filtering compose's -m flags alone cannot remove these.
    # Modify this disposable rootfs, never the shared .sqsh image.
    fstab = Path(_ENROOT_DATA_PATH) / name / 'etc/fstab'
    if fstab.is_file():
        original = fstab.read_text()
        lines = []
        for line in original.splitlines(keepends=True):
            fields = line.split()
            if len(fields) >= 2 and not line.lstrip().startswith('#'):
                target = fields[1].rstrip('/')
                if target == '/sys/fs/cgroup' or target.startswith('/sys/fs/cgroup/'):
                    continue
            lines.append(line)
        repaired = ''.join(lines)
        if repaired != original:
            fstab.write_text(repaired)
            print(f'  enroot {name}: omitted image cgroup volume from /etc/fstab', flush=True)


def _uid_runtime_env_mounts(
    envs: list[str], preload_shim: bool = True
) -> tuple[list[str], list[str]]:
    """Return env/mount flags for the subuid shim, su wrappers, and rc.local.

    preload_shim=False keeps the wrappers and rc.local but leaves LD_PRELOAD
    alone, for images that refuse to run with an interposed library.
    """
    from .runtime_assets import validate_runtime_assets
    validate_runtime_assets()
    mounts: list[str] = []
    envs = list(envs)
    bootstrap_rc = _STATE_DIR / "bootstrap-rc.local"
    bootstrap = (
        "mkdir -p /run /run/nginx /run/mysqld /run/postgresql /run/dovecot /var/run "
        "/run/apache2 /var/run/nginx /var/log/nginx /tmp "
        "/baserow/data/plugins || true\n"
        # Enroot cannot implement Docker's --ulimit flag, and image entrypoint
        # overrides arrive as CMD arguments. A normal 8 MiB stack prevents
        # SQL Server from selecting the legacy low-address VA layout.
        "ulimit -s 8192 2>/dev/null || true\n"
        "chmod 700 /etc/ssl/private /var/lib/postgresql /baserow/data/postgres "
        "2>/dev/null || true\n"
        "chmod 600 /etc/ssl/private/* /etc/ssl/private/ssl-cert-snakeoil.key "
        "2>/dev/null || true\n"
        "export PATH=/saas-bin:${PATH}\n"
    )
    # SlotManager starts apps concurrently. A bind-mounted script must remain
    # complete while other apps prepare their own mounts.
    if not bootstrap_rc.is_file() or bootstrap_rc.read_text() != bootstrap:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=bootstrap_rc.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(bootstrap)
        try:
            temporary.replace(bootstrap_rc)
        finally:
            temporary.unlink(missing_ok=True)
    mounts.extend(["-m", f"{bootstrap_rc}:/etc/rc.local:none:x-create=file,bind,ro"])
    if _UID_WRAPPERS.is_dir():
        mounts.extend(["-m", f"{_UID_WRAPPERS}:/saas-bin:none:x-create=dir,bind,ro"])
    _ensure_uid_shim()
    if _UID_SHIM.is_file() and preload_shim:
        mounts.extend(["-m", f"{_UID_SHIM}:{_UID_SHIM_CTR}:none:x-create=file,bind,ro"])
        preload = _UID_SHIM_CTR
        for i in range(0, len(envs) - 1, 2):
            if envs[i] == "-e" and envs[i + 1].startswith("LD_PRELOAD="):
                preload = f"{_UID_SHIM_CTR}:{envs[i + 1].split('=', 1)[1]}"
                envs[i + 1] = f"LD_PRELOAD={preload}"
                preload = ""
                break
        if preload:
            envs.extend(["-e", f"LD_PRELOAD={preload}"])
    return envs, mounts


def _require_enroot_bins() -> None:
    for bin_name in ("enroot", "socat", "ss", "unshare", "nsenter", "ip"):
        if subprocess.run(["which", bin_name], capture_output=True).returncode != 0:
            raise RuntimeError(f"enroot backend requires `{bin_name}` on PATH")


def start_enroot_app(
    *,
    name: str,
    start_cmd: str,
    container_port: int,
    host_port: int,
    fmt: dict,
    label: str = "",
    listen_timeout: float = 90.0,
) -> None:
    """Start an Enroot app in a private rootless netns and publish its port."""
    _require_enroot_bins()
    if _port_listening(host_port):
        raise EnrootPortBusy(
            f"{label}host publish port :{host_port} is already in use"
        )

    image = _image_ref_from_start(start_cmd)
    sqsh = _sqsh_for_image(image)
    envs, mounts, extra = _parse_env_and_mounts(start_cmd, fmt)
    envs = _jvm_env_for_image(image, envs)
    image_repo = image.split(":", 1)[0]
    if image_repo == "mw-baserow":
        envs.extend(["-e", "SAAS_FAKE_ID=1"])
    if image_repo == "mw-roundcubemail":
        # docker-mailserver defaults to eth0, while this rootless slirp netns
        # exposes tap0. It briefly serves HTTP and then shuts the whole stack
        # down when IP discovery fails unless these match the actual topology.
        envs.extend(
            [
                "-e", "NETWORK_INTERFACE=tap0",
                "-e", "PERMIT_DOCKER=none",
                # Dovecot creates /run/dovecot/login as root:dovecot.  The
                # rootless UID shim virtualizes the otherwise-unmapped chown;
                # make stat(2) report that same virtual group for this path.
                # Only the login socket directories are root:dovecot.  Other
                # Dovecot runtime directories intentionally remain root:root.
                "-e", "SAAS_FAKE_OWNER_PATHS=/run/dovecot/login,/run/dovecot/token-login",
                "-e", "SAAS_FAKE_OWNER_UID=0",
                "-e", "SAAS_FAKE_OWNER_GID=113",
            ]
        )
    if image_repo == "mw-recipya":
        # ENROOT_MOUNT_HOME=n deliberately unmounts the host home. Recipya
        # requires one of HOME/XDG_CONFIG_HOME even though it stores app data
        # elsewhere.
        envs.extend(["-e", "HOME=/root", "-e", "XDG_CONFIG_HOME=/root/.config"])
    paths = _state_paths(name)
    stop_enroot_app(name)

    _ensure_created(name, sqsh)

    # sqlservr rejects any interposed library: it validates its own address
    # space at start-up and aborts on the mapping LD_PRELOAD introduces.
    envs, uid_mounts = _uid_runtime_env_mounts(
        envs, preload_shim=image_repo != "mw-elabel"
    )
    mounts.extend(uid_mounts)
    mounts.extend(_net_file_mounts(paths))

    enroot_cmd = [
        "enroot",
        "start",
        "--root",
        "--rw",
        *envs,
        *mounts,
        name,
        *extra,
    ]
    # `unshare --fork` remains as a stable parent in the outer user+network
    # namespace while its child is PID 1 in a private PID namespace. Enroot
    # replaces that child, so s6-overlay images see their init as PID 1.
    # The unshare parent remains a stable nsenter target for checks and socat.
    keeper_script = """
set -e
ip link set lo up
# Permit unprivileged processes in this private netns to bind image ports such
# as 80. This sysctl is namespaced and does not modify the GPU host.
sysctl -q -w net.ipv4.ip_unprivileged_port_start=0
export ENROOT_LOGIN_SHELL=no
# Without this the host home is mounted over the container's, so writes an
# image expects to land in its rootfs hit the shared filesystem quota instead.
export ENROOT_MOUNT_HOME=n
export ENROOT_DATA_PATH="$1"
export ENROOT_CACHE_PATH="$2"
export ENROOT_RUNTIME_PATH="$3"
export ENROOT_TEMP_PATH="$4"
ready="$5"
shift 5
while [ ! -e "$ready" ]; do sleep 0.05; done
exec "$@"
"""
    cmd = [
        "unshare",
        "--user",
        "--map-root-user",
        "--net",
        "--pid",
        "--fork",
        "--mount-proc",
        "bash",
        "-c",
        keeper_script,
        "saas-enroot-keeper",
        _ENROOT_DATA_PATH,
        _ENROOT_CACHE_PATH,
        _ENROOT_RUNTIME_PATH,
        _ENROOT_TEMP_PATH,
        str(paths["net_ready"]),
        *enroot_cmd,
    ]
    logf = open(paths["log"], "wb")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    paths["enroot_pid"].write_text(str(proc.pid))
    print(
        f"  {label}enroot start {name} keeper={proc.pid} "
        f"ctr=:{container_port} publish=:{host_port}",
        flush=True,
    )

    namespace_deadline = time.time() + 10
    while time.time() < namespace_deadline:
        if proc.poll() is not None:
            logf.flush()
            detail = paths["log"].read_text(errors="replace")[-800:]
            stop_enroot_app(name)
            raise RuntimeError(f"{label}unshare failed for {name}\n{detail}")
        if _private_namespaces_ready(proc.pid):
            break
        time.sleep(0.05)
    else:
        stop_enroot_app(name)
        raise RuntimeError(f"{label}private namespaces did not start for {name}")

    if not _SLIRP_BIN.is_file():
        stop_enroot_app(name)
        raise RuntimeError(
            f"{label}missing slirp4netns runtime {_SLIRP_BIN}; install "
            f"slirp4netns + libslirp0 under SAAS_ENROOT_RUNTIME_DIR"
        )
    slirp_env = os.environ.copy()
    slirp_env["LD_LIBRARY_PATH"] = ":".join(
        filter(None, [str(_SLIRP_LIB), slirp_env.get("LD_LIBRARY_PATH", "")])
    )
    slirp_logf = open(paths["slirp_log"], "wb")  # noqa: SIM115
    slirp = subprocess.Popen(
        [
            str(_SLIRP_BIN),
            "--configure",
            "--mtu=65520",
            str(proc.pid),
            "tap0",
        ],
        stdout=slirp_logf,
        stderr=subprocess.STDOUT,
        env=slirp_env,
        start_new_session=True,
    )
    paths["slirp_pid"].write_text(str(slirp.pid))
    net_deadline = time.time() + 15
    while time.time() < net_deadline:
        if proc.poll() is not None or slirp.poll() is not None:
            slirp_logf.flush()
            detail = paths["slirp_log"].read_text(errors="replace")[-800:]
            stop_enroot_app(name)
            raise RuntimeError(f"{label}slirp setup failed for {name}\n{detail}")
        if _netns_interface_ready(proc.pid, "tap0"):
            break
        time.sleep(0.1)
    else:
        stop_enroot_app(name)
        raise RuntimeError(f"{label}slirp tap0 did not become ready for {name}")
    _start_loopback_alias(name, proc.pid, host_port, container_port)
    paths["net_ready"].touch()

    deadline = time.time() + listen_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            logf.flush()
            tail = paths["log"].read_text(errors="replace")[-800:]
            stop_enroot_app(name)
            raise RuntimeError(
                f"{label}enroot {name} exited early (rc={proc.returncode})\n{tail}"
            )
        if _netns_port_listening(proc.pid, container_port):
            break
        time.sleep(0.4)
    else:
        logf.flush()
        tail = paths["log"].read_text(errors="replace")[-800:]
        stop_enroot_app(name)
        raise RuntimeError(
            f"{label}enroot {name} did not listen on :{container_port}\n{tail}"
        )

    app_pid = _child_pid(proc.pid)
    if not app_pid:
        stop_enroot_app(name)
        raise RuntimeError(f"{label}could not resolve app PID for {name}")
    inner = _wait_private_mnt_pid(app_pid, timeout=15.0) if app_pid else None
    paths["app_pid"].write_text(str(inner or app_pid))

    # A private netns always needs publishing, even when host and container
    # port numbers happen to match.
    socat_cmd = [
        "socat",
        f"TCP-LISTEN:{host_port},bind={_PUBLISH_BIND},reuseaddr,fork",
        (
            f"EXEC:nsenter -t {proc.pid} -U --preserve-credentials -n "
            f"socat STDIO TCP\\:127.0.0.1\\:{container_port}"
        ),
    ]
    sp = _start_forwarder(socat_cmd, paths["socat_pid"], paths["socat_log"])
    time.sleep(0.2)
    if sp.poll() is not None:
        stderr = paths["socat_log"].read_text(errors="replace")[-2000:]
        stop_enroot_app(name)
        raise _socat_publish_error(
            name=name, host_port=host_port, label=label, stderr=stderr or ""
        )


def start_enroot_compose(
    *,
    name: str,
    services: list[dict],
    container_port: int,
    host_port: int,
    label: str = "",
    listen_timeout: float = 90.0,
) -> None:
    """Start compose services as Enroot containers sharing one rootless netns."""
    _require_enroot_bins()
    if _port_listening(host_port):
        raise EnrootPortBusy(f"{label}host publish port :{host_port} is already in use")
    if not services:
        raise RuntimeError(f"{label}empty compose for {name}")
    static_shell = _distroless_shell() if any(
        not svc.get("uid_runtime", True) for svc in services
    ) else None
    services = _ordered_compose_services(services)

    paths = _state_paths(name)
    paths["resolv"].write_text("nameserver 10.0.2.3\noptions timeout:2 attempts:2\n")
    svc_names = [str(svc["name"]) for svc in services]
    paths["svcs"].write_text("\n".join(svc_names) + "\n")

    for svc in services:
        _ensure_created(svc["name"], _sqsh_for_image(svc["image"]))
        _populate_named_volumes(Path(_ENROOT_DATA_PATH) / svc["name"], svc.get("volume_copies", []))

    keeper_script = """
set -e
ip link set lo up
sysctl -q -w net.ipv4.ip_unprivileged_port_start=0
ready="$1"
while [ ! -e "$ready" ]; do sleep 0.05; done
exec sleep infinity
"""
    cmd = [
        "unshare",
        "--user",
        "--map-root-user",
        "--net",
        "--fork",
        "bash",
        "-c",
        keeper_script,
        "saas-enroot-compose-keeper",
        str(paths["net_ready"]),
    ]
    logf = open(paths["log"], "wb")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    paths["enroot_pid"].write_text(str(proc.pid))
    print(
        f"  {label}enroot compose {name} keeper={proc.pid} "
        f"svcs={len(services)} ctr=:{container_port} publish=:{host_port}",
        flush=True,
    )

    namespace_deadline = time.time() + 10
    while time.time() < namespace_deadline:
        if proc.poll() is not None:
            logf.flush()
            detail = paths["log"].read_text(errors="replace")[-800:]
            stop_enroot_app(name)
            raise RuntimeError(f"{label}unshare failed for {name}\n{detail}")
        if _private_namespaces_ready(proc.pid):
            break
        time.sleep(0.05)
    else:
        stop_enroot_app(name)
        raise RuntimeError(f"{label}private namespaces did not start for {name}")

    if not _SLIRP_BIN.is_file():
        stop_enroot_app(name)
        raise RuntimeError(
            f"{label}missing slirp4netns runtime {_SLIRP_BIN}; install "
            f"slirp4netns + libslirp0 under SAAS_ENROOT_RUNTIME_DIR"
        )
    slirp_env = os.environ.copy()
    slirp_env["LD_LIBRARY_PATH"] = ":".join(
        filter(None, [str(_SLIRP_LIB), slirp_env.get("LD_LIBRARY_PATH", "")])
    )
    slirp_logf = open(paths["slirp_log"], "wb")  # noqa: SIM115
    slirp = subprocess.Popen(
        [
            str(_SLIRP_BIN),
            "--configure",
            "--mtu=65520",
            str(proc.pid),
            "tap0",
        ],
        stdout=slirp_logf,
        stderr=subprocess.STDOUT,
        env=slirp_env,
        start_new_session=True,
    )
    paths["slirp_pid"].write_text(str(slirp.pid))
    net_deadline = time.time() + 15
    while time.time() < net_deadline:
        if proc.poll() is not None or slirp.poll() is not None:
            slirp_logf.flush()
            detail = paths["slirp_log"].read_text(errors="replace")[-800:]
            stop_enroot_app(name)
            raise RuntimeError(f"{label}slirp setup failed for {name}\n{detail}")
        if _netns_interface_ready(proc.pid, "tap0"):
            break
        time.sleep(0.1)
    else:
        stop_enroot_app(name)
        raise RuntimeError(f"{label}slirp tap0 did not become ready for {name}")
    _start_loopback_alias(name, proc.pid, host_port, container_port)
    paths["net_ready"].touch()

    inner = """
set -e
export ENROOT_LOGIN_SHELL=no
export ENROOT_MOUNT_HOME=n
export ENROOT_DATA_PATH="$1"
export ENROOT_CACHE_PATH="$2"
export ENROOT_RUNTIME_PATH="$3"
export ENROOT_TEMP_PATH="$4"
shift 4
exec "$@"
"""
    svc_pids: list[str] = []
    svc_logs = []
    children: dict[str, subprocess.Popen] = {}
    launched: dict[str, dict] = {}
    healthy: set[str] = set()
    for svc in services:
        for dependency in svc.get("depends_on", []):
            dep_name = dependency["name"]
            if dependency.get("condition") == "service_healthy" and dep_name not in healthy:
                try:
                    _wait_compose_dependency(launched[dep_name], children[dep_name], proc,
                                             timeout=listen_timeout, label=label)
                except BaseException:
                    stop_enroot_app(name)
                    raise
                healthy.add(dep_name)
                print(f"  {label}dependency {dep_name} healthy", flush=True)
        envs = _jvm_env_for_image(svc['image'], list(svc.get("envs") or []))
        mounts = list(svc.get("mounts") or [])
        rootfs = Path(_ENROOT_DATA_PATH) / svc["name"]
        forced_uid = next(
            (
                item.split("=", 1)[1]
                for item in envs
                if item.startswith("SAAS_FORCE_UID=")
            ),
            "",
        )
        if forced_uid:
            passwd_src = rootfs / "etc/passwd"
            try:
                passwd = passwd_src.read_text()
            except OSError:
                passwd = ""
            if not any(
                len(fields := line.split(":")) > 2 and fields[2] == forced_uid
                for line in passwd.splitlines()
            ):
                passwd_file = _STATE_DIR / f"{svc['name']}.passwd"
                passwd_file.write_text(
                    passwd.rstrip("\n")
                    + f"\nsaas:x:{forced_uid}:{forced_uid}:SaaS user:/tmp:/bin/sh\n"
                )
                mounts.extend(
                    [
                        "-m",
                        f"{passwd_file}:/etc/passwd:none:x-create=file,bind,ro",
                    ]
                )
        if svc.get("patch_owncloud_console"):
            console_src = rootfs / "var/www/owncloud/console.php"
            console = console_src.read_text()
            needle = "if ($user['name'] !== $configUser['name']) {"
            if needle not in console:
                raise RuntimeError(
                    f"{label}ownCloud console ownership check changed in {console_src}"
                )
            console_file = _STATE_DIR / f"{svc['name']}.console.php"
            console_file.write_text(console.replace(needle, "if (false) {", 1))
            mounts.extend(
                [
                    "-m",
                    f"{console_file}:/var/www/owncloud/console.php:none:x-create=file,bind,ro",
                ]
            )
        if svc.get("uid_runtime", True):
            envs, uid_mounts = _uid_runtime_env_mounts(envs)
            mounts.extend(uid_mounts)
        else:
            empty_rc = _STATE_DIR / "empty-rc.local"
            empty_rc.write_text(":\n")
            mounts.extend(
                [
                    "-m",
                    f"{empty_rc}:/etc/rc.local:none:x-create=file,bind,ro",
                    "-m",
                    f"{static_shell}:/bin/sh:none:x-create=file,bind,ro",
                ]
            )
        mounts.extend(_net_file_mounts(paths))
        extra = list(svc.get("extra") or [])
        enroot_cmd = [
            "enroot",
            "start",
            "--root",
            "--rw",
            *envs,
            *mounts,
            svc["name"],
            *extra,
        ]
        svc_cmd = [
            "nsenter",
            "-t",
            str(proc.pid),
            "-U",
            "--preserve-credentials",
            "-n",
            "unshare",
            "--pid",
            "--fork",
            "--mount-proc",
            "bash",
            "-c",
            inner,
            "saas-enroot-svc",
            _ENROOT_DATA_PATH,
            _ENROOT_CACHE_PATH,
            _ENROOT_RUNTIME_PATH,
            _ENROOT_TEMP_PATH,
            *enroot_cmd,
        ]
        svc_logf = open(_svc_log_path(svc["name"]), "wb")  # noqa: SIM115
        svc_logs.append(svc_logf)
        child = subprocess.Popen(
            svc_cmd,
            stdout=svc_logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        svc_pids.append(str(child.pid))
        # Persist incrementally so a failed dependency gate cleans up every
        # service already launched, including the database that failed health.
        paths["svc_pids"].write_text("\n".join(svc_pids) + "\n")
        children[svc["name"]] = child
        launched[svc["name"]] = svc
        print(f"  {label}enroot svc {svc['name']} pid={child.pid}", flush=True)
        svc["wrap_pid"] = child.pid
    paths["svc_pids"].write_text("\n".join(svc_pids) + "\n")

    deadline = time.time() + listen_timeout
    while time.time() < deadline:
        exited = [svc_name for svc_name, child in children.items() if child.poll() is not None]
        if exited:
            tail = compose_log_tail(svc_names, per_svc=2000)
            stop_enroot_app(name)
            raise RuntimeError(f"{label}compose service exited before app readiness: {', '.join(exited)}\n{tail}")
        if proc.poll() is not None:
            for f in svc_logs:
                f.flush()
            tail = compose_log_tail(svc_names)
            stop_enroot_app(name)
            raise RuntimeError(
                f"{label}enroot compose {name} keeper exited (rc={proc.returncode})\n{tail}"
            )
        if _netns_port_listening(proc.pid, container_port):
            break
        time.sleep(0.4)
    else:
        for f in svc_logs:
            f.flush()
        tail = compose_log_tail(svc_names)
        stop_enroot_app(name)
        raise RuntimeError(
            f"{label}enroot compose {name} did not listen on :{container_port}\n{tail}"
        )

    paths["app_pid"].write_text(str(proc.pid))
    for svc in services:
        wrap_pid = int(svc.get("wrap_pid") or 0)
        inner = _wait_private_mnt_pid(wrap_pid, timeout=25.0) if wrap_pid else None
        _write_exec_pids(svc["name"], proc.pid, inner, wrap_pid=wrap_pid or None)
        if inner:
            print(
                f"  {label}exec target {svc['name']} keeper={proc.pid} app={inner}",
                flush=True,
            )
        else:
            print(
                f"  {label}warning: no private-mnt pid yet for {svc['name']}; "
                "docker exec will walk wrap.pid at verify time",
                flush=True,
            )
    for svc in services:
        if svc.get("ready_gate"):
            _wait_compose_dependency(svc, children[svc["name"]], proc,
                                     timeout=listen_timeout, label=label)
    socat_cmd = [
        "socat",
        f"TCP-LISTEN:{host_port},bind={_PUBLISH_BIND},reuseaddr,fork",
        (
            f"EXEC:nsenter -t {proc.pid} -U --preserve-credentials -n "
            f"socat STDIO TCP\\:127.0.0.1\\:{container_port}"
        ),
    ]
    sp = _start_forwarder(socat_cmd, paths["socat_pid"], paths["socat_log"])
    time.sleep(0.2)
    if sp.poll() is not None:
        stderr = paths["socat_log"].read_text(errors="replace")[-2000:]
        stop_enroot_app(name)
        raise _socat_publish_error(
            name=name, host_port=host_port, label=label, stderr=stderr or ""
        )


def stop_enroot_app(name: str) -> None:
    paths = _state_paths(name)
    _kill_process_group(_read_pid(paths["socat_pid"]), signal.SIGKILL)
    _kill_process_group(_read_pid(paths["loopback_pid"]), signal.SIGKILL)
    _kill_pid(_read_pid(paths["slirp_pid"]), signal.SIGKILL)
    try:
        extra_pids = [
            int(line.strip())
            for line in paths["svc_pids"].read_text().splitlines()
            if line.strip().isdigit()
        ]
    except (OSError, ValueError):
        extra_pids = []
    for pid in extra_pids:
        _kill_process_group(pid, signal.SIGTERM)
    time.sleep(0.2)
    for pid in extra_pids:
        _kill_process_group(pid, signal.SIGKILL)
    _kill_process_group(_read_pid(paths["enroot_pid"]), signal.SIGTERM)
    time.sleep(0.3)
    _kill_process_group(_read_pid(paths["enroot_pid"]), signal.SIGKILL)
    names = {name}
    try:
        names.update(
            line.strip() for line in paths["svcs"].read_text().splitlines() if line.strip()
        )
    except OSError:
        pass
    for ctr in names:
        subprocess.run(
            ["enroot", "remove", "-f", ctr],
            capture_output=True,
            text=True,
            check=False,
        )
        for extra_name in (f"{ctr}.app.pid", f"{ctr}.enroot.pid", f"{ctr}.wrap.pid"):
            try:
                (_STATE_DIR / extra_name).unlink()
            except OSError:
                pass
    for key in (
        "enroot_pid",
        "app_pid",
        "socat_pid",
        "loopback_pid",
        "slirp_pid",
        "net_ready",
        "svcs",
        "svc_pids",
    ):
        try:
            paths[key].unlink()
        except OSError:
            pass
    shutil.rmtree(_STATE_DIR / f"{name}.vols", ignore_errors=True)
    shutil.rmtree(compose_volume_root(name), ignore_errors=True)
    for pattern in (f"{name}*.passwd", f"{name}*.console.php"):
        for generated_file in _STATE_DIR.glob(pattern):
            try:
                generated_file.unlink()
            except OSError:
                pass
