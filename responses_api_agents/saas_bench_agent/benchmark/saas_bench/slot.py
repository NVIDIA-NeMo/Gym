# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/UniPat-AI/SaaS-Bench/blob/48c22deb18b98c0ed78f81e7f3be82bc162de2c8/saas_bench/slot.py
# Original license: Apache-2.0; Copyright: SaaS-Bench Authors; see bundled licenses.
"""Per-slot Docker container manager for SaaS-Bench.

Port formula: BASE_PORT + slot_id * slot_offset + app_index
Default:      30000     + slot_id * 20           + app_index

Container naming: rollout_{slot_id}_{app_name}
Compose project:  rollout_{slot_id}_{app_name}

NRT / rootless Podman (SAAS_NRT_PODMAN=1):
  - Rewrite `-p HOST:CTR` → `-p 127.0.0.1:HOST:CTR` and force slirp4netns
    (bridge CNI is broken on NRT cpu nodes due to firewall plugin mismatch).
  - Compose stacks use a Podman *pod* (shared netns) instead of CNI bridges,
    with service hostnames rewritten to 127.0.0.1.
  - Podman 3.4 leaks the slirp4netns helper on container removal: conmon exits,
    the helper is reparented to init and keeps its hostfwd listening socket, so
    the next `podman run -p` on the same port dies with
    `add_hostfwd: slirp_add_hostfwd failed`. Teardown therefore reaps orphaned
    helpers by port (see `_reap_port`), and `start_apps` rotates to a fresh port
    range if a port still cannot be freed.

NRT / Enroot on GPU nodes (SAAS_NRT_ENROOT=1):
  - No Podman/Docker/subuid on interactive GPU nodes → use enroot + socat.
  - Cluster enroot lacks ``--net``/``--pid`` → private rootless netns via unshare.
  - Compose apps share one netns; service hostnames are rewritten to 127.0.0.1.
"""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from string import Template

from saas_bench.enroot_slot import (
    EnrootPortBusy,
    _ordered_compose_services,
    compose_volume_root,
    enroot_enabled,
    start_enroot_app,
    start_enroot_compose,
    stop_enroot_app,
    wait_enroot_http_json,
)


_BASE_PORT = int(os.environ.get("SAAS_BASE_PORT", 30000))
_SLOT_OFFSET = 40  # 23 apps (index 0-22); 40 gives headroom
_SLOT_PREFIX = os.environ.get("SAAS_SLOT_PREFIX", "rollout")
_NRT = os.environ.get("SAAS_NRT_PODMAN", "").lower() in {"1", "true", "yes"}
_ENROOT = enroot_enabled()
_PUBLISH_BIND = os.environ.get("SAAS_NRT_PUBLISH_BIND", "127.0.0.1")


def _wait_podman_health(name: str, test: list[str], timeout: float = 240) -> None:
    """Run declared health checks before starting dependent pod services."""
    if not test or test[0] not in {"CMD", "CMD-SHELL"}:
        raise RuntimeError(f"{name}: service_healthy requires a healthcheck")
    command = ["sh", "-c", test[1]] if test[0] == "CMD-SHELL" else test[1:]
    deadline = time.monotonic() + timeout
    detail = "healthcheck has not passed"
    while time.monotonic() < deadline:
        state = subprocess.run(["podman", "inspect", "--format", "{{.State.Running}}", name],
                               capture_output=True, text=True, timeout=10)
        if state.returncode or state.stdout.strip() == "false":
            raise RuntimeError(f"{name}: exited before becoming healthy: {state.stderr[-400:]}")
        try:
            result = subprocess.run(["podman", "exec", name, *command], capture_output=True,
                                    text=True, timeout=min(10, max(.1, deadline-time.monotonic())))
            if result.returncode == 0:
                return
            detail = (result.stderr or result.stdout)[-500:]
        except subprocess.TimeoutExpired:
            detail = "healthcheck timed out"
        time.sleep(.5)
    raise RuntimeError(f"{name}: did not become healthy in {timeout}s: {detail}")


def _nrt_nofile_args() -> list[str]:
    """Return an explicit container descriptor limit for rootless Podman.

    The cpu_short launcher raises and records ``SAAS_NRT_NOFILE`` before
    starting Podman.  Passing it explicitly prevents compose services from
    falling back to a lower Podman/systemd default than the batch shell.
    """
    value = os.environ.get("SAAS_NRT_NOFILE")
    if not value:
        return []
    try:
        limit = int(value)
    except ValueError:
        print(f"[nrt] ignoring invalid SAAS_NRT_NOFILE={value!r}", flush=True)
        return []
    if limit < 1024:
        print(f"[nrt] ignoring too-low SAAS_NRT_NOFILE={limit}", flush=True)
        return []
    return ["--ulimit", f"nofile={limit}:{limit}"]

# Whole-slot start attempts when a publish port is still held by a leaked helper.
_START_ATTEMPTS = int(os.environ.get("SAAS_NRT_START_ATTEMPTS", 4))
# Seconds to wait for a port to become free while reaping its holder.
_PORT_RELEASE_TIMEOUT = float(os.environ.get("SAAS_NRT_PORT_RELEASE_TIMEOUT", 20))
# Fallback: shift the whole slot's port block when a port cannot be freed.
# Stride is wide enough that a shifted block never collides with another slot.
_PORT_EPOCH_STRIDE = int(os.environ.get("SAAS_NRT_PORT_EPOCH_STRIDE", 2000))
_MAX_PORT_EPOCH = int(os.environ.get("SAAS_NRT_MAX_PORT_EPOCH", 8))

# Rootless port forwarders that may outlive their container. conmon is
# deliberately excluded: it owns a *live* container's main process.
_PORT_HELPERS = ("slirp4netns", "rootlessport", "rootlesskit")

# Podman/slirp errors that mean "host port could not be claimed".
_PORT_BUSY_MARKERS = (
    "slirp_add_hostfwd",
    "address already in use",
    "port is already allocated",
    "bind: address already in use",
    "cannot listen on the TCP port",
)


class _PortBusy(RuntimeError):
    """A publish port was still held when the container tried to claim it."""


def _is_port_busy_error(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _PORT_BUSY_MARKERS)


def _port_bind_error(port: int, bind_addr: str = "127.0.0.1") -> OSError | None:
    """None if `port` can be bound right now, else the bind error.

    SO_REUSEADDR mirrors what podman/slirp4netns do when claiming a publish
    port: it tolerates TIME_WAIT leftovers from the previous task's HTTP traffic
    (which linger ~60s and are *not* a leak) while still failing against a live
    LISTEN socket (which is).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((bind_addr, port))
            return None
        except OSError as e:
            return e


def _port_free(port: int, bind_addr: str = "127.0.0.1") -> bool:
    """True if `port` can be bound right now (authoritative, no `ss` needed)."""
    return _port_bind_error(port, bind_addr) is None


def _proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def _port_listener_pids(port: int) -> list[int]:
    """PIDs with a LISTEN socket on `port`. Rootless, so `ss` reveals our own."""
    r = subprocess.run(
        ["ss", "-Hltnp", f"sport = :{port}"], capture_output=True, text=True
    )
    if r.returncode != 0:
        return []
    return sorted({int(p) for p in re.findall(r"pid=(\d+)", r.stdout)})


def _reap_port(port: int, label: str = "", timeout: float = _PORT_RELEASE_TIMEOUT) -> bool:
    """Make `port` bindable again, killing leaked rootless port forwarders.

    Only processes in `_PORT_HELPERS` are signalled, so a still-running
    container is never taken down by mistake. Returns True if the port ended up
    free.
    """
    if _port_free(port):
        return True
    started = time.time()
    deadline = started + timeout
    while True:
        # The helper's container is already gone, so there is nothing to shut
        # down gracefully — SIGKILL releases the socket immediately.
        for pid in _port_listener_pids(port):
            cmdline = _proc_cmdline(pid)
            if not any(h in cmdline for h in _PORT_HELPERS):
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                continue
            print(
                f"  {label}reaped leaked {os.path.basename(cmdline.split()[0])} "
                f"pid={pid} holding :{port}",
                flush=True,
            )
        time.sleep(0.2)
        if _port_free(port):
            return True
        if time.time() >= deadline:
            ss = subprocess.run(
                ["ss", "-Hltnp", f"sport = :{port}"], capture_output=True, text=True
            )
            print(
                f"  {label}:{port} still unusable after {time.time() - started:.1f}s "
                f"(bind: {_port_bind_error(port)}); "
                f"listeners={ss.stdout.strip() or 'none'}",
                flush=True,
            )
            return False


_TMP_DIR = os.environ.get(
    "SAAS_BENCH_TMP",
    os.path.join(tempfile.gettempdir(), "saas_bench"),
)
os.makedirs(_TMP_DIR, exist_ok=True)

# Repo root = parent of the saas_bench/ package directory.
# Used to resolve {repo_root} in apps.yaml `start` commands and to anchor
# compose template paths declared there.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_READY_OK_STATUSES = {200, 301, 302, 303, 401, 403}
_READY_INTERVAL = 2.0
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# onlyoffice runs two nginx instances; the documentserver's moves off :80.
_OO_DS_PORT = 8888


def _wait_ready(
    port: int,
    health_path: str,
    timeout: int,
    hostname: str = "localhost",
    consecutive: int = 1,
    not_contains: list[str] | None = None,
    contains: list[str] | None = None,
) -> float:
    """Poll http://{hostname}:{port}{health_path} until ready or timeout.

    Accepts response statuses in _READY_OK_STATUSES as ready. When
    ``consecutive`` > 1, requires that many successive OK probes (used by
    apps like roundcubemail whose HTTP stack flaps during warm-up).
    ``not_contains`` rejects an otherwise-OK response whose body still has
    those substrings (Twenty's error shell is HTTP 200).
    ``contains`` requires initialization markers in the HTTP response, so a
    socket that is listening but still bootstrapping is not treated as ready.
    Raises RuntimeError on timeout (caller treats as fatal task failure).
    Returns elapsed seconds when ready.
    """
    url = f"http://{hostname}:{port}{health_path}"
    start = time.time()
    deadline = start + timeout
    last_err = "no probe attempted"
    need = max(1, int(consecutive or 1))
    streak = 0
    needles = [n for n in (not_contains or []) if n]
    required = [n for n in (contains or []) if n]
    while time.time() < deadline:
        status = None
        body = ""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rollout-probe/1.0"})
            with urllib.request.urlopen(req, timeout=3) as r:
                status = r.status
                body = r.read(65536).decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            status = e.code
            try:
                body = e.read(65536).decode("utf-8", errors="replace")
            except Exception:
                body = ""
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:100]}"
            streak = 0
        if status is not None:
            if status in _READY_OK_STATUSES:
                hit = next((n for n in needles if n in body), None)
                missing = next((n for n in required if n not in body), None)
                if hit or missing:
                    streak = 0
                    last_err = (
                        f"HTTP {status} body still contains {hit!r}"
                        if hit
                        else f"HTTP {status} body missing {missing!r}"
                    )
                else:
                    streak += 1
                    if streak >= need:
                        return round(time.time() - start, 1)
            else:
                streak = 0
                last_err = f"HTTP {status}"
        time.sleep(_READY_INTERVAL)
    elapsed = round(time.time() - start, 1)
    raise RuntimeError(
        f"app on port {port} not ready in {timeout}s "
        f"(probed {url}, last={last_err}, consecutive_need={need}, elapsed={elapsed}s)"
    )


def _nrt_rewrite_docker_run(cmd: str) -> str:
    """Adapt a docker-run start line for rootless Podman on NRT."""
    if not _NRT:
        return cmd
    # Bind publishes to loopback (slirp hostfwd is reliable for 127.0.0.1).
    def _pub(m: re.Match[str]) -> str:
        host_port, ctr_port = m.group(1), m.group(2)
        return f"-p {_PUBLISH_BIND}:{host_port}:{ctr_port}"

    cmd = re.sub(r"-p\s+(\d+):(\d+)", _pub, cmd)
    # Avoid broken CNI bridges; rootless default without --network is slirp,
    # but be explicit when no network flag is present.
    if "--network" not in cmd and " --net " not in cmd:
        cmd = cmd.replace(
            "docker run ",
            "docker run --network=slirp4netns:port_handler=slirp4netns ",
            1,
        )
    # NRT sets pidns/ipcns=host cluster-wide; s6-overlay images require /init
    # to be PID 1. containers.conf also overrides this, but be explicit here.
    if "--pid" not in cmd:
        cmd = cmd.replace("docker run ", "docker run --pid=private ", 1)
    if "--ipc" not in cmd:
        cmd = cmd.replace("docker run ", "docker run --ipc=private ", 1)
    # docker-mailserver (roundcubemail): under slirp the iface is tap0, not eth0.
    # Without this DMS fails IP detection and shuts down after a brief HTTP 200.
    if "mw-roundcubemail" in cmd and "NETWORK_INTERFACE=" not in cmd:
        cmd = cmd.replace(
            "docker run ",
            "docker run -e NETWORK_INTERFACE=tap0 -e PERMIT_DOCKER=none ",
            1,
        )
    return cmd


class SlotManager:
    def __init__(self, apps_config: dict, slot_id: int):
        self.apps = apps_config
        self.slot_id = slot_id
        # Bumped only when a publish port cannot be freed; shifts the whole
        # slot's block so callers of get_port_map() stay consistent.
        self._port_epoch = 0

    # -- Public interface -----------------------------------------------------

    def get_port(self, app: str) -> int:
        return (
            _BASE_PORT
            + self._port_epoch * _PORT_EPOCH_STRIDE
            + self.slot_id * _SLOT_OFFSET
            + self.apps[app]["app_index"]
        )

    def get_container_name(self, app: str) -> str:
        return f"{_SLOT_PREFIX}_{self.slot_id}_{app}"

    def get_port_map(self, apps: list[str]) -> dict[str, int]:
        return {app: self.get_port(app) for app in apps}

    def get_pg_port(self, app: str) -> int | None:
        """Return the exposed Postgres port for an app, or None if not configured."""
        offset = self.apps.get(app, {}).get("pg_port_offset")
        if offset is not None:
            return self.get_port(app) + int(offset)
        return None

    def start_apps(self, apps: list[str], hostname: str = "localhost") -> None:
        """Start `apps` on this slot, retrying the whole set on port conflicts.

        Retries are slot-wide (not per-app) because the fallback shifts every
        port in the slot; a per-app retry would desync get_port_map() from the
        containers already running.
        """
        last_busy: Exception | None = None
        for attempt in range(1, _START_ATTEMPTS + 1):
            for app in apps:
                self._stop_one(app)
            self._release_ports(apps)

            errors: list[tuple[str, Exception]] = []
            with ThreadPoolExecutor(max_workers=len(apps)) as pool:
                futures = {
                    pool.submit(self._start_one, app, hostname): app for app in apps
                }
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001 — aggregated below
                        errors.append((futures[fut], exc))
            if not errors:
                return

            # Podman and Enroot report an unavailable host publish port via
            # separate backend-specific exceptions.  Both are safe to retry:
            # stop the partially-started slot and move its *entire* port block
            # so the agent's port map remains internally consistent.
            busy = [
                (a, e)
                for a, e in errors
                if isinstance(e, (_PortBusy, EnrootPortBusy))
            ]
            if not busy:
                raise errors[0][1]
            last_busy = busy[0][1]
            if attempt == _START_ATTEMPTS:
                break
            # Ports were still claimed after reaping; move the slot's whole port
            # block so a helper we cannot kill stops mattering.
            if self._port_epoch < _MAX_PORT_EPOCH:
                self._port_epoch += 1
                print(
                    f"  [slot {self.slot_id}] port conflict on "
                    f"{', '.join(a for a, _ in busy)}; shifting slot ports to "
                    f"epoch {self._port_epoch} (attempt {attempt}/{_START_ATTEMPTS})",
                    flush=True,
                )
            else:
                print(
                    f"  [slot {self.slot_id}] port conflict on "
                    f"{', '.join(a for a, _ in busy)}; retrying "
                    f"(attempt {attempt}/{_START_ATTEMPTS})",
                    flush=True,
                )
        raise RuntimeError(
            f"[slot {self.slot_id}] could not claim publish ports after "
            f"{_START_ATTEMPTS} attempts: {last_busy}"
        )

    def stop_apps(self, apps: list[str]) -> None:
        for app in apps:
            self._stop_one(app)
        # Podman 3.4 leaks slirp4netns on removal; free the ports now so the
        # next task on this slot can reuse them.
        self._release_ports(apps)

    # -- Internal helpers -----------------------------------------------------

    def _release_ports(self, apps: list[str]) -> None:
        """Reap leaked forwarders still holding any of `apps`' publish ports."""
        if not _NRT:
            return
        for app in apps:
            for port in self._host_ports_for(app):
                if not _reap_port(port, label=f"[slot {self.slot_id}] "):
                    print(
                        f"  [slot {self.slot_id}] WARNING: :{port} still bound "
                        f"after reaping (will shift ports if start fails)",
                        flush=True,
                    )

    def _host_ports_for(self, app: str) -> list[int]:
        """Every host port this app publishes on the current port epoch."""
        cfg = self._get(app)
        port = self.get_port(app)
        ports = {port}
        pg_port = self.get_pg_port(app)
        if pg_port is not None:
            ports.add(pg_port)
        if cfg.get("start_type") == "compose":
            ports.update(self._compose_host_ports(app, cfg, port))
        else:
            fmt = dict(
                container=self.get_container_name(app),
                port=port,
                hostname="localhost",
                repo_root=_REPO_ROOT,
            )
            if pg_port is not None:
                fmt["pg_port"] = pg_port
            try:
                cmd = cfg.get("start", "").format(**fmt)
            except (KeyError, IndexError):
                cmd = ""
            for m in re.finditer(r"-p\s+(?:[\d.]+:)?(\d+):\d+", cmd):
                ports.add(int(m.group(1)))
        return sorted(ports)

    def _compose_host_ports(self, app: str, cfg: dict, port: int) -> list[int]:
        tpl_path = cfg.get("compose_template_file", "")
        if not tpl_path:
            return []
        if not os.path.isabs(tpl_path):
            tpl_path = os.path.join(_REPO_ROOT, tpl_path)
        try:
            with open(tpl_path) as f:
                raw = Template(f.read()).safe_substitute(
                    prefix=self.get_container_name(app),
                    port=port,
                    hostname="localhost",
                )
        except OSError:
            return []
        return [
            int(m.group(1))
            for m in re.finditer(
                r"^\s*-\s*[\"']?(?:[\d.]+:)?(\d+):\d+[\"']?\s*$", raw, re.M
            )
        ]

    def _start_one(self, app: str, hostname: str) -> None:
        cfg = self._get(app)
        port = self.get_port(app)
        name = self.get_container_name(app)

        print(f"  [slot {self.slot_id}] starting {app} on :{port}...", flush=True)

        if cfg.get("start_type") == "compose":
            if _ENROOT:
                self._start_compose_enroot(app, cfg, port, name, hostname)
            elif _NRT:
                self._start_compose_nrt_pod(app, cfg, port, name, hostname)
            else:
                self._start_compose(app, cfg, port, name, hostname)
        elif _ENROOT:
            ctr_port = int(cfg.get("container_port") or 0)
            if not ctr_port:
                raise RuntimeError(
                    f"[slot {self.slot_id}] {app} missing container_port "
                    f"(required for SAAS_NRT_ENROOT=1)"
                )
            fmt = dict(container=name, port=port, hostname=hostname, repo_root=_REPO_ROOT)
            pg_offset = cfg.get("pg_port_offset")
            if pg_offset is not None:
                fmt["pg_port"] = port + int(pg_offset)
            start_enroot_app(
                name=name,
                start_cmd=cfg["start"],
                container_port=ctr_port,
                host_port=port,
                fmt=fmt,
                label=f"[slot {self.slot_id}] ",
                listen_timeout=float(cfg.get("startup_wait", 600)),
            )
        else:
            # Build extra format vars (e.g. pg_port for Baserow Postgres exposure)
            fmt = dict(container=name, port=port, hostname=hostname, repo_root=_REPO_ROOT)
            pg_offset = cfg.get("pg_port_offset")
            if pg_offset is not None:
                fmt["pg_port"] = port + int(pg_offset)
            cmd = _nrt_rewrite_docker_run(cfg["start"].format(**fmt))
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if r.returncode != 0:
                subprocess.run(f"docker rm -f {name} 2>/dev/null", shell=True,
                               capture_output=True, text=True)
                if _NRT and _is_port_busy_error(r.stderr):
                    raise _PortBusy(
                        f"[slot {self.slot_id}] {app} could not claim :{port}: "
                        f"{r.stderr.strip()}"
                    )
                raise RuntimeError(f"[slot {self.slot_id}] failed to start {app}: {r.stderr.strip()}")

        # Readiness probe (replaces fixed sleep)
        health_path = cfg.get("health_path", "/")
        timeout = cfg.get("startup_wait", 600)
        consecutive = int(cfg.get("ready_consecutive", 1) or 1)
        not_contains = cfg.get("ready_not_contains") or []
        contains = cfg.get("ready_contains") or []
        if isinstance(not_contains, str):
            not_contains = [not_contains]
        if isinstance(contains, str):
            contains = [contains]
        try:
            elapsed = _wait_ready(
                port,
                health_path,
                timeout,
                hostname,
                consecutive=consecutive,
                not_contains=list(not_contains),
                contains=list(contains),
            )
            print(f"  [slot {self.slot_id}] {app} ready in {elapsed}s", flush=True)
        except RuntimeError as e:
            if _ENROOT:
                raise RuntimeError(
                    f"[slot {self.slot_id}] {app} readiness FAILED: {e}"
                ) from e
            ps = subprocess.run(
                f"docker ps -a --filter 'name={name}' --format '{{{{.Names}}}} {{{{.Status}}}}'",
                shell=True, capture_output=True, text=True,
            )
            logs = subprocess.run(
                f"docker logs --tail 20 {name} 2>&1",
                shell=True, capture_output=True, text=True,
            )
            raise RuntimeError(
                f"[slot {self.slot_id}] {app} readiness FAILED: {e}\n"
                f"  docker_ps: {ps.stdout.strip()}\n"
                f"  logs_tail: {logs.stdout[-600:]}"
            ) from e

    def _start_compose_enroot(
        self, app: str, cfg: dict, port: int, prefix: str, hostname: str
    ) -> None:
        """Start a compose app as Enroot containers sharing one unshare netns."""
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Enroot compose requires PyYAML (`pip install pyyaml`)"
            ) from e

        ctr_port = int(cfg.get("container_port") or 0)
        if not ctr_port:
            raise RuntimeError(
                f"[slot {self.slot_id}] {app} missing container_port "
                f"(required for SAAS_NRT_ENROOT=1 compose)"
            )
        # Compose volume directories are created below, so remove the previous
        # stack before constructing the new mounts.
        stop_enroot_app(prefix)

        tpl_path = cfg["compose_template_file"]
        if not os.path.isabs(tpl_path):
            tpl_path = os.path.join(_REPO_ROOT, tpl_path)
        with open(tpl_path) as f:
            raw = Template(f.read()).safe_substitute(
                prefix=prefix, port=port, hostname=hostname
            )
        raw = re.sub(
            rf"@{re.escape(prefix)}[A-Za-z0-9_-]*:",
            "@127.0.0.1:",
            raw,
        )
        data = yaml.safe_load(raw)
        services: dict = data.get("services") or {}
        if not services:
            raise RuntimeError(f"[slot {self.slot_id}] empty compose for {app}")

        host_aliases = []
        for sname, sval in services.items():
            host_aliases.append((sval or {}).get("container_name") or f"{prefix}-{sname}")
        host_aliases = sorted(set(host_aliases), key=len, reverse=True)

        def _host_rw(text: str) -> str:
            for alias in host_aliases:
                if alias and alias != prefix:
                    text = text.replace(alias, "127.0.0.1")
            return text

        def _rank(item: tuple[str, dict]) -> int:
            n = item[0].lower()
            if any(k in n for k in ("postgres", "mysql", "mariadb", "mongo", "redis", "db")):
                return 0
            return 1

        vol_root = str(compose_volume_root(prefix))
        os.makedirs(vol_root, exist_ok=True)

        parsed: list[dict] = []
        for _svc_name, svc in sorted(services.items(), key=_rank):
            cname = svc.get("container_name") or f"{prefix}-{_svc_name}"
            image = svc.get("image")
            if not image:
                raise RuntimeError(f"[slot {self.slot_id}] {_svc_name} missing image")
            image_repo = str(image).rsplit("/", 1)[-1].split(":", 1)[0]
            env = svc.get("environment")
            env_pairs: list[tuple[str, str]] = []
            if isinstance(env, dict):
                env_pairs = [(str(k), _host_rw(str(v))) for k, v in env.items()]
            elif isinstance(env, list):
                for item in env:
                    k, _, v = _host_rw(str(item)).partition("=")
                    env_pairs.append((k, v))
            if image_repo == "oo-bundle-community2":
                # Follow the documentserver to the port it was moved to.
                env_pairs = [
                    (k, f"127.0.0.1:{_OO_DS_PORT}" if k == "DOCUMENT_SERVER_HOST" else v)
                    for k, v in env_pairs
                ]
                env_pairs.append(("SERVER_HOST", f"{hostname}:{port}"))
            # enroot exports every -e through a shell, so a name that is not a
            # valid identifier (elasticsearch's `discovery.type`) makes it reject
            # the service outright. Collect those and apply them as config.
            envs: list[str] = []
            file_settings: list[tuple[str, str]] = []
            for k, v in env_pairs:
                if _ENV_NAME_RE.match(k):
                    envs.extend(["-e", f"{k}={v}"])
                else:
                    file_settings.append((k, v))
            if any(token in image_repo for token in ("postgres", "mysql", "mariadb")):
                envs.extend(["-e", "SAAS_FAKE_ID=1"])
            if "postgres" in image_repo:
                envs.extend(["-e", "SAAS_FORCE_UID=999"])
            if image_repo == "oo-bundle-ds":
                # The bundled postgres init script refuses to start while its
                # data dir looks root-owned. Report the image's postgres uid for
                # that subtree only, so `id -u` root checks elsewhere still pass.
                # Its nginx also has to move off :80, which the community
                # server owns in the shared netns.
                envs.extend(
                    [
                        "-e",
                        "SAAS_FAKE_OWNER_PATH=/var/lib/postgresql",
                        "-e",
                        "SAAS_FAKE_OWNER_UID=107",
                        "-e",
                        f"SAAS_NGINX_LISTEN_PORT={_OO_DS_PORT}",
                    ]
                )
            mounts: list[str] = []
            volume_copies: list[dict] = []
            for vol in svc.get("volumes") or []:
                spec = str(vol)
                parts = spec.split(":")
                if len(parts) < 2:
                    continue
                src, dst = parts[0], parts[1]
                # The Enroot community init starts services directly instead
                # of systemd. Binding Docker's privileged cgroup hierarchy
                # conflicts with Enroot's own sysfs mounts and fails before
                # the application can execute.
                if any(path == '/sys/fs/cgroup' or path.startswith('/sys/fs/cgroup/')
                       for path in (src.rstrip('/'), dst.rstrip('/'))):
                    continue
                if src.startswith("/"):
                    if not os.path.exists(src):
                        continue
                    host = src
                else:
                    host = os.path.join(vol_root, src.replace("/", "_"))
                    os.makedirs(host, exist_ok=True)
                    volume_copies.append({"source": dst, "target": host})
                opts = "x-create=dir,bind"
                if len(parts) >= 3 and parts[2] in {"ro", "readonly"}:
                    opts += ",ro"
                mounts.extend(["-m", f"{host}:{dst}:none:{opts}"])
            if image_repo == "oo-bundle-es":
                lines = ['cluster.name: "docker-cluster"', "network.host: 0.0.0.0"]
                settings = {
                    # Small, single-node benchmark indices need absolute free
                    # space protection; percentages misclassify large shared
                    # filesystems with terabytes free as being out of space.
                    "cluster.routing.allocation.disk.watermark.low": "5gb",
                    "cluster.routing.allocation.disk.watermark.high": "2gb",
                    "cluster.routing.allocation.disk.watermark.flood_stage": "1gb",
                    **dict(file_settings),
                }
                for k, v in settings.items():
                    # Locking memory needs a memlock ulimit we cannot raise
                    # rootless; elasticsearch refuses to start if it is asked to.
                    lines.append(f"{k}: {'false' if k == 'bootstrap.memory_lock' else v}")
                es_yml = os.path.join(vol_root, "elasticsearch.yml")
                with open(es_yml, "w") as fh:
                    fh.write("\n".join(lines) + "\n")
                mounts.extend(
                    [
                        "-m",
                        f"{es_yml}:/usr/share/elasticsearch/config/elasticsearch.yml"
                        ":none:x-create=file,bind,ro",
                    ]
                )
            if image_repo == "mw-pretix":
                # Use the compose DB and Redis. The all-in-one image otherwise
                # starts duplicate local copies under Supervisor, which collide
                # in our shared network namespace. Also wait for both external
                # services before `pretix migrate`; Django imports consult Redis
                # even during migration.
                # The image defaults to 2*nproc Gunicorn workers plus a Celery
                # pool sized from the host. One evaluation agent does not need
                # hundreds of workers or their persistent database connections.
                if not any(k == "NUM_WORKERS" for k, _ in env_pairs):
                    envs.extend(["-e", "NUM_WORKERS=2"])
                task_conf = os.path.join(vol_root, "pretix-taskworker.conf")
                with open(task_conf, "w") as fh:
                    fh.write(
                        "[program:pretixtask]\n"
                        "command=/usr/local/bin/pretix taskworker --concurrency=2\n"
                        "autostart=true\nautorestart=true\npriority=5\nuser=pretixuser\n"
                        "stdout_logfile=/dev/fd/1\nstdout_logfile_maxbytes=0\n"
                        "stderr_logfile=/dev/fd/2\nstderr_logfile_maxbytes=0\n"
                    )
                mounts.extend(["-m", f"{task_conf}:/etc/supervisord/pretixtask.conf:none:x-create=file,bind,ro"])
                supervisor_conf = os.path.join(vol_root, "pretix-supervisord.conf")
                with open(supervisor_conf, "w") as fh:
                    fh.write(
                        "[include]\n"
                        "files = /etc/supervisord/base.conf "
                        "/etc/supervisord/nginx.conf "
                        "/etc/supervisord/pretixtask.conf "
                        "/etc/supervisord/pretixweb.conf\n"
                    )
                pretix_init = os.path.join(vol_root, "pretix-entrypoint.sh")
                with open(pretix_init, "w") as fh:
                    fh.write(
                        "#!/bin/bash\n"
                        "set -e\n"
                        "for port in 5432 6379; do\n"
                        "  for i in $(seq 1 120); do\n"
                        "    (echo >/dev/tcp/127.0.0.1/$port) "
                        ">/dev/null 2>&1 && break\n"
                        "    sleep 0.25\n"
                        "  done\n"
                        "done\n"
                        'PRETIX_URL="${PRETIX_URL:-http://localhost:6089}"\n'
                        'sed -i "s|^url=.*|url=${PRETIX_URL}|" '
                        "/etc/pretix/pretix.cfg\n"
                        "# SaaS-Bench paid-event tasks need a credential-free "
                        "payment provider before Pretix allows Go live.\n"
                        "sed -i \"/plugins = models.TextField(/,"
                        "/verbose_name=_(\\\"Plugins\\\")/ "
                        "s/null=False, blank=True,/"
                        "null=False, blank=True, "
                        "default='pretix.plugins.manualpayment',/\" "
                        "/pretix/src/pretix/base/models/event.py\n"
                        "sed -i \"s/return self.settings.get('_enabled', "
                        "as_type=bool)/return self.settings.get('_enabled', "
                        "default=self.identifier == 'manual', "
                        "as_type=bool)/\" "
                        "/pretix/src/pretix/base/payment.py\n"
                        'exec /usr/local/bin/pretix "$@"\n'
                    )
                os.chmod(pretix_init, 0o755)
                mounts.extend(
                    [
                        "-m",
                        f"{supervisor_conf}:/etc/supervisord.all.conf"
                        ":none:x-create=file,bind,ro",
                        "-m",
                        f"{pretix_init}:/entrypoint-mw.sh"
                        ":none:x-create=file,bind,ro",
                    ]
                )
            extra: list[str] = []
            ep = svc.get("entrypoint")
            cmd = svc.get("command")
            if ep is not None:
                if isinstance(ep, list):
                    extra.extend(_host_rw(str(x)) for x in ep)
                else:
                    extra.extend(["sh", "-c", _host_rw(str(ep))])
            elif cmd is not None:
                if isinstance(cmd, list):
                    extra.extend(_host_rw(str(x)) for x in cmd)
                else:
                    extra.extend(["sh", "-c", _host_rw(cmd)])
            if "postgres" in image_repo:
                # Enter the image entrypoint with a persisted fake uid. The
                # stack uses loopback, so Mattermost's pg_hba network patch is
                # unnecessary here.
                extra = [
                    "/saas-bin/gosu",
                    "postgres",
                    "docker-entrypoint.sh",
                    "postgres",
                ]
            elif image_repo == "mw-mattermost":
                extra = ["/mattermost/bin/mattermost", "server"]
            elif image_repo == "oo-bundle-community2":
                # The image's setup script ends by exec'ing systemd, which
                # cannot run here; this init starts the units we need instead.
                init_sh = os.path.join(_REPO_ROOT, "saas_bench", "oo_community_init.sh")
                mounts.extend(
                    ["-m", f"{init_sh}:/saas-oo-init.sh:none:x-create=file,bind,ro"]
                )
                extra = ["/bin/bash", "/saas-oo-init.sh"]
            elif image_repo == "mw-pretix":
                # enroot's /etc/rc always prepends the image ENTRYPOINT, so a
                # compose `entrypoint:` override cannot be expressed; our args
                # arrive as CMD. Pass the image's own CMD. The compose override
                # only rewrites pretix.cfg hosts to the per-service names, which
                # the baked config already has right (localhost) for one netns.
                extra = ["all"]
            healthcheck = [
                _host_rw(str(value)).replace("$$", "$")
                for value in (svc.get("healthcheck") or {}).get("test", [])
            ]
            if image_repo == "oo-bundle-ds":
                healthcheck = [value.replace("http://localhost/healthcheck",
                                             f"http://127.0.0.1:{_OO_DS_PORT}/healthcheck")
                              for value in healthcheck]
            if "postgres" in image_repo:
                pg_env = dict(env_pairs)
                pg_user = pg_env.get("POSTGRES_USER", "postgres")
                # pg_isready on the Unix socket can succeed for the temporary
                # seed/import server. The consumer needs the final TCP server,
                # the requested database, and its actual login credentials.
                healthcheck = ["CMD", "env", "PGPASSWORD=" + pg_env.get("POSTGRES_PASSWORD", ""),
                               "psql", "-h", "127.0.0.1", "-U", pg_user,
                               "-d", pg_env.get("POSTGRES_DB", pg_user), "-tAc", "SELECT 1"]
            if image_repo == "oo-bundle-community2":
                healthcheck = ["CMD-SHELL", "test -f /run/saas-onlyoffice-ready && curl -fsS http://127.0.0.1/auth.aspx >/dev/null"]
            parsed.append(
                {
                    "name": str(cname),
                    "image": str(image),
                    "envs": envs,
                    "mounts": mounts,
                    "extra": extra,
                    # The Mattermost application image is distroless and has no
                    # /bin/sh, so it cannot run rc.local or shell wrappers.
                    "uid_runtime": image_repo != "mw-mattermost",
                    "patch_owncloud_console": image_repo == "mw-owncloud-server",
                    "depends_on": [
                        {"name": (services[dep] or {}).get("container_name") or f"{prefix}-{dep}",
                         "condition": (spec or {}).get("condition", "service_started")
                                      if isinstance(spec, dict) else "service_started"}
                        for dep, spec in (
                            (svc.get("depends_on") or {}).items()
                            if isinstance(svc.get("depends_on"), dict)
                            else ((dep, {}) for dep in svc.get("depends_on") or [])
                        )
                    ],
                    "healthcheck": healthcheck,
                    "ready_gate": image_repo == "oo-bundle-community2",
                    "volume_copies": volume_copies,
                }
            )

        start_enroot_compose(
            name=prefix,
            services=parsed,
            container_port=ctr_port,
            host_port=port,
            label=f"[slot {self.slot_id}] ",
            listen_timeout=float(cfg.get("startup_wait", 600)),
        )
        if app == 'onlyoffice':
            # Docker waits for these dependencies via service_healthy. The
            # rootless translation must also wait before handing off to agents.
            timeout = float(cfg.get('startup_wait', 180))
            wait_enroot_http_json(prefix, 9200, '/_cluster/health',
                                  json_key='status', accepted_values=('green', 'yellow'), timeout=timeout)
            wait_enroot_http_json(prefix, _OO_DS_PORT, '/healthcheck',
                                  accepted_values=(True,), timeout=timeout)

    def _start_compose(self, app: str, cfg: dict, port: int, prefix: str, hostname: str) -> None:
        tpl_path = cfg["compose_template_file"]
        # resolve relative path from repo root
        if not os.path.isabs(tpl_path):
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            tpl_path = os.path.join(here, tpl_path)

        with open(tpl_path) as f:
            content = Template(f.read()).safe_substitute(prefix=prefix, port=port, hostname=hostname)

        tmp = f"{_TMP_DIR}/{prefix}.yml"
        with open(tmp, "w") as f:
            f.write(content)

        cmd = f"docker compose --project-name {prefix} -f {tmp} up -d"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"[slot {self.slot_id}] compose failed for {app}: {r.stderr.strip()}")

    def _start_compose_nrt_pod(
        self, app: str, cfg: dict, port: int, prefix: str, hostname: str
    ) -> None:
        """Start a compose app as a Podman pod (shared netns) to avoid broken CNI.

        Service-to-service DNS (`$prefix-postgres`) is rewritten to 127.0.0.1.
        Only the primary published port from the template is exposed on the pod.
        """
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "NRT compose-via-pod requires PyYAML (`pip install pyyaml`)"
            ) from e

        tpl_path = cfg["compose_template_file"]
        if not os.path.isabs(tpl_path):
            tpl_path = os.path.join(_REPO_ROOT, tpl_path)
        with open(tpl_path) as f:
            raw = Template(f.read()).safe_substitute(prefix=prefix, port=port, hostname=hostname)
        # Rewrite inter-service hostnames to loopback (shared pod netns).
        # e.g. postgres://user:pass@rollout_0_mattermost-postgres:5432 → ...@127.0.0.1:5432
        raw = re.sub(
            rf"@{re.escape(prefix)}[A-Za-z0-9_-]*:",
            "@127.0.0.1:",
            raw,
        )
        data = yaml.safe_load(raw)
        services: dict = data.get("services") or {}
        if not services:
            raise RuntimeError(f"[slot {self.slot_id}] empty compose for {app}")

        # In a pod all services share one netns → they reach each other on
        # 127.0.0.1, NOT via container hostname. Compose templates reference peers
        # by container name in env values (OWNCLOUD_DB_HOST=$prefix-mariadb) and
        # inside entrypoint/command scripts (pretix sed 'host=$prefix-db'). Rewrite
        # every prefix-based container hostname to loopback. Longest-first so
        # `$prefix-db` is replaced before the bare `$prefix`.
        host_aliases = []
        for sname, sval in services.items():
            host_aliases.append((sval or {}).get("container_name") or f"{prefix}-{sname}")
        host_aliases = sorted(set(host_aliases), key=len, reverse=True)

        def _host_rw(text: str) -> str:
            for alias in host_aliases:
                if alias and alias != prefix:  # don't rewrite the app's own name
                    text = text.replace(alias, "127.0.0.1")
            return text

        # Collect published ports for the pod (host side).
        publish_args: list[str] = []
        for svc in services.values():
            for p in svc.get("ports") or []:
                # formats: "30011:8065" or "127.0.0.1:30011:8065"
                parts = str(p).split(":")
                if len(parts) == 2:
                    host_p, ctr_p = parts
                elif len(parts) == 3:
                    host_p, ctr_p = parts[1], parts[2]
                else:
                    continue
                publish_args.extend(["-p", f"{_PUBLISH_BIND}:{host_p}:{ctr_p}"])

        pod_name = f"{prefix}-pod"
        pod_host_ports = [
            int(spec.split(":")[-2]) for spec in publish_args if spec != "-p"
        ]

        def _fail(msg: str, stderr: str) -> RuntimeError:
            """Classify a pod failure as port conflict vs real error, and clean up."""
            subprocess.run(["podman", "pod", "rm", "-f", pod_name], capture_output=True)
            # Podman surfaces an unavailable pod publish port indirectly
            # ("error starting some container dependencies"), so a still-bound
            # published port also counts as a port conflict.
            if _NRT and (
                _is_port_busy_error(stderr)
                or any(not _port_free(p) for p in pod_host_ports)
            ):
                return _PortBusy(f"[slot {self.slot_id}] {msg}: {stderr}")
            return RuntimeError(f"[slot {self.slot_id}] {msg}: {stderr}")

        subprocess.run(["podman", "pod", "rm", "-f", pod_name], capture_output=True)
        # NRT forces netns=host cluster-wide; publishing ports on a pod requires
        # slirp4netns (or bridge). Force slirp so the pod's infra container owns
        # a private netns and hostfwd works on 127.0.0.1.
        create = [
            "podman", "pod", "create", "--name", pod_name,
            "--network", "slirp4netns:port_handler=slirp4netns",
            *publish_args,
        ]
        r = subprocess.run(create, capture_output=True, text=True)
        if r.returncode != 0:
            raise _fail(f"pod create failed for {app}", r.stderr.strip())

        ordered = _ordered_compose_services([
            {"name": name, "service": service, "depends_on": [
                {"name": dependency, "condition": (value or {}).get("condition", "service_started")}
                for dependency, value in ((service.get("depends_on") or {}).items()
                    if isinstance(service.get("depends_on"), dict)
                    else ((dep, {}) for dep in service.get("depends_on") or []))
            ]} for name, service in services.items()
        ])
        healthy = set()
        for row in ordered:
            svc_name, svc = row["name"], row["service"]
            for dependency in row["depends_on"]:
                dep_name = dependency["name"]
                if dependency["condition"] == "service_healthy" and dep_name not in healthy:
                    dep = services[dep_name]
                    dep_image = str(dep.get("image", "")).rsplit("/", 1)[-1].split(":", 1)[0]
                    test = [_host_rw(str(value)) for value in (dep.get("healthcheck") or {}).get("test", [])]
                    if dep_image == "oo-bundle-ds":
                        test = ["CMD", "curl", "-fsS", f"http://127.0.0.1:{_OO_DS_PORT}/healthcheck"]
                    if "postgres" in dep_image:
                        test = ["CMD-SHELL", 'PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atqc "SELECT 1"']
                    _wait_podman_health(dep.get("container_name") or f"{prefix}-{dep_name}", test)
                    healthy.add(dep_name)
            cname = svc.get("container_name") or f"{prefix}-{svc_name}"
            image = svc.get("image")
            if not image:
                raise RuntimeError(f"[slot {self.slot_id}] {svc_name} missing image")
            args = [
                "podman", "run", "-d", "--pod", pod_name, "--name", str(cname),
                *_nrt_nofile_args(),
            ]
            if svc.get("user") is not None:
                args.extend(["--user", str(svc["user"])])
            if svc.get("privileged"):
                args.append("--privileged")
            image_repo = str(image).rsplit("/", 1)[-1].split(":", 1)[0]
            env = svc.get("environment")
            env_pairs = []
            if isinstance(env, dict):
                env_pairs = [(str(k), _host_rw(str(v))) for k, v in env.items()]
            elif isinstance(env, list):
                env_pairs = [
                    (key, value if separator else None)
                    for item in env
                    for key, separator, value in [_host_rw(str(item)).partition("=")]
                ]
            if image_repo == "oo-bundle-es":
                env_pairs = [(k, v) for k, v in env_pairs if k != "bootstrap.memory_lock"]
                env_pairs.append(("bootstrap.memory_lock", "false"))
                existing = dict(env_pairs)
                for level, space in [("low", "5gb"), ("high", "2gb"), ("flood_stage", "1gb")]:
                    key = "cluster.routing.allocation.disk.watermark." + level
                    if key not in existing:
                        env_pairs.append((key, space))
            if image_repo == "oo-bundle-community2":
                env_pairs = [
                    (k, f"127.0.0.1:{_OO_DS_PORT}" if k == "DOCUMENT_SERVER_HOST" else v)
                    for k, v in env_pairs
                ]
                env_pairs.append(("SAAS_OO_PUBLIC_PORT", str(port)))
                env_pairs.append(("SERVER_HOST", f"{hostname}:{port}"))
            for k, v in env_pairs:
                args.extend(["-e", k if v is None else f"{k}={v}"])
            if image_repo == "oo-bundle-ds":
                args.extend(["-e", f"SAAS_OO_DS_PORT={_OO_DS_PORT}"])
            for vol in svc.get("volumes") or []:
                # named volumes → podman -v name:path
                args.extend(["-v", str(vol)])
            for opt in svc.get("security_opt") or []:
                o = str(opt)
                # docker uses `no-new-privileges:true`; podman wants the bare flag.
                if o.startswith("no-new-privileges"):
                    args.extend(["--security-opt", "no-new-privileges"])
                elif ":" in o and "=" not in o:
                    # docker `key:value` → podman `key=value` (e.g. label:disable)
                    args.extend(["--security-opt", o.replace(":", "=", 1)])
                else:
                    args.extend(["--security-opt", o])
            if svc.get("pids_limit") is not None:
                args.extend(["--pids-limit", str(svc["pids_limit"])])
            for tfs in svc.get("tmpfs") or []:
                args.extend(["--tmpfs", str(tfs)])
            # entrypoint override (list → JSON array form; str → passthrough).
            ep = svc.get("entrypoint")
            if image_repo in {"oo-bundle-ds", "oo-bundle-community2"}:
                # The default template assumes separate container networks and
                # systemd. Neither assumption holds in this rootless shared pod.
                script = (
                    "oo_documentserver_init.sh"
                    if image_repo == "oo-bundle-ds"
                    else "oo_community_init.sh"
                )
                target = "/saas-onlyoffice-init.sh"
                source = os.path.join(_REPO_ROOT, "saas_bench", script)
                args.extend(["-v", f"{source}:{target}:ro"])
                ep = ["/bin/bash", target]
            if ep is not None:
                import json as _json
                if isinstance(ep, list):
                    args.extend(["--entrypoint", _json.dumps([_host_rw(str(x)) for x in ep])])
                else:
                    args.extend(["--entrypoint", _host_rw(str(ep))])
            # IMAGE must come after all flags and before the command args.
            args.append(str(image))
            cmd = svc.get("command")
            if isinstance(cmd, list):
                args.extend([_host_rw(str(x)) for x in cmd])
            elif isinstance(cmd, str):
                # shell-form command from YAML
                args.extend(["sh", "-c", _host_rw(cmd)])
            r = subprocess.run(args, capture_output=True, text=True)
            if r.returncode != 0:
                raise _fail(f"pod run {cname} failed", r.stderr.strip())
            if image_repo == "oo-bundle-community2":
                _wait_podman_health(str(cname), ["CMD-SHELL", "test -f /run/saas-onlyoffice-ready && curl -fsS http://127.0.0.1/auth.aspx >/dev/null"])

        print(
            f"  [slot {self.slot_id}] NRT pod {pod_name} started "
            f"({len(services)} services)",
            flush=True,
        )

    def _stop_one(self, app: str) -> None:
        cfg = self._get(app)
        prefix = self.get_container_name(app)

        if _ENROOT:
            stop_enroot_app(prefix)
            return

        if cfg.get("start_type") == "compose":
            if _NRT:
                subprocess.run(
                    ["podman", "pod", "rm", "-f", f"{prefix}-pod"],
                    capture_output=True,
                )
            containers = self._compose_containers(app, cfg, prefix)
            for c in containers:
                subprocess.run(
                    f"docker stop {c} 2>/dev/null; docker rm -f {c} 2>/dev/null",
                    shell=True, capture_output=True, text=True,
                )
            # Remove named volumes
            for suffix in cfg.get("compose_volumes", []):
                vol = f"{prefix}{suffix}"
                subprocess.run(
                    f"docker volume rm {vol} 2>/dev/null",
                    shell=True, capture_output=True, text=True,
                )
            # Remove compose network (naming convention: {project}_{network_in_yaml})
            # Template uses $prefix-net as the network name → actual name = {prefix}_{prefix}-net
            net = f"{prefix}_{prefix}-net"
            subprocess.run(
                f"docker network rm {net} 2>/dev/null",
                shell=True, capture_output=True, text=True,
            )
            # Clean up temporary yml
            tmp = f"{_TMP_DIR}/{prefix}.yml"
            if os.path.exists(tmp):
                os.unlink(tmp)
        else:
            subprocess.run(
                f"docker stop {prefix} 2>/dev/null; docker rm -f -v {prefix} 2>/dev/null",
                shell=True, capture_output=True, text=True,
            )

    def _compose_containers(self, app: str, cfg: dict, prefix: str) -> list[str]:
        """Parse all container_name fields from the compose template, substituting $prefix."""
        tpl_path = cfg.get("compose_template_file", "")
        if not tpl_path:
            return [prefix]
        if not os.path.isabs(tpl_path):
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            tpl_path = os.path.join(here, tpl_path)
        try:
            with open(tpl_path) as f:
                content = f.read()
        except OSError:
            return [prefix]
        names = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("container_name:"):
                raw = stripped.split(":", 1)[1].strip()
                # Substitute $prefix (excluding volume names of the ${prefix}_* form)
                name = raw.replace("$prefix", prefix)
                names.append(name)
        return names if names else [prefix]

    def _get(self, app: str) -> dict:
        if app not in self.apps:
            raise KeyError(f"Unknown app '{app}'. Check rollout/apps.yaml.")
        return self.apps[app]
