"""Start and supervise a nemo_skills sandbox subprocess inside a Singularity container.

[CUSTOM] Adapted from resources_servers/rdkit_chemistry/sandbox_launcher.py — the subprocess
watchdog, throttling proxy, and health-check machinery below are copied from that file nearly
verbatim (it's the only place in this repo that already solves "supervise a long-lived
nemo_skills sandbox subprocess"). Generalized here beyond rdkit_chemistry's single-caller
design in three ways, all load-bearing for this module's actual use (both `ns_tools` and
`reasoning_gym` calling this independently):

1. The sandbox now runs inside `singularity exec --containall --writable-tmpfs ...` instead
   of a bare `python -m nemo_skills...` subprocess — gives real kernel-enforced filesystem
   isolation (ephemeral overlay, absolute-path damage can't reach the real host), validated
   directly against a real compute node this session. Network isolation does NOT use
   Singularity's namespace mechanism (`--net --network none`) despite earlier validation this
   session suggesting it would work -- that validation only checked reachability *from inside
   the same container invocation*, never from an external host process. Confirmed by direct
   testing (this session, after the filesystem/resource-limit work below): unprivileged users
   on this cluster may only use `--network=none`, which gives a fully isolated netns with no
   veth/bridge to the host at all -- the host cannot reach a port the container binds under
   it. `bridge`/`ptp` networking (which would allow both isolation and host reachability)
   require root or `--fakeroot`, and `--fakeroot` itself requires user namespaces, which are
   confirmed cluster-wide disabled (see the resource-limits section below). Since
   ns_tools/reasoning_gym's own process must reach this server's HTTP port from outside any
   container, real namespace-level network isolation is not achievable here -- see
   `_build_singularity_command()`'s comment and
   docs/infrastructure/engineering-notes/sandbox-container-granularity.md for the full
   finding and a scoped, not-yet-implemented Unix-domain-socket-bridging alternative that
   would recover real isolation without needing CNI/root. Network "isolation" for
   `network="none"` is therefore the same Python-level `NEMO_SKILLS_SANDBOX_BLOCK_NETWORK`
   in-process socket patch that existed before this module -- weaker (catches naive/in-process
   calls, not a deliberately adversarial subprocess-based bypass), but the only mechanism
   compatible with host reachability on this cluster. Resource limits
   (CPU-time/process-count/file-size, plus memory via the sandbox server's own internal
   mechanism -- see `_set_resource_limits()`) are applied via `resource.setrlimit()` in a
   `preexec_fn` before `singularity exec` runs — confirmed the only mechanism that works on
   this cluster; Singularity's own native `--memory`/`--cpus`/`--pids-limit` flags need
   cgroups v2 unified mode, which this cluster doesn't have anywhere (cgroups v1 confirmed on
   both login and compute nodes).
2. Cross-*process* idempotency, not just cross-thread: `ns_tools` and `reasoning_gym` are
   separate OS processes (each its own FastAPI/uvicorn server), so rdkit_chemistry's
   module-global lock (`threading.Lock` + a module-level `_sandbox_proc` singleton) only
   protects against concurrent callers *within one process* — it does nothing for two
   different processes racing to spawn the same sandbox. Idempotency here is HTTP-health-check
   first, then an `fcntl.flock`-guarded spawn, re-checking health after acquiring the lock (the
   other caller may have won the race while we waited).
3. The watchdog is health-check-based, not `Popen.poll()`-based: only the process that actually
   spawned the sandbox holds a valid `Popen` handle for it (Unix child processes belong to
   their real parent) — a *different* caller's watchdog can't `.poll()` a subprocess it never
   created. Every caller instead independently polls `/health` over HTTP and, if unhealthy,
   attempts to win the spawn lock and restart — safe with any number of concurrent watchdogs,
   since only the lock-holder actually acts.

Not adapted from rdkit_chemistry: `_ensure_packages` (on-demand `pip install`) is dropped
entirely — this project's convention is a fully pre-built venv (see
`nemo_gym/sandbox_runtime/requirements.txt`, built by `launch.sh`'s existing per-venv loop),
not lazy runtime installs. No `atexit`-based active termination either: since the sandbox is
shared across multiple independent caller processes, no single caller's exit should kill it
for the others — cleanup relies on the SLURM job's own cgroup teardown at job end, the same
mechanism that already reaps every other per-job process (vLLM, Ray workers, etc.).

Node-sharing note: this sandbox typically runs on the SAME node as vLLM generation and/or
training workers for that node's GPUs (the single `NemoGym` Ray actor is scheduled onto
whatever node Ray picks, independent of GPU topology) — it is not an isolated box. Per-process
`setrlimit()` values bound what any ONE forked session can consume, but do nothing to bound
how many sessions can be alive AT ONCE; that aggregate is what can actually starve colocated
vLLM/training processes, and it's governed by `proxy_max_concurrency` (the throttling proxy's
semaphore) and `NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT` (idle-session eviction in the sandbox
server itself, off by default upstream -- set explicitly here so a rollout that dies mid-tool-
call without calling `DELETE /sessions/<id>` doesn't leak its forked kernel for the rest of the
job). `proxy_max_concurrency`'s default below is deliberately conservative and NOT derived
from any real measurement -- tune it against actual concurrent-tool-call telemetry from a live
job before trusting it at scale.

httpx vs. aiohttp: Gym's convention (CLAUDE.md) is aiohttp for async HTTP, because httpx's
O(n^2) connection pooling hangs at high concurrency (16k+ requests). This module uses
synchronous `httpx.Client` deliberately, not `httpx.AsyncClient` -- health checks run one at a
time from a background watchdog thread (one request every ~10s) or a blocking startup call,
never concurrently, and the proxy's own forwarding is bounded by `proxy_max_concurrency`
(low double digits by design -- see the aggregate-concurrency note below). This doesn't hit
the failure mode the convention exists to prevent; switching to aiohttp here would add
complexity (this module has no running event loop of its own to attach an aiohttp session to)
for no isolation or correctness benefit.

Extension point for future non-shared-state envs (e.g. SWE-style tasks, or any env where
per-rollout filesystem state can't share one container's overlay): `ensure_sandbox()` below
has no built-in assumption that it's called once per job. A future caller can invoke it with a
per-rollout/per-session `port` and lock path (e.g. keyed by `session_id`) instead of a fixed
per-job port, getting the same Singularity isolation guarantees at rollout-scoped lifecycle --
deliberately accepting the ~2s startup cost per rollout as the tradeoff for envs that actually
need non-shared state, rather than the shared-server default used for the plain Python
sandbox here.
"""

from __future__ import annotations

import fcntl
import logging
import os
import resource
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal, Optional

import httpx


logger = logging.getLogger(__name__)

_HEALTH_POLL = 2.0
_HEALTH_TIMEOUT = 120.0
_PROXY_HEALTH_TIMEOUT = 30.0
# [CUSTOM-SANDBOX-WATCHDOG-INTERVAL 2026-10-23] was 10.0, increased to 120 to avoid 
# killing healthy processes.
_WATCHDOG_INTERVAL = 120.0
_DEFAULT_PROXY_REQUEST_TIMEOUT = 120.0
_DEFAULT_STARTUP_PROBE_TIMEOUT = 15.0
_QUICK_HEALTH_CHECK_TIMEOUT = 2.0
# Forward-proxy variables, which Singularity passes into the container even under
# --containall. Dropped for network="none" (see _spawn()).
_PROXY_ENV_VARS = tuple(f"{p}_proxy" for p in ("http", "https", "ftp", "all", "no")) + tuple(
    f"{p.upper()}_PROXY" for p in ("http", "https", "ftp", "all", "no")
)
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

# Generic validation steps -- no rdkit_chemistry-specific "can we import this domain library"
# step, since this module is shared across any resources server with a nemo_skills sandbox.
_STARTUP_PROBE_STEPS = (
    ("basic execution", "probe_value = 42\nprint(probe_value)", "42"),
    ("stateful session reuse", "print(probe_value + 1)", "43"),
)

# RLIMIT_NPROC is enforced per real-UID, SYSTEM-WIDE -- not scoped to this process's own
# subtree. Every other process this account owns on the node (Ray workers, vLLM engines, the
# resources servers themselves) counts against the SAME budget once inherited by the sandbox's
# descendants. Set generously high: this exists to catch a genuinely runaway fork bomb, not to
# tightly budget the sandbox specifically -- a tight value risks starving the rest of the job's
# legitimate processes under the same account.
#
# [CUSTOM-SANDBOX-MAX-PROCESSES 2026-10-23] Was 4096, confirmed too tight once the sandbox
# itself started creating threads with "RuntimeError: can't start new thread".
# This is far under Leonardo's per-UID ceiling (~2,059,665 on a compute node).
_DEFAULT_MAX_PROCESSES = 65536
_DEFAULT_MEMORY_LIMIT_MB = 2048
_DEFAULT_CPU_TIME_LIMIT_S = 3600  # aggregate CPU seconds for the sandbox's whole lifetime, generous since long-lived
_DEFAULT_MAX_FILE_SIZE_MB = 512

# Aggregate-concurrency knobs -- these, not the per-process rlimits above, are what actually
# protects colocated vLLM/training processes on the same node from being starved. Both are
# deliberately conservative placeholders, not derived from any real measurement -- tune against
# actual concurrent-tool-call telemetry from a live job. proxy_max_concurrency bounds in-flight
# /execute requests; SESSION_TIMEOUT bounds how long an idle forked session (one that got its
# rollout leaked, e.g. a rollout that crashed without calling DELETE /sessions/<id>) lingers
# before the sandbox server reaps it -- upstream default is disabled (0), which would let leaked
# sessions accumulate for the whole job.
_DEFAULT_PROXY_MAX_CONCURRENCY = 16
_DEFAULT_SESSION_TIMEOUT_S = 1800

# local_sandbox_server.py's __main__ is a bare `app.run(port=6000)` -- not configurable via
# env var or CLI arg. ensure_sandbox()'s `port` arg must always equal this; see the ValueError
# raised in ensure_sandbox() if it doesn't.
_SANDBOX_SERVER_HARDCODED_PORT = 6000

_proxy_port: int | None = None
_proxy_server: "_SandboxProxyServer | None" = None
_proxy_thread: threading.Thread | None = None
_proxy_lock = threading.Lock()


class _SandboxProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 512

    def __init__(
        self,
        server_address: tuple[str, int],
        upstream_port: int,
        max_concurrency: int,
        request_timeout_s: float,
        connect_retries: int,
        retry_backoff_s: float,
    ) -> None:
        super().__init__(server_address, _SandboxProxyHandler)
        self.upstream_base_url = f"http://127.0.0.1:{upstream_port}"
        self.semaphore = threading.Semaphore(max_concurrency)
        self.request_timeout_s = request_timeout_s
        self.connect_retries = connect_retries
        self.retry_backoff_s = retry_backoff_s
        self.client = httpx.Client(timeout=httpx.Timeout(request_timeout_s, connect=5.0))


class _SandboxProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_DELETE(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_GET(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_HEAD(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_PATCH(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_PUT(self) -> None:  # noqa: N802
        self._proxy_request()

    def log_message(self, fmt: str, *args) -> None:
        logger.debug("Sandbox proxy: " + fmt, *args)

    def _proxy_request(self) -> None:
        server = self.server
        assert isinstance(server, _SandboxProxyServer)

        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(content_length) if content_length else b""
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS and key.lower() not in {"content-length", "host"}
        }

        acquired = server.semaphore.acquire(timeout=server.request_timeout_s)
        if not acquired:
            self.send_error(503, "Sandbox proxy is saturated")
            return

        try:
            upstream_response = None
            last_error: Optional[Exception] = None
            for attempt in range(server.connect_retries + 1):
                try:
                    upstream_response = server.client.request(
                        self.command,
                        f"{server.upstream_base_url}{self.path}",
                        headers=headers,
                        content=body or None,
                    )
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                    last_error = e
                    if attempt >= server.connect_retries:
                        break
                    time.sleep(server.retry_backoff_s * (2**attempt))

            if upstream_response is None:
                logger.warning("Sandbox proxy upstream request failed: %s", last_error)
                self.send_error(502, f"Sandbox proxy upstream request failed: {last_error}")
                return

            response_content = upstream_response.content
            self.send_response(upstream_response.status_code)
            for key, value in upstream_response.headers.items():
                lower_key = key.lower()
                if lower_key in _HOP_BY_HOP_HEADERS or lower_key in {"content-length", "date", "server"}:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(response_content)))
            self.end_headers()
            if self.command != "HEAD" and response_content:
                self.wfile.write(response_content)
        finally:
            server.semaphore.release()


def _set_resource_limits(
    cpu_time_limit_s: int,
    max_processes: int,
    max_file_size_mb: int,
):
    """Build a preexec_fn that applies rlimits to the child before it execs singularity.

    Limits are inherited across both fork() and exec() and apply independently to each
    forked descendant (each per-rollout IPython session gets its own copy of the same
    ceiling, not a shared pool) -- see module docstring point 1.

    Deliberately does NOT set RLIMIT_AS/RLIMIT_DATA here (memory) -- local_sandbox_server.py
    already manages those itself via NEMO_SKILLS_SANDBOX_MEM_LIMIT (see _spawn()'s env dict):
    its top-level process sets its own RLIMIT_AS to 2x that value at import time, then tightens
    each forked per-session worker down to exactly that value via its own preexec_fn=set_limits
    call. Confirmed by direct testing: pre-setting RLIMIT_AS as a hard limit from OUT here
    blocks that top-level self-raise (`ValueError: not allowed to raise maximum limit` -- a
    hard limit can only ever be lowered by an unprivileged process, never raised again), so the
    sandbox server fails at import and never starts. Setting the env var instead of a rlimit
    here is also more precise: it gives each forked WORKER exactly the configured ceiling,
    independent of whatever headroom the top-level Flask server process itself needs.
    """

    def _apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit_s,) * 2)
        resource.setrlimit(resource.RLIMIT_NPROC, (max_processes,) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_size_mb * 1024 * 1024,) * 2)

    return _apply


def _build_singularity_command(
    sif_path: str,
    venv_path: str,
    uv_python_cache_dir: str,
    port: int,
) -> list[str]:
    # No --net/--network none here -- see module docstring's "Network isolation" section.
    # Confirmed on this cluster: unprivileged users may only use --network=none (bridge/ptp
    # require root or --fakeroot; --fakeroot itself requires user namespaces, which are
    # cluster-wide disabled -- see the filesystem-isolation validation earlier this session).
    # --network none gives the container a FULLY isolated netns with no veth/bridge to the
    # host at all -- confirmed directly: the host cannot reach a port the container binds
    # under it, which breaks the actual deployment topology (ns_tools/reasoning_gym's own
    # process, running outside any container, must reach this server's HTTP port). Network
    # mode is instead enforced by NEMO_SKILLS_SANDBOX_BLOCK_NETWORK (see _spawn()) -- weaker
    # than real namespace isolation, but the only mechanism compatible with host reachability
    # on this cluster. Filesystem isolation below is unaffected by this.
    cmd = ["singularity", "exec", "--containall", "--writable-tmpfs"]
    # bind the venv (holds nemo-skills-tools + flask/ipython/psutil/traitlets) and uv's own
    # shared interpreter cache -- uv-built venvs are NOT self-contained, bin/python3 is a
    # symlink chain ending at an absolute path outside the venv dir (confirmed directly this
    # session: bind-mounting only the venv fails with "stat: no such file or directory").
    # Read-only: model-generated code must not be able to pip install into (or otherwise
    # modify) the shared venv, nor the uv interpreters that other host venvs are based on.
    cmd += ["--bind", f"{venv_path}:{venv_path}:ro"]
    cmd += ["--bind", f"{uv_python_cache_dir}:{uv_python_cache_dir}:ro"]
    cmd += [sif_path]
    cmd += [
        os.path.join(venv_path, "bin", "python3"),
        "-m",
        "nemo_skills.code_execution.local_sandbox.local_sandbox_server",
    ]
    return cmd


def _spawn(
    sif_path: str,
    venv_path: str,
    uv_python_cache_dir: str,
    port: int,
    network: Literal["none", "host"],
    memory_limit_mb: int,
    cpu_time_limit_s: int,
    max_processes: int,
    max_file_size_mb: int,
    session_timeout_s: int,
) -> "subprocess.Popen":
    import subprocess

    log_path = f"/tmp/sandbox_{port}.log"
    log_file = open(log_path, "a")  # noqa: SIM115
    cmd = _build_singularity_command(sif_path, venv_path, uv_python_cache_dir, port)
    env = dict(os.environ)
    # SINGULARITYENV_ prefix, NOT plain names: --containall contains "not only file systems,
    # but also PID, IPC, and environment" (confirmed via `singularity help exec`) -- a plain
    # env var here is silently invisible inside the container (confirmed by direct testing:
    # local_sandbox_server.py logged the stock 50 GiB default even with NEMO_SKILLS_SANDBOX_
    # MEM_LIMIT set in this dict, because --containall stripped it before the process ever
    # started). SINGULARITYENV_<VAR> is Singularity's own documented mechanism for punching
    # a variable through that boundary regardless of --cleanenv/--containall -- see also
    # launch.sh's own comment on why ray_bare_metal.sub does NOT need this prefix (nothing
    # there runs inside a Singularity container; this is the first thing in this repo that
    # actually does). No NEMO_SKILLS_SANDBOX_PORT here -- confirmed by reading the source,
    # local_sandbox_server.py's bind port is a bare hardcoded `app.run(port=6000)`, no env var
    # or CLI arg reads it at all (see ensure_sandbox()'s own port==6000 guard).
    # This Python-level in-process socket patch is now the ONLY network-blocking mechanism
    # for network="none" -- see _build_singularity_command()'s comment on why real namespace
    # isolation isn't available to unprivileged users on this cluster. Must be OFF for
    # network="host" (a future dangerous-tool env that genuinely needs network, e.g. reaching
    # squidward's proxy) -- it would block that env's own legitimate calls too.
    if network == "none":
        env["SINGULARITYENV_NEMO_SKILLS_SANDBOX_BLOCK_NETWORK"] = "1"
        # Singularity forwards proxy variables even under --containall, and compute
        # nodes reach the internet only through that proxy. The in-process socket patch
        # above does not cover subprocesses (`!pip install`, curl): without the proxy
        # variables they have no route out.
        for var in _PROXY_ENV_VARS:
            for name in (var, f"SINGULARITYENV_{var}", f"APPTAINERENV_{var}"):
                env.pop(name, None)
    # Upstream default is 0 (disabled) -- reap idle forked sessions so a rollout that dies
    # mid-tool-call without calling DELETE /sessions/<id> doesn't leak its kernel for the rest
    # of the job (see the aggregate-concurrency note in the module docstring).
    env["SINGULARITYENV_NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT"] = str(session_timeout_s)
    # Memory is deliberately NOT set via our own preexec_fn rlimit -- see
    # _set_resource_limits()'s docstring for why that conflicts with the sandbox server's own
    # internal per-worker memory-limiting mechanism. This env var IS that mechanism: each
    # forked per-session worker gets tightened to exactly this many bytes via the library's
    # own preexec_fn=set_limits.
    env["SINGULARITYENV_NEMO_SKILLS_SANDBOX_MEM_LIMIT"] = str(memory_limit_mb * 1024 * 1024)
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=_set_resource_limits(cpu_time_limit_s, max_processes, max_file_size_mb),
    )
    logger.info("Sandbox spawned (pid=%d, port=%d, network=%s, log=%s)", proc.pid, port, network, log_path)
    return proc


def _check_health(port: int, timeout_s: float = _QUICK_HEALTH_CHECK_TIMEOUT) -> bool:
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(f"http://127.0.0.1:{port}/health")
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError):
        return False


def _wait_for_health(port: int, timeout_s: float = _HEALTH_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _check_health(port, timeout_s=5.0):
            return
        time.sleep(_HEALTH_POLL)
    log_tail = _tail_log(port)
    raise TimeoutError(f"Sandbox not healthy after {timeout_s}s on port {port}\n--- sandbox log tail ---\n{log_tail}")


def _tail_log(port: int, n: int = 30) -> str:
    log_path = f"/tmp/sandbox_{port}.log"
    if not os.path.exists(log_path):
        return "(no log file)"
    try:
        with open(log_path) as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except OSError as e:
        return f"(could not read log: {e})"


def _run_startup_probe(port: int, timeout_s: float = _DEFAULT_STARTUP_PROBE_TIMEOUT) -> None:
    """Issue real sandbox execution requests before serving rollout traffic."""
    session_id = f"sandbox-startup-probe-{uuid.uuid4().hex}"
    base_url = f"http://127.0.0.1:{port}"
    timeout = httpx.Timeout(timeout_s, connect=5.0)
    headers = {"X-Session-ID": session_id}

    with httpx.Client(timeout=timeout) as client:
        for step_name, generated_code, expected_stdout in _STARTUP_PROBE_STEPS:
            response = client.post(
                f"{base_url}/execute",
                headers=headers,
                json={
                    "generated_code": generated_code,
                    "timeout": timeout_s,
                    "language": "ipython",
                    "traceback_verbosity": "Plain",
                },
            )
            response.raise_for_status()
            result = response.json()

            process_status = result.get("process_status")
            stdout = (result.get("stdout") or "").strip()
            stderr = (result.get("stderr") or "").strip()
            if process_status != "completed" or stdout != expected_stdout or stderr:
                raise RuntimeError(
                    "Sandbox startup probe failed during "
                    f"{step_name!r}: process_status={process_status!r}, stdout={stdout!r}, stderr={stderr!r}"
                )

        try:
            client.delete(f"{base_url}/sessions/{session_id}")
        except httpx.HTTPError:
            logger.debug("Best-effort sandbox probe session cleanup failed", exc_info=True)

    logger.info("Sandbox startup probe passed on 127.0.0.1:%d", port)


def _spawn_lock_path(port: int) -> str:
    return f"/tmp/sandbox_{port}.spawn.lock"


def _default_uv_python_cache_dir() -> str:
    """Resolve uv's shared interpreter cache dir (respects UV_PYTHON_INSTALL_DIR)."""
    import subprocess

    try:
        result = subprocess.run(["uv", "python", "dir"], capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return os.path.expanduser("~/.local/share/uv/python")


def ensure_sandbox(
    sif_path: str,
    venv_path: str,
    uv_python_cache_dir: str | None = None,
    port: int = _SANDBOX_SERVER_HARDCODED_PORT,
    network: Literal["none", "host"] = "none",
    proxy_port: int | None = None,
    proxy_max_concurrency: int = _DEFAULT_PROXY_MAX_CONCURRENCY,
    proxy_request_timeout_s: float = _DEFAULT_PROXY_REQUEST_TIMEOUT,
    proxy_connect_retries: int = 3,
    proxy_retry_backoff_s: float = 0.25,
    memory_limit_mb: int = _DEFAULT_MEMORY_LIMIT_MB,
    cpu_time_limit_s: int = _DEFAULT_CPU_TIME_LIMIT_S,
    max_processes: int = _DEFAULT_MAX_PROCESSES,
    max_file_size_mb: int = _DEFAULT_MAX_FILE_SIZE_MB,
    session_timeout_s: int = _DEFAULT_SESSION_TIMEOUT_S,
    startup_probe_enabled: bool = True,
    startup_probe_timeout_s: float = _DEFAULT_STARTUP_PROBE_TIMEOUT,
) -> None:
    """Ensure a containerized nemo_skills sandbox is running, starting it if needed.

    Safe to call from multiple independent resources-server processes (e.g. both ns_tools
    and reasoning_gym) on the same node -- idempotent via an HTTP health check plus a
    cross-process file lock (see module docstring point 2), not a same-process singleton.

    Args:
        sif_path: Path to a pre-pulled local .sif image (never docker:// -- boost_usr_prod
            compute nodes have no internet; see nemo_gym/sandbox_runtime/ for how this is
            built ahead of time).
        venv_path: Path to the dedicated sandbox-runtime venv (nemo-skills-tools + flask +
            ipython + psutil + traitlets -- see nemo_gym/sandbox_runtime/requirements.txt).
        uv_python_cache_dir: uv's shared interpreter cache dir, bind-mounted alongside
            venv_path so the venv's symlinked interpreter resolves inside the container.
        network: "none" for pure code-execution sandboxes (default -- no legitimate reason
            for generated code to reach the network); "host" for a future dangerous-tool env
            whose subprocess genuinely needs network access (e.g. reaching squidward's proxy).
            Enforced via the sandbox server's own in-process NEMO_SKILLS_SANDBOX_BLOCK_NETWORK
            patch, not Singularity network-namespace isolation -- see module docstring point 1
            for why real namespace isolation isn't available on this cluster.
    """
    if port != _SANDBOX_SERVER_HARDCODED_PORT:
        # local_sandbox_server.py's own __main__ block is a bare `app.run(port=6000)` -- no
        # env var, no CLI arg, nothing reads a configurable port at all (confirmed by reading
        # the vendored source directly, and the hard way: a real spawn against a non-6000
        # port timed out on health for 120s with nothing actually explaining why, since the
        # server was silently listening on 6000 the whole time). A caller wanting a different
        # externally-visible port should use proxy_port instead, which IS a real, independent
        # listener this module controls (see _start_proxy()).
        raise ValueError(
            f"port={port} was requested, but local_sandbox_server.py always binds "
            f"{_SANDBOX_SERVER_HARDCODED_PORT} (hardcoded upstream, not configurable). "
            f"Use port={_SANDBOX_SERVER_HARDCODED_PORT} and set proxy_port if you need a "
            "different externally-visible port."
        )

    if uv_python_cache_dir is None:
        uv_python_cache_dir = _default_uv_python_cache_dir()

    if _check_health(port):
        logger.info("Sandbox already healthy on 127.0.0.1:%d, nothing to do", port)
    else:
        lock_path = _spawn_lock_path(port)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if _check_health(port):
                    logger.info("Sandbox became healthy while waiting for the spawn lock, skipping spawn")
                else:
                    _spawn(
                        sif_path,
                        venv_path,
                        uv_python_cache_dir,
                        port,
                        network,
                        memory_limit_mb,
                        cpu_time_limit_s,
                        max_processes,
                        max_file_size_mb,
                        session_timeout_s,
                    )
                    _wait_for_health(port)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    advertised_port = port
    if proxy_port is not None and proxy_port != port:
        _start_proxy(
            proxy_port=proxy_port,
            upstream_port=port,
            max_concurrency=proxy_max_concurrency,
            request_timeout_s=proxy_request_timeout_s,
            connect_retries=proxy_connect_retries,
            retry_backoff_s=proxy_retry_backoff_s,
        )
        _wait_for_health(proxy_port, timeout_s=_PROXY_HEALTH_TIMEOUT)
        advertised_port = proxy_port

    if startup_probe_enabled:
        _run_startup_probe(advertised_port, timeout_s=startup_probe_timeout_s)

    watchdog = threading.Thread(
        target=_watchdog,
        args=(sif_path, venv_path, uv_python_cache_dir, port, network, memory_limit_mb, cpu_time_limit_s,
              max_processes, max_file_size_mb, session_timeout_s),
        daemon=True,
        name=f"sandbox-watchdog-{port}",
    )
    watchdog.start()

    if advertised_port != port:
        logger.info("Sandbox ready on 127.0.0.1:%d via throttling proxy 127.0.0.1:%d", port, advertised_port)
    else:
        logger.info("Sandbox ready on 127.0.0.1:%d", port)


def _watchdog(
    sif_path: str,
    venv_path: str,
    uv_python_cache_dir: str,
    port: int,
    network: Literal["none", "host"],
    memory_limit_mb: int,
    cpu_time_limit_s: int,
    max_processes: int,
    max_file_size_mb: int,
    session_timeout_s: int,
) -> None:
    """Health-check-based, not Popen.poll()-based -- see module docstring point 3.

    Runs independently in every caller process; only the one that wins the spawn lock
    actually restarts, so any number of concurrent watchdogs is safe.
    """
    while True:
        time.sleep(_WATCHDOG_INTERVAL)
        if _check_health(port):
            continue
        logger.warning("Sandbox on port %d unhealthy — attempting restart...", port)
        lock_path = _spawn_lock_path(port)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if _check_health(port):
                    logger.info("Sandbox recovered via another caller, no restart needed")
                    continue
                _spawn(
                    sif_path,
                    venv_path,
                    uv_python_cache_dir,
                    port,
                    network,
                    memory_limit_mb,
                    cpu_time_limit_s,
                    max_processes,
                    max_file_size_mb,
                    session_timeout_s,
                )
                _wait_for_health(port)
                logger.info("Sandbox recovered on port %d", port)
            except (RuntimeError, TimeoutError):
                logger.exception("Sandbox failed to recover after restart on port %d", port)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def _start_proxy(
    proxy_port: int,
    upstream_port: int,
    max_concurrency: int,
    request_timeout_s: float,
    connect_retries: int,
    retry_backoff_s: float,
) -> None:
    global _proxy_port, _proxy_server, _proxy_thread

    with _proxy_lock:
        if _proxy_thread is not None and _proxy_thread.is_alive() and _proxy_port == proxy_port:
            return

        _proxy_server = _SandboxProxyServer(
            ("127.0.0.1", proxy_port),
            upstream_port=upstream_port,
            max_concurrency=max_concurrency,
            request_timeout_s=request_timeout_s,
            connect_retries=connect_retries,
            retry_backoff_s=retry_backoff_s,
        )
        _proxy_port = proxy_port
        _proxy_thread = threading.Thread(target=_proxy_server.serve_forever, daemon=True, name="sandbox-proxy")
        _proxy_thread.start()
        logger.info(
            "Sandbox proxy listening on 127.0.0.1:%d -> 127.0.0.1:%d (max_concurrency=%d)",
            proxy_port,
            upstream_port,
            max_concurrency,
        )
