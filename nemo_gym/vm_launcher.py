"""Start and supervise a per-node QEMU/KVM VM sidecar, SSH into it to run commands.

[CUSTOM] Adapted from nemo_gym/sandbox_launcher.py -- this is the same shape of problem
(lazily and idempotently ensure a long-lived, shared sidecar is running on whichever node
this process happens to be on, survive multiple independent caller processes racing to spawn
it, self-heal via a health-check-based watchdog) applied to a VM instead of a Singularity
sandbox process. See sandbox_launcher.py's own module docstring for the full reasoning behind
the cross-process idempotency (HTTP/SSH health check + fcntl.flock-guarded spawn) and
health-check-based (not Popen.poll()-based) watchdog design -- both apply here verbatim.

Why a VM at all: the SWE-bench-family dataset types (SWE-Gym, R2E-Gym, SWE-smith,
SWE-rebench V1/V2) need a writable /testbed (or /{repo_name}) inside their Apptainer
containers, but that content is root-owned from the upstream Docker build, and this cluster
has unprivileged user namespaces disabled (/proc/sys/user/max_user_namespaces = 0), so
--fakeroot isn't available and --writable-tmpfs alone can't write to those paths. Booting a
real KVM VM sidesteps this entirely: inside the guest we are real root (no --fakeroot
gymnastics needed), so a plain `apptainer exec --writable-tmpfs` against the ORIGINAL,
unmodified image just works -- confirmed directly on this cluster, including the real,
multi-TB Lustre images directory reachable from the guest via a virtio-9p passthrough mount
with negligible overhead (sub-second `singularity exec` against a real image through the 9p
mount, cold or warm).

Lifecycle: one persistent VM per node, shared by every rollout that lands there for the
job's duration -- NOT one per rollout. A per-rollout VM was the first design considered and
rejected: each boot does real work (a qcow2 overlay, a cloud-init seed, a free hostfwd port),
and if enough rollouts start near-simultaneously on one node (plausible at this codebase's
concurrency settings), that's many concurrent QEMU boots competing for the same CPU and the
same Lustre metadata server at once -- the same *shape* of problem as the per-actor-venv-build
race documented in this repo's CLAUDE.md. A persistent per-node VM pays that boot cost exactly
once, at first touch, before any rollout traffic exists to contend with it. The cost: one VM
now serves many concurrent rollouts, so a restart affects everyone in flight on that node, not
just one -- the same tradeoff sandbox_launcher.py already accepts for ns_tools/reasoning_gym,
and for the same reason (each rollout's filesystem state is already isolated one layer down,
here by Apptainer's own --containall --writable-tmpfs --pid namespaces).

What is NOT ported from the igeniusai/domyn-inspect `terminal_bench_kvm` precedent (the
existing, validated prior art for unprivileged QEMU/KVM on this exact cluster/account): its
Docker-socket-over-SSH forwarding (we have no persistent daemon to forward a socket to --
`run_in_vm` just SSHes in and runs the same one-shot `apptainer exec` command the host would
have run directly) and its redsocks/iptables transparent-proxy layer (the existing host-side
code already threads HTTP_PROXY/HTTPS_PROXY into the apptainer command string for the one
case that needs real internet, SWE-rebench-V2's eval; `run_in_vm` just exports the same vars
before the SSH hop, reusing that existing mechanism instead of adding a new one).
"""

from __future__ import annotations

import asyncio
import fcntl
import functools
import hashlib
import logging
import os
import random
import re
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import ray


logger = logging.getLogger(__name__)

_HEALTH_POLL = 2.0
_BOOT_TIMEOUT_S = 300.0
_SSH_CONNECT_TIMEOUT_S = 5

# [CUSTOM-VM-WATCHDOG-TOLERANCE 2026-11-09] A failed SSH health check used to be enough to kill
# qemu and rebuild the VM (watchdog: one failed 5 s check + one immediate re-check under the
# spawn lock; ensure_vm: the same). Under load sshd in the guest can stall for longer than that
# while qemu is perfectly alive -- job 63909244 lost every node's VM this way (each restart
# killed a live qemu, "Killing stale VM process", then hit the lib.sh key-regen bug), and every
# restart also kills every SWE rollout in flight on that node. Now:
#   * qemu exited (nothing holds the overlay disk) -> genuinely dead, restart right away;
#   * qemu alive -> restart only once the VM has been unreachable for _UNREACHABLE_RESTART_S,
#     measured from this process's last successful check (the latest proof of life), so a
#     stalled-but-alive VM is never left unusable for much more than that either. The watchdog
#     polls every _WATCHDOG_INTERVAL; while the VM is failing it times its re-checks to land on
#     the deadline. ensure_vm re-checks every _PATIENT_RETRY_GAP_S up to the same deadline
#     instead of spawning over a live qemu.
_WATCHDOG_INTERVAL = 30.0
_UNREACHABLE_RESTART_S = float(os.environ.get("VM_LAUNCHER_UNREACHABLE_RESTART_S", "180"))
_PATIENT_SSH_TIMEOUT_S = float(os.environ.get("VM_LAUNCHER_HEALTH_TIMEOUT_S", "15"))
_PATIENT_RETRY_GAP_S = 10.0
# time.monotonic() of this process's last successful health check / first failure since then
_last_healthy_at: Optional[float] = None
_first_failure_at: Optional[float] = None

# [CUSTOM-VM-WATCHDOG-ONCE 2026-11-09] ensure_vm() used to start a new watchdog thread on every
# call (every SWE container command), and Ray reuses runner worker processes, so threads piled
# up (>=13 firing in the same tick in one process in 63909244), multiplying SSH probes and the
# chances of a false "unhealthy". One watchdog per process is enough: it watches the node's VM.
_watchdog_lock = threading.Lock()
_watchdog_started = False

# The default assumes this file's actual location in the domyn-rl repo
# (RL/3rdparty/Gym-workspace/Gym/nemo_gym/vm_launcher.py, six levels under the repo root where
# scripts/prepare-env/vm/lib.sh lives) -- override via VM_LAUNCHER_LIB_SH if the Gym submodule
# is ever vendored at a different depth, rather than relying on this guess silently being wrong.
_VM_LIB_SH = os.environ.get(
    "VM_LAUNCHER_LIB_SH",
    str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "scripts" / "prepare-env" / "vm" / "lib.sh"),
)
_VM_IMAGE_DIR_DEFAULT = "/leonardo_scratch/fast/BOOST_EUROPA/images/vm"

# One VM per *node*, not per process -- every caller on a node must agree on the same
# node-local state dir regardless of which resources-server process it's in, exactly like
# sandbox_launcher.py's hardcoded port 6000 lets independent ns_tools/reasoning_gym processes
# find the same sandbox. /tmp is node-local (not shared storage), matching sandbox_launcher.py's
# own choice of /tmp for its lock/log paths.
#
# [CUSTOM-VM-STATE-DIR-JOB-SCOPE 2026-11-03] A bare, unscoped "/tmp/vm_launcher" meant a qemu
# sidecar orphaned by one job was completely invisible to the NEXT job that later landed
# on the same node: that job writes its own fresh pidfile/overlay to the SAME path, "kills" only
# the PID it just wrote itself (a no-op against the real orphan), then collides with the
# orphan's still-held write lock on disk.qcow2.
# Fix: scope the state dir per Ray-cluster-bringup instead of a bare constant, so a different
# job (or a fresh Ray cluster bring-up within the same job) can never collide with a previous
# one's leftover state.
def _cluster_scope_id() -> str:
    gcs_address = ray.get_runtime_context().gcs_address
    return hashlib.sha1(gcs_address.encode()).hexdigest()[:10]


class VMConnectionInfo:
    __slots__ = ("ssh_key", "ssh_port")

    def __init__(self, ssh_key: str, ssh_port: int) -> None:
        self.ssh_key = ssh_key
        self.ssh_port = ssh_port


@functools.lru_cache(maxsize=1)
def _work_dir() -> str:
    # Computed lazily (not at import time) and cached for the life of the process: Ray isn't
    # guaranteed initialized yet when this module is first imported, only by the time ensure_vm()
    # actually runs.
    d = f"/tmp/vm_launcher-{_cluster_scope_id()}"
    os.makedirs(d, exist_ok=True)
    return d


def _hostname() -> str:
    return os.uname().nodename


def _spawn_lock_path() -> str:
    return os.path.join(_work_dir(), "spawn.lock")


def _ssh_key_path() -> str:
    return os.path.join(_work_dir(), "id")


def _ssh_port_path() -> str:
    return os.path.join(_work_dir(), "ssh_port")


def _pid_path() -> str:
    return os.path.join(_work_dir(), "vm.pid")


def _log_path() -> str:
    return os.path.join(_work_dir(), "vm.log")


def _current_connection_info() -> Optional[VMConnectionInfo]:
    key = _ssh_key_path()
    port_path = _ssh_port_path()
    if not (os.path.exists(key) and os.path.exists(port_path)):
        return None
    with open(port_path) as f:
        port = int(f.read().strip())
    return VMConnectionInfo(ssh_key=key, ssh_port=port)


def _check_health(info: VMConnectionInfo, timeout_s: float = _SSH_CONNECT_TIMEOUT_S) -> bool:
    try:
        result = subprocess.run(
            [
                "ssh",
                "-q",
                "-i",
                info.ssh_key,
                "-p",
                str(info.ssh_port),
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                f"ConnectTimeout={int(timeout_s)}",
                "root@127.0.0.1",
                "true",
            ],
            capture_output=True,
            timeout=timeout_s + 5,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def _qemu_alive() -> bool:
    """True if some process (the VM's qemu) still holds this node's overlay disk open."""
    return bool(_fuser_pids(_overlay_disk_path()))


def _mark_healthy() -> None:
    global _last_healthy_at, _first_failure_at
    _last_healthy_at = time.monotonic()
    _first_failure_at = None


def _probe(info: Optional[VMConnectionInfo], timeout_s: float = _SSH_CONNECT_TIMEOUT_S) -> bool:
    """_check_health that also records this process's last success / first failure."""
    global _first_failure_at
    if info is not None and _check_health(info, timeout_s=timeout_s):
        _mark_healthy()
        return True
    if _first_failure_at is None:
        _first_failure_at = time.monotonic()
    return False


def _unreachable_for() -> float:
    """Seconds since the VM was last known reachable from this process (since the first failure
    if this process never saw it healthy)."""
    ref = _last_healthy_at if _last_healthy_at is not None else _first_failure_at
    return 0.0 if ref is None else time.monotonic() - ref


def _healthy_or_patient(info: Optional[VMConnectionInfo]) -> bool:
    """Health check for ensure_vm(): quick check first; if that fails while qemu is still alive,
    keep re-checking until the VM has been unreachable for _UNREACHABLE_RESTART_S before
    declaring it unhealthy -- see [CUSTOM-VM-WATCHDOG-TOLERANCE]. A VM whose qemu exited fails
    immediately."""
    if info is None:
        return False
    if _probe(info):
        return True
    while _qemu_alive() and _unreachable_for() < _UNREACHABLE_RESTART_S:
        logger.warning(
            "VM health check failed on %s but qemu is alive -- unreachable for %.0fs (limit %.0fs), re-checking",
            _hostname(),
            _unreachable_for(),
            _UNREACHABLE_RESTART_S,
        )
        time.sleep(_PATIENT_RETRY_GAP_S)
        if _probe(info, timeout_s=_PATIENT_SSH_TIMEOUT_S):
            return True
    return False


def _tail_log(n: int = 60) -> str:
    path = _log_path()
    if not os.path.exists(path):
        return "(no log file)"
    try:
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except OSError as e:
        return f"(could not read log: {e})"


def _overlay_disk_path() -> str:
    """The fixed path _spawn() always builds its disk overlay at -- see _kill_stale_vm."""
    return os.path.join(_work_dir(), "disk.qcow2")


def _fuser_pids(path: str) -> list[int]:
    """Host PIDs currently holding `path` open, via `fuser`.

    Resolves to the real host PID even when the holder is running inside a PID-namespaced
    container (e.g. qemu under `singularity exec --containall`) -- unlike reading a pidfile the
    process itself wrote, which only ever reflects the PID as seen from *inside* its own
    namespace. See _kill_stale_vm's docstring for why that distinction matters here.
    """
    if not os.path.exists(path):
        return []
    try:
        result = subprocess.run(["fuser", path], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    pids = []
    for token in result.stdout.split():
        match = re.match(r"(\d+)", token)
        if match:
            pids.append(int(match.group(1)))
    return pids


def _kill_stale_vm() -> None:
    """[CUSTOM-VM-WATCHDOG-RESTART-FIX 2026-10-28] _spawn() always builds its disk overlay at
    the same fixed path (_work_dir()/disk.qcow2) regardless of whether this is a fresh first
    boot on this node or a watchdog-triggered restart of an unhealthy VM. If the previous VM's
    qemu-system-x86_64 process is still alive -- confirmed to happen in practice: a VM going
    `healthy=False` (SSH stops responding, almost certainly transient resource pressure, not a
    crash) does NOT mean the process has exited -- `vm_make_overlay`'s `qemu-img create` then
    fails outright: "Failed to get 'write' lock. Is another process using the image?", and the
    whole restart attempt aborts with no retry. Observed hitting this on two separate nodes in a
    single real job (62542185): both recovered on their own moments later (same SSH port, no new
    DEBUG_VM_BOOT success line), meaning the "restart" was chasing a VM that was never really
    dead -- but if a VM ever DOES genuinely crash instead of just stalling, there would be no
    working recovery path at all and that node would be stuck for the rest of the job.

    Fix: before ever touching the overlay, make sure nothing is still holding it.

    [CUSTOM-VM-FUSER-STALE-CHECK 2026-11-04] The original implementation here read the last-known
    QEMU PID from {work_dir}/vm.pid (written by vm_start() from QEMU's own -pidfile) and checked
    it with os.kill(pid, 0). That never actually worked: qemu runs inside `singularity exec
    --containall`, which gives it its own PID namespace, so -pidfile writes the PID *as seen
    inside that namespace* -- always a low number like 13 -- never the real host PID. Checking
    that value on the host either hits an unrelated, root-owned low-numbered kernel thread
    (PermissionError, silently treated below as "not ours, don't touch") or a coincidentally
    already-dead PID -- either way this was a guaranteed no-op on every call. Confirmed directly
    on a live job: zero "Killing stale VM process" log lines were ever printed despite real
    orphaned qemu processes holding the overlay's write lock the whole time.
    Fix: resolve the actual holder(s) via `fuser` against the overlay disk file itself, which
    resolves to the correct HOST pid regardless of the container's PID namespace -- confirmed
    directly against a live, namespaced qemu process (`fuser -v disk.qcow2` found the real host
    PID; `vm.pid` on the same node read the namespace-local "13"). A completely fresh node (no
    overlay file yet, or one nothing currently has open) is a silent no-op.
    """
    overlay_path = _overlay_disk_path()
    pids = _fuser_pids(overlay_path)
    if not pids:
        return
    for pid in pids:
        logger.warning("Killing stale VM process pid=%d (holding %s) before respawning on %s", pid, overlay_path, _hostname())
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    for _ in range(50):  # 5s grace period, matching vm_stop()'s own poll cadence in lib.sh
        time.sleep(0.1)
        if not _fuser_pids(overlay_path):
            return
    for pid in _fuser_pids(overlay_path):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _spawn(
    golden_image_dir: str,
    lustre_share_roots: list[str],
    block_device_mounts: Optional[list[tuple[str, str]]] = None,
) -> None:
    """Boot this node's VM. Blocks until SSH is reachable or _BOOT_TIMEOUT_S elapses.

    Runs scripts/vm/lib.sh's vm_* helpers via a small inline bash script rather than
    reimplementing the boot sequence in Python -- the shell helpers are the same ones this
    change's build/smoke-test scripts use and were validated directly against a real compute
    node, including the virtio-9p share(s) and the cloud-init seed. Keeping one implementation
    (shell) for "how to boot this VM" avoids a second, Python-side copy that could drift from
    what was actually tested.

    block_device_mounts: (host_image_path, guest_mount_path) pairs for read-only ext4 images
    attached as extra virtio-blk drives (see scripts/vm/build_openhands_image.sh) -- used for
    directories with too many small files for 9p to serve without stalling (confirmed directly:
    tuned 9p msize/cache and even a newer QEMU with the 9pfs fid-hashtable fix both still stalled
    on importing OpenHands' full Python dependency tree; a dedicated local block device did not,
    in both single and concurrency=8 testing). Mounted AFTER the 9p shares, at the guest's exact
    absolute path, so it shadows just that one subtree of a 9p-shared parent (e.g. OpenHands'
    setup dir shadowing part of the wider Lustre share) -- same real-path convention as
    [CUSTOM-VM-MOUNT-SAME-PATH] below.
    """
    _kill_stale_vm()
    work = _work_dir()
    log_file = open(_log_path(), "a")  # noqa: SIM115
    qemu_dir = os.path.join(golden_image_dir, "qemu")
    golden = os.path.join(golden_image_dir, "golden.qcow2")
    block_device_mounts = block_device_mounts or []
    shares = " ".join(shlex.quote(f"{root}:{_mount_tag(root)}") for root in lustre_share_roots)
    # qemu.sh (lib.sh) only binds VM_QEMU_PREFIX_DIR and /dev/kvm into the Singularity container
    # QEMU itself runs inside -- every other host path QEMU needs to touch (the golden disk, the
    # per-node work dir, each Lustre root being shared via -virtfs, and each extra block-device
    # image's directory) has to be listed here explicitly, or qemu-system-x86_64 fails with a
    # plain "No such file or directory" that looks like a missing file rather than a missing
    # bind (hit directly while building this module).
    extra_binds = ":".join(
        [
            golden_image_dir,
            work,
            *lustre_share_roots,
            *{os.path.dirname(img) for img, _ in block_device_mounts},
        ]
    )
    extra_drives = ":".join(img for img, _ in block_device_mounts)
    # [CUSTOM-VM-MOUNT-SAME-PATH 2026-10-27] Each share must be mounted in the guest at the EXACT
    # same absolute path it has on the host, not some other guest-side path (e.g. /mnt/<tag>) --
    # that was this module's first real bug, caught by an actual job run: _build_apptainer_command
    # constructs --mount/container-path strings using the real host absolute paths (e.g.
    # /leonardo_scratch/large/.../images/swesmith/<instance>.sif), so if the guest doesn't have
    # something real at that exact path, `singularity exec` fails with "could not open image ...
    # lstat /leonardo_scratch: no such file or directory" -- which is exactly what happened.
    # -virtfs only makes a share AVAILABLE to the guest (via its mount_tag); it does not mount it
    # anywhere on its own, so this loop runs once per boot, right after SSH comes up.
    mount_cmds = "\n".join(
        f'vm_ssh "$WORK/id" "$VM_SSH_PORT" "mkdir -p {shlex.quote(root)} && '
        f'mount -t 9p -o trans=virtio,version=9p2000.L {_mount_tag(root)} {shlex.quote(root)}"'
        for root in lustre_share_roots
    )
    # Block devices attach, in VM_EXTRA_DRIVES order, right after vda=disk/vdb=seed -- so the
    # Nth entry here is /dev/vd<c, d, e, ...>. Mounted AFTER the 9p loop above (see docstring).
    block_device_cmds = "\n".join(
        f'vm_ssh "$WORK/id" "$VM_SSH_PORT" "mkdir -p {shlex.quote(mount_path)} && '
        f'mount -o ro /dev/vd{chr(ord("c") + i)} {shlex.quote(mount_path)}"'
        for i, (_, mount_path) in enumerate(block_device_mounts)
    )
    script = f"""
set -euo pipefail
export VM_QEMU_PREFIX_DIR={shlex.quote(os.path.join(qemu_dir, "prefix"))}
export VM_QEMU_SIF={shlex.quote(os.path.join(qemu_dir, "debian-bookworm.sif"))}
export VM_EXTRA_BINDS={shlex.quote(extra_binds)}
export VM_EXTRA_DRIVES={shlex.quote(extra_drives)}
source {shlex.quote(_VM_LIB_SH)}
WORK={shlex.quote(work)}
vm_make_overlay {shlex.quote(golden)} "$WORK/disk.qcow2"
vm_make_key "$WORK"
vm_root_user_data "$WORK/user-data" "$WORK/id.pub"
vm_make_seed "$WORK/user-data" "$WORK/seed.img" "$WORK"
vm_start {_vcpus()} {_mem_mb()} "$WORK/disk.qcow2" "$WORK/seed.img" "$WORK" {shares}
echo "$VM_SSH_PORT" > {shlex.quote(_ssh_port_path())}
echo "$VM_PID" > {shlex.quote(_pid_path())}
vm_wait_ssh "$WORK/id" "$VM_SSH_PORT" "$VM_PID" "$WORK" {int(_BOOT_TIMEOUT_S)}
{mount_cmds}
{block_device_cmds}
"""
    result = subprocess.run(["bash", "-c", script], stdout=log_file, stderr=subprocess.STDOUT)
    log_file.close()
    if result.returncode != 0:
        raise RuntimeError(f"VM boot failed (see {_log_path()}):\n{_tail_log()}")


def _mount_tag(host_path: str) -> str:
    # virtio-9p mount tags are limited in length/charset; a short stable hash of the path is
    # simpler than sanitizing the path itself and avoids collisions between similarly-named
    # roots (e.g. two different users' scratch dirs).
    import hashlib

    return "vmshare" + hashlib.sha1(host_path.encode()).hexdigest()[:8]  # noqa: S324 (not security-sensitive)


def _vcpus() -> int:
    # [CUSTOM-VM-SIZING placeholder] Not yet derived from real measurement -- see this change's
    # plan doc, "RAM sizing is now a genuinely open, unmeasured question". A modest fixed
    # allocation for now; revisit once real per-node concurrent-rollout telemetry exists.
    return int(os.environ.get("VM_LAUNCHER_VCPUS", "8"))


def _mem_mb() -> int:
    # [CUSTOM-VM-SIZING placeholder] Same caveat as _vcpus() -- --writable-tmpfs writes are
    # RAM-backed, so this has to accommodate the AGGREGATE working-tree/build-artifact
    # footprint of however many concurrent rollouts land on this node at once, not just one.
    # Not yet measured; this default is a starting point to unblock implementation, not a
    # validated sizing decision -- do not scale up real traffic onto this without first
    # measuring real per-rollout --writable-tmpfs usage (see the plan's Verification step 1).
    return int(os.environ.get("VM_LAUNCHER_MEM_MB", "32768"))


def mount_tag_for(host_path: str) -> str:
    """Public accessor for the virtio-9p mount_tag `_spawn` uses for `host_path` -- a pure
    internal protocol identifier, not a guest filesystem path (each share is mounted in the
    guest at the SAME absolute path as the host, see [CUSTOM-VM-MOUNT-SAME-PATH] in `_spawn`,
    so callers building mount strings never need this; it's exposed mainly for tests/debugging
    that want to inspect what tag a given root maps to)."""
    return _mount_tag(host_path)


def ensure_vm(
    golden_image_dir: str = _VM_IMAGE_DIR_DEFAULT,
    lustre_share_roots: Optional[list[str]] = None,
    block_device_mounts: Optional[list[tuple[str, str]]] = None,
) -> VMConnectionInfo:
    """Ensure this node's VM sidecar is running, booting it if needed.

    Safe to call from multiple independent rollouts/processes on the same node --
    idempotent via an SSH health check plus a cross-process file lock (see
    sandbox_launcher.py's ensure_sandbox() for the identical pattern), not a same-process
    singleton. Cheap after the first call on a given node: a health check, not a fresh boot.

    Args:
        golden_image_dir: directory holding golden.qcow2 and the unprivileged QEMU install
            (built once, offline, by scripts/vm/build_golden_vm_image.sh).
        lustre_share_roots: host directories to share into the guest via virtio-9p at boot,
            at a stable per-path mount tag (see mount_tag_for). Only read on the first
            (spawning) call on a node -- a VM already running was booted with whatever roots
            an earlier caller requested; passing a different set later has no effect until
            the VM is restarted. Callers needing a new root added should restart the node's
            VM (kill its pid file) rather than relying on a later ensure_vm() call to add it.
        block_device_mounts: (host_image_path, guest_mount_path) pairs attached as extra
            read-only virtio-blk drives, mounted at guest_mount_path after the 9p shares above
            -- see _spawn's docstring. Same "only read on the first/spawning call" caveat as
            lustre_share_roots.
    """
    lustre_share_roots = lustre_share_roots or []
    block_device_mounts = block_device_mounts or []
    info = _current_connection_info()
    if _healthy_or_patient(info):
        logger.info("VM already healthy on 127.0.0.1:%d, nothing to do", info.ssh_port)
    else:
        lock_path = _spawn_lock_path()
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                info = _current_connection_info()
                if _probe(info):
                    logger.info("VM became healthy while waiting for the spawn lock, skipping spawn")
                else:
                    # [CUSTOM-VM-BOOT-JITTER-RETRY 2026-11-03] Every node's first rollout tends to
                    # land within the same few seconds of each other right after a job starts or
                    # resumes, so every node's first (cold) VM boot was observed firing in the same
                    # ~20s window across the whole cluster -- confirmed directly: 132-424 boot
                    # timeouts per job, all clustered into one 18-27s span. A real boot is cheap on
                    # an uncontended host, but dozens of simultaneous cold boots compete for the
                    # same CPU at once. A small random stagger before the first attempt spreads that
                    # demand out; a single retry (after the storm this node was part of has likely
                    # already cleared) recovers the common case where the first attempt lost that
                    # race anyway. Both are bounded: the stagger only delays this node's first-ever
                    # boot (a one-time, per-node cost -- every later call on this node just hits the
                    # fast health-check path above), and the retry means one cold boot can cost at
                    # most ~2x _BOOT_TIMEOUT_S instead of failing permanently after one attempt.
                    boot_jitter_max_s = float(os.environ.get("VM_LAUNCHER_BOOT_JITTER_MAX_S", "20"))
                    if boot_jitter_max_s > 0:
                        delay = random.uniform(0, boot_jitter_max_s)
                        logger.info(
                            "Staggering VM boot on %s by %.1fs to avoid a cluster-wide cold-start storm",
                            _hostname(),
                            delay,
                        )
                        time.sleep(delay)

                    last_error: Optional[RuntimeError] = None
                    for attempt in range(2):
                        if attempt > 0:
                            logger.warning(
                                "VM boot attempt %d failed on %s, retrying once: %s", attempt, _hostname(), last_error
                            )
                            time.sleep(5)
                        try:
                            _spawn(golden_image_dir, lustre_share_roots, block_device_mounts)
                            info = _current_connection_info()
                            if _probe(info):
                                last_error = None
                                break
                            last_error = RuntimeError(
                                f"VM did not become healthy after boot (see {_log_path()}):\n{_tail_log()}"
                            )
                        except RuntimeError as e:
                            last_error = e
                    if last_error is not None:
                        raise last_error
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    _start_watchdog_once(golden_image_dir, lustre_share_roots, block_device_mounts)

    logger.info("VM ready on 127.0.0.1:%d", info.ssh_port)
    return info


def _watchdog(
    golden_image_dir: str,
    lustre_share_roots: list[str],
    block_device_mounts: Optional[list[tuple[str, str]]] = None,
) -> None:
    """Health-check-based, not Popen.poll()-based -- same reasoning as
    sandbox_launcher.py's _watchdog: only the process that spawned the VM holds a valid
    Popen/pid handle for it, so every caller independently polls over SSH instead. Mitigates
    this design's main accepted cost (one VM restart affects every rollout in flight on that
    node) by making a VM failure self-healing rather than silently wedging the node.
    """
    delay = _WATCHDOG_INTERVAL
    while True:
        time.sleep(delay)
        delay = _watchdog_step(golden_image_dir, lustre_share_roots, block_device_mounts)


def _start_watchdog_once(
    golden_image_dir: str,
    lustre_share_roots: list[str],
    block_device_mounts: Optional[list[tuple[str, str]]],
) -> None:
    """Start this process's single watchdog thread -- see [CUSTOM-VM-WATCHDOG-ONCE]."""
    global _watchdog_started
    with _watchdog_lock:
        if _watchdog_started:
            return
        threading.Thread(
            target=_watchdog,
            args=(golden_image_dir, lustre_share_roots, block_device_mounts),
            daemon=True,
            name="vm-watchdog",
        ).start()
        _watchdog_started = True


def _watchdog_step(
    golden_image_dir: str,
    lustre_share_roots: list[str],
    block_device_mounts: Optional[list[tuple[str, str]]],
) -> float:
    """One watchdog tick; returns how long to sleep before the next one.

    [CUSTOM-VM-WATCHDOG-TOLERANCE] Restarts right away when qemu has exited (or the VM was never
    booted); a live qemu only once the VM has been unreachable for _UNREACHABLE_RESTART_S. While
    it is failing, the next check is scheduled so that it completes around that deadline.
    """
    info = _current_connection_info()
    was_failing = _first_failure_at is not None
    if _probe(info, timeout_s=_PATIENT_SSH_TIMEOUT_S):
        if was_failing:
            logger.warning("VM on %s responsive again", _hostname())
        return _WATCHDOG_INTERVAL
    down = _unreachable_for()
    qemu_alive = _qemu_alive()
    if info is not None and qemu_alive and down < _UNREACHABLE_RESTART_S:
        logger.warning(
            "VM on %s unreachable for %.0fs (limit %.0fs) but qemu is alive -- not restarting yet",
            _hostname(),
            down,
            _UNREACHABLE_RESTART_S,
        )
        check_duration = _PATIENT_SSH_TIMEOUT_S + 5  # _check_health's own subprocess timeout
        return max(1.0, min(_WATCHDOG_INTERVAL, _UNREACHABLE_RESTART_S - down - check_duration))
    logger.warning(
        "VM unhealthy or not yet booted on %s (unreachable for %.0fs, qemu alive: %s) -- attempting restart...",
        _hostname(),
        down,
        qemu_alive,
    )
    lock_path = _spawn_lock_path()
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            info = _current_connection_info()
            if _probe(info):
                logger.warning("VM on %s recovered via another caller, no restart needed", _hostname())
                return _WATCHDOG_INTERVAL
            _spawn(golden_image_dir, lustre_share_roots, block_device_mounts)
            _mark_healthy()  # _spawn returns only once vm_wait_ssh succeeded
            logger.warning("VM on %s restarted and healthy", _hostname())
        except (RuntimeError, OSError):
            logger.exception("VM failed to recover after restart")
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return _WATCHDOG_INTERVAL


async def run_in_vm(
    info: VMConnectionInfo,
    command_str: str,
    env: Optional[dict[str, str]] = None,
    **popen_kwargs,
) -> "asyncio.subprocess.Process":
    """SSH into this node's VM and run `command_str` -- a near drop-in replacement for the
    existing `asyncio.create_subprocess_shell(apptainer_cmd, ...)` call sites in
    responses_api_agents/swe_agents/app.py, not a new execution model: these remain one-shot
    subprocess invocations, just one SSH hop away. No socket-forwarding/daemon complexity
    needed (unlike Docker in the terminal_bench_kvm precedent) because there's no persistent
    daemon on the other end, just the same apptainer exec that would have run directly on the
    host.

    `env` vars are exported in the remote shell before `command_str` runs (SSH does not punch
    through arbitrary environment variables by default) -- used for the same HTTP_PROXY/
    HTTPS_PROXY passthrough app.py's _build_apptainer_command already does for SWE-rebench-V2.
    """
    env_prefix = ""
    if env:
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " "
    remote_cmd = f"{env_prefix}{command_str}"
    ssh_cmd = (
        f"ssh -q -i {shlex.quote(info.ssh_key)} -p {info.ssh_port} -o BatchMode=yes "
        f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 "
        f"root@127.0.0.1 {shlex.quote(remote_cmd)}"
    )
    return await asyncio.create_subprocess_shell(ssh_cmd, **popen_kwargs)
