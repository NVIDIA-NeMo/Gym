# Enroot Sandbox Provider

A [NeMo Gym](../../../../README.md) sandbox provider backed by the local
[enroot](https://github.com/NVIDIA/enroot) CLI. It runs each sandbox as a
long-lived enroot container on the host and shells out to the `enroot` binary —
no daemon, no network service, no Kubernetes, and (unlike opensandbox) no server
or API key.

Use it when you want lightweight, **rootless** container isolation on a single
machine or HPC node where enroot is the supported container runtime (common on
NVIDIA/Slurm clusters, where it pairs with pyxis).

> **Provider name:** `enroot` (select it via the sandbox config; see below).

## Requirements

- The **`enroot` binary** must be installed and on `PATH`. The provider does
  **not** auto-install it; constructing the provider raises `RuntimeError` if it
  is missing. See the
  [enroot install guide](https://github.com/NVIDIA/enroot/blob/master/doc/installation.md).
- Unprivileged user namespaces must be enabled (`unprivileged_userns_clone=1`),
  which enroot needs for rootless operation.
- A container **image**: a local `.sqsh` squashfs, an enroot import URI, or a
  Docker image reference such as `ubuntu:22.04` or `nvcr.io/nvidia/pytorch:24.01`.
- **All enroot calls must run as the same OS user.** `enroot exec` re-enters
  namespaces owned by the launching user, so the provider (a single process)
  must own the whole container lifecycle. Switching to another *container* user
  is done with `su` inside the container.

## Quick start

The provider is used through NeMo Gym's provider-neutral sandbox API
(`nemo_gym.sandbox.api`). You pick the provider with a single-key mapping and
describe the sandbox with a `SandboxSpec`.

### Synchronous

```python
from nemo_gym.sandbox.api import Sandbox
from nemo_gym.sandbox.providers import SandboxSpec

spec = SandboxSpec(
    image="ubuntu:22.04",              # or "nvcr.io/nvidia/pytorch:24.01", or "/path/to/image.sqsh"
    workdir="/sandbox",
    env={"GREETING": "hello"},
    files={"/sandbox/input.txt": "some seed content"},
)

with Sandbox({"enroot": {}}, spec) as sandbox:
    sandbox.start()

    result = sandbox.exec("echo $GREETING && cat /sandbox/input.txt")
    print(result.return_code, result.stdout)

    sandbox.upload("./local_script.sh", "/sandbox/script.sh")
    sandbox.download("/sandbox/result.txt", "./result.txt")
# leaving the `with` block kills the container and cleans up
```

### Asynchronous

```python
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxSpec

async def run():
    spec = SandboxSpec(image="ubuntu:22.04", workdir="/sandbox")
    async with AsyncSandbox({"enroot": {}}, spec) as sandbox:
        await sandbox.start()
        result = await sandbox.exec("uname -a")
        print(result.stdout)
```

> **Lifecycle contract:** download anything you want to keep *before* the sandbox
> is stopped. Stopping is teardown — it kills the container init, removes the
> rootfs (`enroot remove -f`), and deletes the host staging directory.

## Selecting and configuring the provider

The provider config is a single-key mapping: `{"enroot": {<kwargs>}}`. The kwargs
are grouped into three optional sections, each of which accepts a plain mapping
(e.g. from Hydra YAML) or the corresponding dataclass. A ready-to-use config is
shipped at [`configs/enroot.yaml`](./configs/enroot.yaml).

```yaml
enroot:
  create:
    base_dir: null            # provider-scoped enroot home (auto = per-user /tmp dir)
    data_path: null           # ENROOT_DATA_PATH override
    cache_path: null          # ENROOT_CACHE_PATH override
    runtime_path: null        # ENROOT_RUNTIME_PATH override
    sqsh_cache_dir: null      # where imported .sqsh images are cached
    rw: true
    remap_root: false
    start_timeout_s: 600
  exec:
    default_timeout_s: 180
    concurrency: 32
  probe:
    deadline_s: 180
    stable_count: 2
```

### `create` — `EnrootCreateConfig`

| Field | Default | Meaning |
|---|---|---|
| `mount_point` | `/sandbox` | Absolute path inside the container where the host staging dir is mounted. Powers the file-transfer fast path. |
| `base_dir` | auto | Base dir for the provider-scoped enroot paths when the specific paths below are unset. Defaults to `${TMPDIR}/nemo-gym-enroot-<uid>`. |
| `data_path` / `cache_path` / `runtime_path` | env → base | Pinned `ENROOT_DATA_PATH` / `ENROOT_CACHE_PATH` / `ENROOT_RUNTIME_PATH`, passed to **every** enroot subprocess (see below). |
| `sqsh_cache_dir` | `<base>/sqsh` | Where imported squashfs images are cached (keyed by image name). |
| `rw` | `true` | Start the container with a writable root filesystem (`--rw`). |
| `remap_root` | `false` | Remap the launching user to root inside the container (`--root`). |
| `init_command` | `while true; do sleep 86400; done` | The long-lived init keeping the container alive between `exec` calls (portable across busybox/coreutils). |
| `import_timeout_s` | `1800` | Max seconds for `enroot import` (image pull/convert). |
| `create_timeout_s` | `600` | Max seconds for `enroot create` (rootfs unpack). |
| `start_timeout_s` | `600` | Max seconds to wait for the container init PID to appear. |
| `start_poll_s` | `0.5` | Polling interval while waiting for the init PID. |
| `extra_import_args` / `extra_create_args` / `extra_start_args` | `[]` | Extra raw flags appended to the respective enroot command. |

### `exec` — `EnrootExecConfig`

| Field | Default | Meaning |
|---|---|---|
| `default_timeout_s` | `180` | Default per-command timeout when the caller doesn't pass one. |
| `default_mounts` | `[]` | Extra `-m src:dst` mounts added at container start. |
| `extra_exec_args` | `[]` | Extra raw flags appended to every `enroot exec`. |
| `concurrency` | `32` | Upper bound on concurrent `enroot` subprocesses (shared semaphore). |

### `probe` — `EnrootProbeConfig`

Same shape as the apptainer provider's probe. The default probe writes to and
reads back from `/sandbox`, so a returned sandbox is guaranteed to have a live,
writable staging mount. Set `command: null` to skip the probe.

### Relevant `SandboxSpec` fields

| Field | Used for |
|---|---|
| `image` | `.sqsh` path, enroot URI, or Docker reference. Required. |
| `env` | Passed as `-e KEY=VALUE` at start and re-applied on every `exec`. |
| `workdir` | Default working directory for `exec` (applied as a `cd` prefix). |
| `files` | Seed files written into the sandbox at `start()` (via the API's `upload`). |
| `resources.gpu` | Mapped to `NVIDIA_VISIBLE_DEVICES` (see below). `cpu`/`memory_mib`/`disk_gib` are **ignored with a warning**. |
| `provider_options` | `mounts`: a `"src:dst[:type:opts]"` string or list of enroot fstab entries — extra per-sandbox mounts added at start. |
| `ttl_s` | **Not supported** — ignored with a warning. Tear down via `stop()`/`close()`. |

## How it works

### Lifecycle: one detached container per sandbox

| Step | Enroot command |
|---|---|
| Import (cached) | `enroot import -o <cache>/<hash>.sqsh docker://<image>` |
| Create rootfs | `enroot create -n nemo-gym-<uuid> <sqsh>` |
| Start (detached) | `enroot start --rw -m <staging>:/sandbox [-e ...] <name> sh -c "<init>"` |
| Exec | `enroot exec [-e ...] <pid> sh -c <command>` |
| Status | `enroot list -f` (liveness from the PID column) |
| Close | kill the start process group, then `enroot remove -f <name>` |

Containers are named `nemo-gym-<uuid>` and persist across `exec` calls, so state
written by one command is visible to the next — agents rely on this.

### Why the init is launched detached (not "daemonized" like apptainer)

Unlike `apptainer instance start`, **`enroot start` does not self-daemonize** —
it stays in the foreground for the container's entire lifetime. So the provider
launches it truly detached in its own session (`start_new_session=True`, output
to temp files) and **does not await its exit**. It keeps the process-group id
(for teardown) and then polls `enroot list -f` until the container's init PID
appears, which confirms readiness. Every subsequent `enroot exec` targets that
PID.

### Pinned `ENROOT_*` paths (correctness requirement)

enroot's default paths derive from `XDG_*`, which are often unset in server or
daemon contexts (making `ENROOT_RUNTIME_PATH` resolve to an unwritable location).
More importantly, if `ENROOT_RUNTIME_PATH` differed between the `start` call and a
later `exec`/`list`/`remove`, the running container could not be found. The
provider therefore resolves `ENROOT_DATA_PATH`, `ENROOT_CACHE_PATH`, and
`ENROOT_RUNTIME_PATH` once (from config, then existing env, then a per-user base
dir) and passes the same environment to **every** enroot subprocess.

### Image references and the `#`-registry rule

enroot's docker scheme separates the registry host with `#`, not `/`
(`docker://[USER@][REGISTRY#]IMAGE[:TAG]`). The provider translates references:

| `spec.image` | enroot import URI |
|---|---|
| `ubuntu:22.04` | `docker://ubuntu:22.04` |
| `nvcr.io/nvidia/pytorch:24.01` | `docker://nvcr.io#nvidia/pytorch:24.01` |
| `docker.io/swebench/foo:latest` | `docker://swebench/foo:latest` (Hub host dropped — the real API host is `registry-1.docker.io`, not `docker.io`) |
| `docker://…` / other scheme | passed through unchanged |
| `/path/to/image.sqsh` (or existing path) | used directly, no import |

Imports are cached under `sqsh_cache_dir` keyed by a hash of the image name, and
serialized with a per-image lock plus atomic temp-then-rename so concurrent
creates of the same image never observe a half-written squashfs.

### File transfer: a shared mounted directory

On create, the provider makes a temporary host directory and mounts it into the
container at `mount_point` (default `/sandbox`). Paths inside the mount are
read/written directly on the host side (fast path); arbitrary in-container paths
are staged in the shared folder and moved with an in-container `cp` (fallback).
Because enroot is rootless, files land back on the host owned by the launching
user.

### Resource limits

| Resource | Handling |
|---|---|
| `gpu` (truthy) | `NVIDIA_VISIBLE_DEVICES=0,1,…,n-1` set at start; the enroot NVIDIA hook injects the GPUs. |
| `cpu`, `memory_mib`, `disk_gib` | **Ignored with a warning** — standalone rootless enroot does not enforce cgroups (that is pyxis/Slurm's job). |

### Status mapping

`enroot list -f` lists created rootfs *and* running containers, distinguishing
them by the `PID` column:

- name present **with a PID** → `RUNNING`,
- name present without a PID, or name absent → `STOPPED`,
- timeout / non-zero → `UNKNOWN`.

### Error reporting

`exec` never raises for command failure; it returns a `SandboxExecResult`:

- **Normal** — the command's real `return_code`, `error_type=None`.
- **Timeout** — `return_code=125`, `error_type="timeout"`.
- **Enroot runtime failure** (container gone, nsenter error, etc., detected via
  stderr markers like `[ERROR]`) — `return_code=125`, `error_type="sandbox"`.

`125` is the sentinel `SANDBOX_RUNTIME_RETURN_CODE`, signaling "the sandbox
runtime failed" rather than "the command exited 125".

## Limitations

- **No `ttl_s`.** enroot has no native auto-expiry; the field is ignored. Manage
  lifetime with `stop()` / `close()`.
- **No CPU/memory enforcement standalone.** Use pyxis/Slurm for cgroup limits.
- **Root inside the container** requires `create.remap_root: true`; without it,
  `exec(..., user="root")` runs as the launching user.
- **Same-user requirement.** The provider process must own the container for
  `enroot exec` to work; cross-user exec is not supported.
- **Runtime-failure detection is heuristic** — it keys off stderr markers, so a
  user command whose own output contains `[ERROR]` could be misclassified.

## Development

Source: [`provider.py`](./provider.py). The provider implements the
`SandboxProvider` protocol from [`../base.py`](../base.py) structurally (no
subclassing) and is registered under the name `enroot` in
[`../registry.py`](../registry.py).

### Running the tests

The unit tests live in
[`tests/unit_tests/test_enroot_provider.py`](../../../../tests/unit_tests/test_enroot_provider.py)
and run as part of the core library test suite — no `enroot` binary required:

```bash
uv venv && uv sync --extra dev      # one-time environment setup
pytest tests/unit_tests/test_enroot_provider.py -q
```

The suite mocks at the **subprocess boundary**: `_require_enroot` is
monkeypatched to a fake path, `EnrootProvider._run` (the CLI chokepoint) is
replaced with a recorder returning canned `(return_code, stdout, stderr)`, and
`_start_detached` is stubbed with a fake process, so no real enroot or container
is ever used. Async tests need no decorator because the repo sets
`asyncio_mode = "auto"` in `pyproject.toml`.
