# Apptainer Sandbox Provider

A [NeMo Gym](../../../../README.md) sandbox provider backed by the local
[Apptainer](https://apptainer.org/) (formerly Singularity) CLI. It runs each sandbox as
a persistent Apptainer *instance* on the host and shells out to the `apptainer` binary —
no daemon, no network service, no Kubernetes.

Use it when you want lightweight, container-based isolation on a single machine or HPC
node where Apptainer is already the supported container runtime (common on clusters
where Docker is unavailable).

> **Provider name:** `apptainer` (select it via the sandbox config; see below).

## Requirements

- The **`apptainer` binary** must be installed and on `PATH`. The provider does **not**
  auto-install it; constructing the provider raises `RuntimeError` if it is missing.
  See the [Apptainer install guide](https://apptainer.org/docs/admin/main/installation.html).
- A container **image**: a local `.sif`, a URI Apptainer can pull, or a bare Docker
  image name such as `ubuntu:22.04`.
- Running commands as a different user needs Apptainer **`--fakeroot`** support.
  CPU/memory limits need cgroups delegation. See [Limitations](#limitations).

## Quick start

The provider is used through NeMo Gym's provider-neutral sandbox API
(`nemo_gym.sandbox.api`). You pick the provider with a single-key mapping and describe
the sandbox with a `SandboxSpec`.

### Synchronous

```python
from nemo_gym.sandbox.api import Sandbox
from nemo_gym.sandbox.providers import SandboxSpec

spec = SandboxSpec(
    image="docker://ubuntu:22.04",   # or "/path/to/image.sif"
    workdir="/sandbox",
    env={"GREETING": "hello"},
    files={"/sandbox/input.txt": "some seed content"},
    resources={"cpu": 2, "memory_mib": 4096},
)

with Sandbox({"apptainer": {}}, spec) as sandbox:
    sandbox.start()

    result = sandbox.exec("echo $GREETING && cat /sandbox/input.txt")
    print(result.return_code, result.stdout)

    sandbox.upload("./local_script.sh", "/sandbox/script.sh")
    sandbox.download("/sandbox/result.txt", "./result.txt")
# leaving the `with` block stops the instance and cleans up
```

### Asynchronous

```python
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxSpec

async def run():
    spec = SandboxSpec(image="docker://ubuntu:22.04", workdir="/sandbox")
    async with AsyncSandbox({"apptainer": {}}, spec) as sandbox:
        await sandbox.start()
        result = await sandbox.exec("uname -a")
        print(result.stdout)
```

> **Lifecycle contract:** download anything you want to keep *before* the sandbox is
> stopped. Stopping is teardown — it stops the (ephemeral) instance and deletes the host
> staging directory, leaving nothing behind.

## Selecting and configuring the provider

The provider config is a single-key mapping: `{"apptainer": {<kwargs>}}`. The kwargs are
grouped into three optional sections, each of which accepts a plain mapping (e.g. from
Hydra YAML) or the corresponding dataclass:

```yaml
# Provider config (the value passed as the sandbox provider)
apptainer:
  exec:
    fakeroot_for_root: true
    default_binds: ["/tmp"]
    extra_exec_args: ["--writable-tmpfs"]
    default_timeout_s: 180
    concurrency: 32
  create:
    mount_point: /sandbox
    start_timeout_s: 600
    extra_start_args: []
    apply_resource_limits: true
  probe:
    command: printf apptainer-sandbox-ready
    expected_stdout: apptainer-sandbox-ready
    deadline_s: 120
```

### `create` — `ApptainerCreateConfig`

Settings for starting the instance (`apptainer instance start`).

| Field | Default | Meaning |
|---|---|---|
| `mount_point` | `/sandbox` | Absolute path inside the container where the host staging dir is bind-mounted. Powers the file-transfer fast path. |
| `start_timeout_s` | `600` | Max seconds to wait for `instance start` (`None` = no timeout). |
| `extra_start_args` | `[]` | Extra raw flags appended to `instance start`. |
| `apply_resource_limits` | `true` | Add CPU/memory cgroup flags from `SandboxSpec.resources`. |

### `exec` — `ApptainerExecConfig`

Settings for running commands (`apptainer exec`) and global provider behavior.

| Field | Default | Meaning |
|---|---|---|
| `default_timeout_s` | `180` | Default per-command timeout when the caller doesn't pass one (`None` = no timeout). |
| `fakeroot_for_root` | `true` | When running as root, add `--fakeroot` (map the host user to root inside the container). |
| `default_binds` | `[]` | Extra `--bind host:container` mounts added at instance start. |
| `extra_exec_args` | `[]` | Extra raw flags appended to every `apptainer exec` (e.g. `--no-home`, `--writable-tmpfs`, `--contain`). |
| `concurrency` | `32` | Upper bound on concurrent `apptainer` subprocesses (shared semaphore). |

### `probe` — `ApptainerProbeConfig`

Readiness-probe knobs. After starting an instance, `create` runs `command` and checks
its output before returning the sandbox, so callers never receive a sandbox that can't
actually run commands. Set `command: null` to skip the probe.

| Field | Default | Meaning |
|---|---|---|
| `command` | `printf apptainer-sandbox-ready` | Probe command (`None` disables the probe). |
| `expected_stdout` | `apptainer-sandbox-ready` | Output the probe must produce to count as ready. |
| `timeout_s` | `30` | Per-probe-attempt timeout. |
| `deadline_s` | `None` | Overall time budget for the probe loop. |
| `stable_count` | `1` | Consecutive successes required before the sandbox is considered ready. |
| `stable_delay_s` | `0.0` | Delay between probe attempts. |

### Relevant `SandboxSpec` fields

The spec is provider-neutral; the Apptainer provider uses these fields:

| Field | Used for |
|---|---|
| `image` | Bare Docker image name, local `.sif` path, or URI Apptainer can pull. Required. |
| `env` | Passed as `--env KEY=VALUE` at instance start and every `exec`. |
| `workdir` | Default working directory for `exec` (applied as `--pwd`). |
| `files` | Seed files written into the sandbox at `start()` (handled by the sandbox API via `upload`). |
| `resources` | Mapped to cgroup flags (see below). |
| `provider_options` | `binds`: a `"src:dst[:opts]"` string or list of them — extra per-sandbox `--bind` mounts added at instance start (on top of the staging mount and `exec.default_binds`). |
| `ttl_s` | **Not supported** — ignored with a warning. Tear down via `stop()`/`close()` instead. |

## How it works

### Lifecycle: one persistent instance per sandbox

| Step | Apptainer command |
|---|---|
| Create | `apptainer instance start --bind <staging>:<mount_point> [...] <image> <name>` |
| Exec | `apptainer exec [flags] instance://<name> sh -c <command>` |
| Status | `apptainer instance list --json` |
| Close | `apptainer instance stop <name>` |

Instances are named `nemo-gym-<uuid>` and persist across `exec` calls, so state written
by one command is visible to the next — agents rely on this.

### Why `create` runs `instance start` in "daemonize" mode

`apptainer instance start` is different from every other command the provider runs: it
**launches a long-lived background instance**, then the foreground process returns. The
started instance inherits the foreground process's stdout/stderr. If we captured output
the normal way (`communicate()`, which reads the pipes until they close), the call would
block until the *instance* exits — i.e. it would appear to hang for the full
`start_timeout_s` (default 600s) even though the container came up in ~1s.

To avoid this, `_run` takes a `daemonize` flag:

| `daemonize` | Used by | Behavior |
|---|---|---|
| `False` (default) | `exec`, `status`, `close`, file copies | Pipe-captures stdout/stderr and waits for the command to finish. Supports `stdin`. |
| `True` | `create` (`instance start`) only | Captures stdout/stderr to temp files (which the lingering instance may inherit harmlessly) and waits only for the **foreground** process to exit. No `stdin`. |

`daemonize=True` is **not** a general default — it intentionally drops `stdin` support (a
background launcher has nothing to read), which commands like `exec` need to pipe in large
prompts. Only flip it on for a command that spawns a process which outlives the call.

### File transfer: a shared bind-mounted directory

On create, the provider makes a temporary host directory and bind-mounts it into the
container at `mount_point` (default `/sandbox`). This shared folder is the transfer
channel:

- **Fast path** — if the target/source path is *inside* `mount_point`, the provider
  reads/writes the host side of the shared folder directly (no container call).
- **Fallback** — for arbitrary in-container paths, the provider stages the bytes in the
  shared folder and runs an in-container `cp` (as root) to move them to/from the target.

### Running as a specific user

The neutral `user` argument to `exec` maps onto Apptainer like this:

| `user` | Behavior |
|---|---|
| `None` | Run as the default (launching) user. |
| `"root"` or `0` | Add `--fakeroot` (gated by `exec.fakeroot_for_root`). |
| other name / uid | Add `--fakeroot` and wrap the command in `su -s /bin/sh -c '<cmd>' <user>`. |

### Piping input via `stdin`

`exec` accepts an optional `stdin: bytes | None` argument (an apptainer-provider
extension beyond the neutral `SandboxProvider` protocol). When set, the bytes are piped
to the command's standard input:

```python
await provider.exec(handle, "cat > /work/out.txt", stdin=b"large payload")
```

This is the right way to pass inputs that would exceed the kernel's per-argument size
limit (`MAX_ARG_STRLEN`, ~128KB) — e.g. a long prompt — which would otherwise fail with
`E2BIG` if passed as a command-line argument. It defaults to `None`, so existing callers
are unaffected.

### Resource limits

`SandboxResources` is translated to cgroup flags on `instance start`:

| Resource | Flag |
|---|---|
| `cpu` | `--cpus <n>` |
| `memory_mib` | `--memory <n>m` |
| `gpu` (truthy) | `--nv` (NVIDIA passthrough) |
| `disk_gib`, `gpu_type` | No direct Apptainer flag — ignored. |


### Status mapping

`apptainer instance list --json` only lists *live* instances, so:

- name present → `RUNNING` (unless a `state` field says otherwise),
- name absent → `STOPPED`,
- timeout / non-zero / unparseable output → `UNKNOWN`.

### Error reporting

`exec` never raises for command failure; it returns a `SandboxExecResult`:

- **Normal** — the command's real `return_code`, `error_type=None`.
- **Timeout** — `return_code=125`, `error_type="timeout"`.
- **Apptainer runtime failure** (instance gone, etc., detected via stderr markers like
  `FATAL:`) — `return_code=125`, `error_type="sandbox"`.

`125` is the sentinel `SANDBOX_RUNTIME_RETURN_CODE`, signaling "the sandbox runtime
failed" rather than "the command exited 125".

## Limitations

- **No `ttl_s`.** Apptainer has no native auto-expiry; the field is ignored (warned once
  per relevant create). Manage lifetime with `stop()` / `close()`.
- **Numeric uids.** The `su`-based user switch expects a *username*; a bare numeric uid
  may not resolve. Prefer named users.
- **`--fakeroot` on exec.** Whether `--fakeroot` works on `exec` into an instance that
  was started *without* fakeroot varies by Apptainer version and host configuration.
- **Resource enforcement.** CPU/memory cgroup flags require cgroups v2 delegation.
  Disable them with `create.apply_resource_limits: false`.
- **Runtime-failure detection is heuristic.** It keys off stderr markers, so a user
  command whose own output contains `FATAL:` could be misclassified as a sandbox error.

## Development

Source: [`provider.py`](./provider.py). The provider implements the
`SandboxProvider` protocol from [`../base.py`](../base.py) structurally (no subclassing)
and is registered under the name `apptainer` in [`../registry.py`](../registry.py).

### Running the tests

The unit tests live in
[`tests/unit_tests/test_apptainer_provider.py`](../../../../tests/unit_tests/test_apptainer_provider.py)
and run as part of the core library test suite — no `apptainer` binary required:

```bash
uv venv && uv sync --extra dev      # one-time environment setup
pytest tests/unit_tests/test_apptainer_provider.py -q
```

Async tests need no decorator because the repo sets `asyncio_mode = "auto"` in
`pyproject.toml`.

### How the tests avoid needing Apptainer

The suite mocks at the **subprocess boundary** so it is fully hermetic:

- `_require_apptainer` is monkeypatched to return a fake path, so constructing the
  provider never checks for a real binary.
- `ApptainerProvider._run` (the single chokepoint every CLI call goes through) is
  replaced with a small recorder that captures the `argv` / `timeout_s` and returns
  canned `(return_code, stdout, stderr)`. Tests then assert the exact command line built
  for `create` / `exec` / `upload` / `download` / `status` / `close`.

This mirrors how [`test_opensandbox_provider.py`](../../../../tests/unit_tests/test_opensandbox_provider.py)
works: that suite mocks at the **SDK boundary** instead — it calls
`pytest.importorskip("tenacity")` to skip when the optional dependency is absent, then
monkeypatches `_require_opensandbox_sdk` to hand back fake `Sandbox` classes, so no real
SDK or network is ever used. Same idea, different seam: OpenSandbox talks to a remote SDK
(mock the SDK), Apptainer shells out to a local CLI (mock the subprocess). The Apptainer
provider has no optional Python dependency, so it needs no `importorskip`.

A few tests do exercise the real subprocess plumbing in `_run` using harmless system
binaries (`echo`, `cat`, `sleep`), each guarded so they skip where the binary is missing:

```python
import shutil, pytest

@pytest.mark.skipif(shutil.which("apptainer") is None, reason="apptainer not installed")
def test_real_apptainer(): ...
```
