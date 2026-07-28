# Docker Sandbox Provider

A [NeMo Gym](../../../../README.md) sandbox provider backed by the **local Docker daemon**
via the `docker` CLI. Each sandbox is one long-lived container; the provider shells out to
`docker run` / `docker exec` / `docker cp` / `docker rm` — **no OpenSandbox server, no
control plane, no Kubernetes**. It gives real container isolation out-of-the-box for anyone
with Docker installed.

Use it for local/CI runs and single-machine setups where Docker is the available runtime.
(For an HPC node without a Docker daemon, use the [`apptainer`](../apptainer/README.md)
provider; for a managed remote pool, use `opensandbox`.)

> **Provider name:** `docker` (select it via the sandbox config; see below).

## Requirements

- The **`docker` binary** on `PATH` **and a running daemon**. The provider does not install
  or start Docker; constructing it raises `RuntimeError` if the binary is missing, and
  `create()` fails if the daemon is unreachable.
- A container **image** Docker can run (`ubuntu:22.04`, `python:3.12-slim`, a registry ref,
  …). An apptainer-style `docker://` prefix is tolerated and stripped.
- For GPUs: the NVIDIA Container Toolkit (`--gpus`). For CPU/memory limits: cgroups.

## Quick start

The provider is used through NeMo Gym's provider-neutral sandbox API
(`nemo_gym.sandbox.api`); you pick the provider with a single-key mapping and describe the
sandbox with a `SandboxSpec`.

```python
from nemo_gym.sandbox.api import Sandbox
from nemo_gym.sandbox.providers import SandboxSpec

spec = SandboxSpec(
    image="python:3.12-slim",
    workdir="/sandbox",
    env={"GREETING": "hello"},
    files={"/sandbox/input.txt": "some seed content"},
    resources={"cpu": 2, "memory_mib": 4096},
)

with Sandbox({"docker": {}}, spec) as sandbox:
    sandbox.start()
    result = sandbox.exec("echo $GREETING && cat /sandbox/input.txt")
    print(result.return_code, result.stdout)
    sandbox.upload("./local_script.sh", "/sandbox/script.sh")
    sandbox.download("/sandbox/result.txt", "./result.txt")
# leaving the `with` block removes the container
```

`AsyncSandbox` is the async equivalent (`async with` + `await`). Download anything you want
to keep **before** the sandbox stops — stopping force-removes the container.

## Selecting and configuring the provider

The provider config is a single-key mapping `{"docker": {<kwargs>}}`, grouped into three
optional sections (each accepts a plain mapping from Hydra YAML or the dataclass):

```yaml
docker:
  create:
    keepalive_shell: /bin/sh
    keepalive_cmd: "while :; do sleep 2147483647; done"
    start_timeout_s: 600
    use_init: true
    apply_resource_limits: true
    network: none          # isolation knobs (see Isolation & security)
    read_only: true
    cap_drop: ["ALL"]
    pids_limit: 512
  exec:
    default_timeout_s: 180
    concurrency: 32
  probe:
    command: printf docker-sandbox-ready
    expected_stdout: docker-sandbox-ready
    deadline_s: 60
```

### `create` — `DockerCreateConfig`

| Field | Default | Meaning |
|---|---|---|
| `keepalive_shell` | `/bin/sh` | Entrypoint the container is started with (overrides the image ENTRYPOINT). |
| `keepalive_cmd` | `while :; do sleep …; done` | Long-lived PID 1 command that keeps the container up across `exec` calls. |
| `start_timeout_s` | `600` | Max seconds for `docker run` (covers an implicit image pull; `None` = no timeout). |
| `use_init` | `true` | Add `--init` (tini) to reap zombies from many short `docker exec` processes. |
| `network` | `None` | `None` = default bridge (egress); `"none"` = no network; else a named network. |
| `read_only` | `false` | Mount the container root filesystem read-only (`--read-only`). |
| `cap_drop` | `[]` | Linux capabilities to drop (`--cap-drop`), e.g. `["ALL"]`. |
| `security_opt` | `[]` | `--security-opt` values, e.g. `["no-new-privileges"]` (blocks setuid escalation). |
| `pids_limit` | `None` | Cap the container's process count (`--pids-limit`). |
| `extra_run_args` | `[]` | Extra raw flags appended to `docker run`. |
| `apply_resource_limits` | `true` | Apply CPU/memory flags from `SandboxSpec.resources`. |

### `exec` — `DockerExecConfig`

| Field | Default | Meaning |
|---|---|---|
| `default_timeout_s` | `180` | Default per-command timeout when the caller passes none (`None` = no timeout). |
| `extra_exec_args` | `[]` | Extra raw flags appended to every `docker exec`. |
| `concurrency` | `32` | Upper bound on concurrent `docker` subprocesses (shared semaphore). |
| `exec_shell` | `null` (auto) | Shell for `<shell> -c <command>`. `null` **auto-detects `bash`** at create (for conda/SWE-bench `source`, which POSIX `sh`/dash lacks) and falls back to `sh`. Pin to `/bin/sh` or `/bin/bash` to skip the probe. |

### `probe` — `DockerProbeConfig`

Readiness-probe knobs. After `docker run`, `create` runs `command` and checks its output
before returning, so callers never get a container that can't `exec`. Set `command: null`
to skip. Same fields as the other providers: `command`, `expected_stdout`, `timeout_s`,
`deadline_s`, `stable_count`, `stable_delay_s`.

### Relevant `SandboxSpec` fields

| Field | Used for |
|---|---|
| `image` | Docker image ref (required). A `docker://` prefix is stripped. |
| `env` | `--env KEY=VALUE` at `docker run` and re-applied on every `exec`. |
| `workdir` | `-w` at `docker run` (created if absent) and default `exec` cwd. |
| `files` | Seed files written in at `start()` (via the sandbox API's `upload`). |
| `resources` | `cpu`→`--cpus`, `memory_mib`→`--memory` (+`--memory-swap`, hard cap), `gpu`→`--gpus <n>`. `disk_gib`/`gpu_type` ignored. |
| `entrypoint` | Overrides the keep-alive: the container runs this as its main process instead. |
| `provider_options` | `volumes` (a `src:dst[:opts]` string or list → `-v` mounts) and `run_args` (extra `docker run` flags), applied per-sandbox. |
| `ttl_s` | Max lifetime: the keep-alive `sleep`s for `ttl_s` and the container self-removes (`--rm`) on exit. Not enforced with a custom `entrypoint`. |

## How it works

| Step | Docker command |
|---|---|
| Create | `docker run -d --name nemo-gym-<uuid> --label nemo-gym.sandbox=1 --init [flags] --entrypoint <sh> <image> -c <keepalive>` |
| Exec | `docker exec [-w cwd] [--env…] [--user…] <name> <shell> -c <command>` |
| Upload | `mkdir -p <parent>` (via exec), then `docker cp <src> <name>:<dst>` |
| Download | `docker cp <name>:<src> <local>` |
| Status | `docker inspect -f '{{.State.Status}}' <name>` |
| Close | `docker rm -f <name>` |

- **Keep-alive + `--entrypoint`.** The container is started with `--entrypoint` pointing at
  the keep-alive so images that define their own `ENTRYPOINT` (common for SWE benchmark
  images) don't swallow it. Container state persists across `exec` calls — agents rely on it.
- **`--init`.** A naive PID 1 doesn't reap children; `--init` inserts tini so the many
  short-lived `docker exec` processes don't accumulate as zombies.
- **File transfer** uses `docker cp` (arbitrary paths, both directions). Uploaded files are
  owned by **root** — read them as root, or `chown` them for a non-root user.
- **`user`** maps to `--user` (`None`→container default, `"root"`/`0`→`--user 0`, else
  `--user <value>`).
- **Resources** map to `--cpus`, `--memory <n>m` (+ a matching `--memory-swap` so it's a hard cap,
  not 2x via swap), and `--gpus <n>`. `disk_gib` / `gpu_type` have no direct flag and are ignored.
- **Status**: `running`/`paused`→`RUNNING`, `created`/`restarting`→`STARTING`,
  `exited`/`dead`/`removing`→`STOPPED`, missing container→`STOPPED`, timeout/other→`UNKNOWN`.
- **Errors**: `exec` never raises for command failure — it returns a `SandboxExecResult` with
  the real `return_code`; a timeout or a docker-runtime failure (daemon down, container gone —
  detected via stderr markers) returns `return_code=125` with `error_type` `"timeout"` /
  `"sandbox"`.

## Isolation & security

**Docker containers share the host kernel — this is process/namespace isolation, not a VM.
It is not a hard boundary for hostile code.** For adversarial or untrusted workloads, run
Docker on a disposable VM or use a stronger runtime (gVisor `runsc`, Kata Containers, or a
microVM). For that reason:

- **Never** run with `--privileged` and don't mount the Docker socket into the sandbox
  (either would grant host root).
- **Networking** is on by default (the bridge network) because most tasks need egress
  (pip/git). For untrusted code set `network: none` to cut it entirely (or attach a locked
  internal network via `network: <name>`).
- Harden with `read_only: true`, `cap_drop: ["ALL"]`, `security_opt: ["no-new-privileges"]`, and
  `pids_limit: <n>` (fork-bomb guard). Enforce `resources` (`--cpus`/`--memory` — a hard cap, swap
  disabled) so one sandbox can't starve the host.
- **Rootless Docker** narrows the blast radius (container root ≠ host root) and is recommended
  for shared machines; `docker cp` ownership and some `--user`/cgroup behaviors differ under it.
- Every container is labeled `nemo-gym.sandbox=1`; if a run is killed uncleanly, reap leaks with
  `docker rm -f $(docker ps -aq --filter label=nemo-gym.sandbox=1)`.

## Limitations

- **`ttl_s` is a _max_ lifetime**, not a scheduler: the container self-removes when the keep-alive
  `sleep` expires. Not enforced with a custom `entrypoint` (that process owns its lifetime).
- **Distroless / no-shell images** — the default keep-alive and `sh -c` exec need a shell;
  set `keepalive_shell` / a custom `entrypoint` for images without `/bin/sh`.
- **`disk_gib` / `gpu_type`** have no direct docker flag and are ignored.
- **Runtime-failure detection is heuristic** — it keys off stderr markers, so a user command
  whose own output contains e.g. `Error response from daemon` could be misclassified.
- **Command timeouts** return a timeout result and leave the sandbox running — it is reused across a
  rollout's commands, so one command's timeout doesn't tear it down. The command's process is reaped
  with the container at `close()` (`docker rm -f` stops the whole process tree); resource limits bound
  it in the meantime.

## Development

Source: [`provider.py`](./provider.py). It implements the `SandboxProvider` protocol from
[`../base.py`](../base.py) structurally (no subclassing) and is registered under the name
`docker` in [`../registry.py`](../registry.py).

Unit tests live in
[`tests/unit_tests/test_docker_provider.py`](../../../../tests/unit_tests/test_docker_provider.py)
and run as part of the core suite — **no Docker required**:

```bash
uv venv && uv sync --extra dev
pytest tests/unit_tests/test_docker_provider.py -q
```

The suite mocks at the **subprocess boundary**: `_require_docker` is monkeypatched to a fake
path, and `DockerProvider._run` (the single chokepoint every CLI call goes through) is
replaced with a recorder that captures `argv` / `timeout_s` and returns canned
`(return_code, stdout, stderr)`, so tests assert the exact command line built for each
operation. A few tests exercise the real subprocess plumbing with harmless binaries
(`echo`, `cat`, `sleep`), each guarded with `@pytest.mark.skipif(shutil.which(...) is None)`;
real end-to-end tests are additionally guarded on a reachable Docker daemon. Async tests need
no decorator (`asyncio_mode = "auto"`).
