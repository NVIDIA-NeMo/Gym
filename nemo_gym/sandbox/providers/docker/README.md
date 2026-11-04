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
    ports=[8000],
)

with Sandbox({"docker": {}}, spec) as sandbox:
    sandbox.start()
    result = sandbox.exec("echo $GREETING && cat /sandbox/input.txt")
    print(sandbox.endpoint(8000).endpoint)
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
    publish_host: 127.0.0.1
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
| `read_only` | `false` | Mount the container root filesystem read-only (`--read-only`). `/tmp` is then read-only too unless a tmpfs is mounted there (`extra_run_args: ["--tmpfs", "/tmp"]`); PTY sessions keep working, their pid record moves to `/dev/shm` (see PTY sessions). |
| `cap_drop` | `[]` | Linux capabilities to drop (`--cap-drop`), e.g. `["ALL"]`. |
| `security_opt` | `[]` | `--security-opt` values, e.g. `["no-new-privileges"]` (blocks setuid escalation). |
| `pids_limit` | `None` | Cap the container's process count (`--pids-limit`). |
| `extra_run_args` | `[]` | Extra raw flags appended to `docker run`. |
| `apply_resource_limits` | `true` | Apply CPU/memory flags from `SandboxSpec.resources`. |
| `publish_host` | `127.0.0.1` | Host address used for dynamic mappings declared by `SandboxSpec.ports`. |

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
| `ports` | Container TCP ports published on dynamic host ports and resolved with `sandbox.endpoint(port)`. |
| `provider_options` | `volumes` (a `src:dst[:opts]` string or list → `-v` mounts) and `run_args` (extra `docker run` flags), applied per-sandbox. |
| `ttl_s` | Max lifetime: the keep-alive `sleep`s for `ttl_s` and the container self-removes (`--rm`) on exit. Not enforced with a custom `entrypoint`. |

## How it works

| Step | Docker command |
|---|---|
| Create | `docker run -d --name nemo-gym-<uuid> --label nemo-gym.sandbox=1 --init [flags] --entrypoint <sh> <image> -c <keepalive>` |
| Exec | `docker exec [-w cwd] [--env…] [--user…] <name> <shell> -c <command>` |
| PTY session | `docker exec -i [-t] [-w cwd] [--env…] --env NEMO_GYM_PTY_SESSION=<id> [--user…] <name> <shell> -c <launcher> <pid file> <fallback pid file> <program>`, the client under a host pty |
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
- **PTY sessions** (`sandbox.pty.create` / `sandbox.pty.exec`, `SupportsSandboxPty`): one `docker exec -i -t`
  client per session, run under a host pseudo-terminal so the CLI allocates a terminal in the container;
  `rows`/`cols` size it before start and `resize()` re-sizes it through `SIGWINCH`; `pty=False` uses pipes
  with stdout and stderr split. The program starts through a launcher that writes its pid to
  `/tmp/.nemo-gym-pty-<id>.pid` in the container, or to `/dev/shm/.nemo-gym-pty-<id>.pid` when `/tmp`
  refuses the write (`read_only: true` without a tmpfs on `/tmp`; `/dev/shm` is a writable tmpfs in every
  container not run with `--ipc=none`), which `send_signal()` (`kill` through a second exec, as
  root, to the terminal's foreground process group in PTY mode, so it reaches the job an interactive shell
  is running as a key press would; to the process's own group and the process on pipes) and `close()`
  (of a live session: `SIGHUP` to the group, up to a second for a shell to relay it to its jobs, then
  `SIGKILL` to every process left in the recorded process's session; both paths are removed either way) use.
  A session whose pid can be recorded in neither place is not started: the launcher prints
  `nemo-gym: cannot record the pid of this PTY session in ...` on stderr (the terminal in PTY mode) and exits 1,
  so the session ends at once with that message rather than running where it could be neither signalled nor
  cleaned up (`create_pty` itself returns as soon as the client is spawned). The read of the record waits up to two
  seconds for the launcher to write it, since a `send_signal()`/`close()` issued right after `create_pty()` reaches
  the container through a second exec that races the session's own. It reads a regular file only (a FIFO or a
  device at the path is refused at once, the open file is tested again on its own descriptor, 32 bytes at most are
  read) under a 30 s exec timeout of its own (`PID_READ_TIMEOUT_S`), so the session's user cannot stall
  `send_signal()`/`close()` on the record.
  The file is the session user's (content and owner alike are forgeable: a hard link to a file of
  another user keeps that user as owner), so it is read once, as the session's own user (its process's
  `/proc/<pid>/environ` is readable by that uid, not by a root without `CAP_SYS_PTRACE`, which Docker does not
  grant by default; the `kill` and the cleanup run as root), and followed only to a plain pid above 1
  whose process carries the session's `NEMO_GYM_PTY_SESSION=<id>` marker in its environment (set by the
  exec's own `--env`, so only the session's own tree has it), has `PPid` 0 (injected by the runtime, so
  not a child started inside the container) and, in PTY mode, leads its session; the `kill` lines are
  busybox-compatible (`kill -INT -<pgid>`). `wait_exit()` returns the program's exit status as `docker exec`
  reports it; a client that exits without having started the program (the daemon refused the exec: the container
  paused or gone, the daemon unreachable, told by the client's stderr with the same lines that make `exec()` return
  `error_type="sandbox"`) ends the session with `SandboxPtyError` from `wait_exit()` and `read()` instead, so
  `sandbox.pty.exec()` raises where `sandbox.exec()` returns the runtime sentinel. In PTY mode the client's own
  stderr is a pipe of the session's, kept out of the terminal's output for that; in pipe mode it is the program's
  stderr too, classified as `exec()` classifies its stderr. Sessions live in the Gym process: no re-attach by id,
  and a detached exec runs attached. See [`pty.py`](./pty.py).
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

- **PTY sessions are local** to the process that opened them: another process cannot attach to one by
  id (`SupportsSandboxPtyAttach` is not implemented), and a session dies with the Gym process. The
  container's own process is ended at `close()` through its recorded pid; a session the process never
  closed leaves that process running until the container is removed. The recorded pid is trusted only
  for a process with `PPid` 0 (the runtime's execs), so a container sharing the host's pid namespace
  (`--pid=host` in `extra_run_args`) cannot be signalled or cleaned up through `send_signal()`/`close()`, nor can a
  session whose program changes its user or `exec`s a setuid program (`su`, `sudo -i`): its environment is no
  longer readable by the session's user, so `send_signal()` raises and `close()` leaves it to the container.
  A record the session's user has made unusable (rewritten, relinked, or no longer a regular file) is refused, not
  waited on: `send_signal()` raises within the two-second wait and `close()` only removes it.
  The pid record needs a writable `/tmp` or `/dev/shm` in the container: under `read_only: true` without a
  tmpfs on `/tmp` it goes to `/dev/shm`, and a container with neither (`--ipc=none` as well) cannot host a PTY
  session at all; the session then ends at once with `nemo-gym: cannot record the pid of this PTY session`
  on its terminal. Under `--ipc=host` the `/dev/shm` record is a file on the host's `/dev/shm` for the session's
  life.

## Development

Source: [`provider.py`](./provider.py) and [`pty.py`](./pty.py). It implements the `SandboxProvider` protocol from
[`../base.py`](../base.py) structurally (no subclassing) and is registered under the name
`docker` in [`../registry.py`](../registry.py).

Unit tests live in
[`tests/unit_tests/test_docker_provider.py`](../../../../tests/unit_tests/test_docker_provider.py) and
[`tests/unit_tests/test_docker_pty.py`](../../../../tests/unit_tests/test_docker_pty.py)
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
