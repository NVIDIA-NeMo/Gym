# bwrap sandbox provider

Runs each sandboxed command under [bubblewrap](https://github.com/containers/bubblewrap)
on the host of the Gym process, usually a Slurm/pyxis container of the task image.
It needs no daemon and no image unpack, and it can be connected from another Gym
server on the same host and uid.

```yaml
sandbox:
  bwrap:
    bwrap_path: /abs/path/to/bwrap-0.11.2/bwrap
    base_dir: /raid/nemo-gym-bwrap   # node-local
    network: host                    # or none (default)
```

A full annotated config is in [`configs/bwrap.yaml`](configs/bwrap.yaml).

## Model

- **A sandbox is a directory**: `<base_dir>/<sandbox_id>/{workspace,tmp,home}`, plus
  two host-only entries, `run/` (process registry) and `sandbox.json` (state). There is
  no long-lived process. `create` makes the directory and, by default, runs `true`
  once as a readiness probe.
- **Each `exec` spawns its own `bwrap`**:

  ```
  bwrap --die-with-parent --new-session
        --unshare-user --unshare-pid --unshare-ipc --unshare-uts [--unshare-net]
        --cap-drop ALL
        --ro-bind <system paths>... [--ro-bind <network: host paths>...]
        --proc /proc --dev /dev
        --bind <tmp> /tmp --bind <home> <home_path>
        --bind <workspace> <workdir> [--bind <workspace> <workspace host path>]
        [--ro-bind-fd|--bind-fd <open source fd> <mount dst>]...
        --remount-ro /
        --chdir <cwd or workdir> [extra_args...] -- <shell> -c <command>
  ```

  `--remount-ro /` comes after the last bind. Without it, writes to `/` or `/etc`
  "succeed" into bwrap's throwaway tmpfs root.
- **Environment.** bwrap is started with exactly the sandbox's environment, so it passes
  only that environment through. This has the same effect as `--clearenv` plus
  `--setenv`, but values are not exposed in the world-readable argv. Each exec's
  environment is built in this order (later entries win):
  1. `env_passthrough` names copied from the Gym process (default `PATH LANG
     LD_LIBRARY_PATH CUDA_HOME`);
  2. `env`;
  3. defaults: `PATH` and `LANG=C.UTF-8` if still unset, and always `HOME=<home_path>`;
  4. `SandboxSpec.env`;
  5. the per-exec `env`;
  6. with `hide_gpus` (default), `CUDA_VISIBLE_DEVICES=""` and
     `NVIDIA_VISIBLE_DEVICES=void`.

  The fresh `/dev` has no NVIDIA device nodes either.
- **Processes do not outlive their exec.** When the command exits, bwrap's pid-1 exits
  and the kernel kills everything left in that pid namespace. On a timeout or a
  cancellation, the provider SIGKILLs the bwrap process group. `--die-with-parent`
  then tears down the namespace. A timeout returns `error_type="timeout"` with
  return code 125; a cancellation re-raises `CancelledError`. When bwrap itself
  fails before starting the command (it never reported a namespace init on
  `--info-fd`, a pipe the command cannot write), the exec returns
  `error_type="sandbox"` with return code 125. Whatever the command prints, a
  `bwrap: ` line included, never makes its own failure a sandbox error.
- **The workspace's host path.** The workspace is also bound at its own host path,
  so host paths a caller hands to sandboxed code resolve inside, unless that path
  lies under the sandbox's `/tmp`, home or workdir (the default `base_dir` is under
  the host's `/tmp`). bwrap would create that mount point in sandbox-writable space.
- **Stop and execs in flight.** An exec is *pending* from its start until its bwrap
  process is registered: queued on `exec_concurrency` or about to spawn. It records
  that in `run/` (`pending-*`, with its owner's pid and start time), so every provider
  instance sees it: `exec_processes_terminated()` is False while a live owner has one,
  and `close` waits (bounded by `kill_timeout_s`) for them before its final kill pass.
  A queued exec re-checks the closing marker after it gets its slot, so no exec starts
  after `close` returns. A dead owner's record is ignored and removed.
- **Own directories.** `/tmp`, `home_path` and the workdir (including a
  `SandboxSpec.workdir` override) must be disjoint: nested, a later bind would hide an
  earlier one from commands while transfers still resolve to it.
- **Process proof.** `exec_processes_terminated()` (`AsyncSandbox.exec_processes_terminated()`,
  protocol `SupportsSandboxExecScopedProcesses`) is True only when no exec of the
  sandbox is queued, starting or running, and no recorded bwrap process or
  namespace init of any provider instance is alive.
- **Files.** `upload_file`, `download_file` and `exists` are plain host-side file
  operations, so no process is started. Sandbox paths are mapped to host
  directories by longest prefix:
  - `workdir` and, when it is bound, the workspace's host path → `workspace/`;
  - `/tmp` → `tmp/`;
  - `home_path` → `home/`;
  - extra mounts → their `src`. Read-only mounts are download-only. A regular-file
    mount is addressed only at its exact destination.

  Trusted system and host-network bind aliases are resolved when their configuration
  is first used, keeping the original sandbox destination; later alias retargeting
  cannot select another source. Extra mount sources must not be symlink aliases.
  Bind sources are opened without following symlinks in any component and passed to
  bubblewrap through `--ro-bind-fd` or `--bind-fd`. A source or ancestor replaced with
  a symlink is refused before another command starts. An already-open source stays
  pinned to that filesystem object; bubblewrap verifies the mounted device and inode
  before starting guest code. Use real paths for extra mount
  sources, not symlink aliases. This requires bubblewrap's FD-bind options, available
  in the documented 0.11.2 runtime.
  The table follows bind order: a later mount at or above an earlier one's
  destination hides it, and transfers resolve to the mount a command sees.

  Relative paths are relative to the workdir. Paths with `..`, paths outside those
  prefixes, and paths whose components are symlinks are refused (`BwrapPathError`).
  Directories are opened one component at a time with `O_NOFOLLOW`, so code inside
  the sandbox cannot redirect a transfer through a symlink.
- **Stop.** `close` writes a `closing` marker, so new execs are refused. It then
  SIGKILLs every bwrap process recorded in `run/` by *any* provider instance. Each
  record holds the pid and its `/proc` start time, which guards against pid reuse.
  Finally it deletes the directory, restoring write permission on any directories
  that sandboxed code locked. `aclose` kills only this instance's live execs.

## Connect (borrowing)

`serialize()` returns:

```json
{"provider": "bwrap", "host": "<hostname>", "uid": 0, "base_dir": "/raid/nemo-gym-bwrap",
 "sandbox_id": "bwrap-<32 hex>", "workdir": "/workspace", "network": "host"}
```

`connect` refuses the descriptor in any of these cases:
- the provider name differs;
- the host or uid differs;
- the `base_dir` differs from the connecting provider's own;
- the id is malformed;
- the directory is missing or closing;
- the state file is missing or does not match the descriptor (`sandbox_id`, `network`).

The sandbox's workspace bind (`workdir` in the state file), `SandboxSpec.env` and
per-sandbox mounts come from the state file, not from the descriptor. The
descriptor's `workdir` is only the facade's current default cwd, which
`set_default_workdir` may have moved. Like apptainer's, the descriptor is a trusted control-plane
handle, not a lease: either side can stop the sandbox. A borrower should
`disconnect()`; the creator stops it.

## Config

| Key | Default | Meaning |
| --- | --- | --- |
| `bwrap_path` | required | Absolute path to an executable bwrap. |
| `base_dir` | `<tmp>/nemo-gym-bwrap-<uid>` | Parent of sandbox directories. It is created with mode 0700 and must be owned by the Gym uid. |
| `workdir` | `/workspace` | Workdir when `SandboxSpec.workdir` is unset. |
| `home_path` | `/home/sandbox` | Where `home/` is mounted; the value of `HOME`. |
| `network` | `none` | `none` = `--unshare-net`, loopback only. `host` = share the network and add `network_host_ro_binds`. |
| `shell` | `/bin/bash` | Runs `-c <command>`. |
| `ro_binds` | `/usr /bin /lib`, plus optional `/sbin /lib64 /opt /venv /etc/ld.so.cache /etc/ld.so.conf /etc/ld.so.conf.d /etc/alternatives` | System binds. Each entry is a string (required) or `{path, optional}`; globs are allowed. A missing required path fails provider construction. |
| `network_host_ro_binds` | `/etc/resolv.conf /etc/hosts /etc/nsswitch.conf /etc/ssl /etc/ca-certificates /etc/passwd /etc/group`, plus optional `/etc/pki` | Used only with `network: host`. A smaller set broke HTTPS on a Slurm + pyxis cluster (curl rc 77). |
| `mounts` | `[]` | `{src, dst, ro: true}` or `"src:dst[:ro\|rw]"`. `SandboxSpec.provider_options["mounts"]` adds per-sandbox mounts. A `dst` inside a read-only system bind is refused, because bwrap cannot create the mount point there. A `dst` at, above or inside the sandbox's `/tmp`, home or workdir is refused too: bwrap would create its mount point in sandbox-writable space, or the mount would shadow that directory. Use top-level targets such as `/codex-runtime`. |
| `env`, `env_passthrough`, `hide_gpus` | see above | Sandbox environment. |
| `remount_root_ro` | `true` | Adds `--remount-ro /`. |
| `default_files` | `{}` | `{sandbox path: text}` written into every new sandbox. |
| `extra_args` | `[]` | Extra bwrap flags, inserted before `--`. |
| `allow_nested_userns` | `false` | Let sandboxed processes create a nested user namespace, e.g. the Codex agent's `tool_pid_namespace`. A sandboxed root can map uid 0 into one only with CAP_SETFCAP (Linux >= 5.12): when the Gym process is root, this keeps that one capability (`--cap-add CAP_SETFCAP` after `--cap-drop ALL`). A non-root Gym needs nothing, and bwrap nested in a non-root sandbox refuses to start with capabilities, so the flag adds none there. |
| `exec_concurrency` | `256` | Concurrent bwrap processes per provider instance (`null` = unbounded). A long-running exec, such as a Codex episode, holds a slot for its whole duration, so size this above the number of concurrent episodes plus their control execs. |
| `default_timeout_s` | `180` | Used when an exec passes no timeout. |
| `kill_timeout_s` | `10` | Bound on reaping killed process trees. |
| `probe_on_create`, `probe_timeout_s` | `true`, `30` | Run `true` once in each new sandbox. |

These `SandboxSpec` fields are unsupported and raise an error: `entrypoint`, `resources.gpu`,
any `provider_options` key other than `mounts`, and `user` on `exec`. These are ignored
with a warning: `ttl_s`, cpu/memory/disk, `ports`, and `image`. The image is the Gym
process's own filesystem.

## Limits

- **No cross-exec signals.** Each exec has its own pid namespace, so a pid written by one
  exec means nothing to another, and `kill` from a second exec cannot reach the first.
  Gym's process supervisor does not depend on it: `stop_and_confirm_cleanup`
  (`nemo_gym/agents/utils/supervisor_client.py`) requests a stop by writing
  `stop.request`, which the supervisor polls, so an external cancellation (agent close
  before the harness finishes) still stops the harness and confirms its cleanup.
- **No background services** survive an exec (`SupportsSandboxBackgroundServices` is
  not implemented), and there are no endpoints or PTYs.
- **Isolation.** All sandboxes share the Gym process's uid and host. Isolation relies
  on the namespaces, the read-only binds and `--cap-drop ALL`. With `network: host`,
  sandboxed code can reach every service the Gym process can.

## Slurm / pyxis notes (measured on a GB300 cluster, bwrap 0.11.2, a CUDA task image)

- **A pinned binary.** The compute nodes and the task image had no bwrap. Copy a distribution build (>= 0.9)
  from a login node of the same architecture into its own read-only directory with its sha256 and provenance,
  mount that directory into the container, and point `bwrap_path` at the binary inside it.
- **Mount `/proc`.** The pyxis step needs `/proc:/proc` in `--container-mounts`.
  Otherwise NVIDIA's filtered proc mount makes bwrap fail with
  "Can't mount proc … Operation not permitted".
- **uid 0.** The pyxis container runs as uid 0. Descriptors therefore carry
  `"uid": 0`, and sandboxed files are owned by root on the host.
- **Measured costs and checks:**
  - `bwrap … true` takes 52 ms per exec.
  - 20 concurrent nvcc compiles took 0.89 s wall.
  - The GPU is hidden even in a GPU-holding step.
  - PID isolation holds: `pkill` and `kill -9 -1` inside the sandbox don't reach
    outside.
  - A unix socket bound into a net-unshared sandbox works. It is a possible future
    channel for a submit-only `network: none` mode.
