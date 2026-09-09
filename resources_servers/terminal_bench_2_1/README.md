# Description

Data links: ?

## Dataset row fields

Each JSONL row carries the task coordinates top-level (schema: `task_data.py`):

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `task_name` | `str` | yes | Terminal-Bench 2.1 task id, e.g. `terminal-bench/path-tracing`; keys sandbox metadata and is echoed in the verify response. |
| `docker_image` | `str` | yes | Image the task sandbox is started from (`/seed_session`). |
| `task_folder` | `str` | yes | Repo-relative task directory; `verify()` uploads its `tests/` (and `solution/` in golden-patch mode) into the sandbox. |
| `agent_user` | `str \| int \| null` | no (default `null`) | Identity the agent harness runs as inside the sandbox; see below. |

## Agent identity (`agent_user`)

`agent_user` mirrors Harbor's `task.toml` `[agent] user`: a `str` is an account name (the sandbox
provider runs the agent's commands through `su`), an `int` is a uid. Digit-only strings such as
`"1000"` are normalized to the int `1000` on the way in. `null` keeps the image default.

- `/seed_session` echoes the row's `agent_user` in its response. The agent harness reads it from there,
  so the row override is honored only when the resources server echoes it (`terminal_bench_2_1` does);
  an agent lane's own `agent_user` default applies with every resources server. Precedence: row value
  when present, otherwise the lane default. A row value of `"root"` (or `0`) is the explicit per-row
  image default: it beats a lane default such as `agent` and skips the agent's identity check. For mixed
  pools, leave the lane default at `null` and set `agent_user` per row.
- Today only `terminus_2_sandboxed_agent` honors `agent_user`; other allowed agents (for example
  `opencode_sandboxed_agent`) run as the image default regardless of the row.
- The verifier (`mkdir -p /tests`, the `tests/` upload and `bash /tests/test.sh`) always runs as the
  image default (root on the supported images), regardless of `agent_user`. A non-root `agent_user`
  therefore requires a root-default image on which that account exists (the `<task_hash>-userroot`
  derivatives of the `USER 1000:1000` task images).
- `/verify` records the identity that was in force as `agent_user` in its response.
- Golden-patch mode (`is_verifying_golden_patch=true`): the uploaded `solution/` files are first
  `chown`ed to a non-root `agent_user` (as image default), then `bash <cwd>/solve.sh` runs with
  `user=agent_user`, since the reference solution stands in for the agent. With `agent_user` omitted
  the behavior is unchanged (no chown, `solve.sh` as image default).
- Residual risk under the no-teardown ruling: agent processes (for example the tmux session, uid 1000)
  stay alive during root verification and can read `/tests` (0755) and modify the workspace, but cannot
  write `/tests`, `/logs/verifier` or system paths. Containment is only as strong as the image's
  sudoers/setuid configuration (the `-userroot` derivatives are probed for `sudo -n` in the agent's
  opt-in integration test).

# Quickstart
## Apply golden patches
### Start resources server
```bash
gym env start \
    --config resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    +terminal_bench_2_1_resources_server.resources_servers.terminal_bench_2_1.debug=true \
    +terminal_bench_2_1_resources_server.resources_servers.terminal_bench_2_1.is_verifying_golden_patch=true
```

### Full Terminal Bench 2.1 golden patch smoke test
In a separate terminal:
```bash
python resources_servers/terminal_bench_2_1/apply_golden_patch.py \
    +benchmark_jsonl=benchmarks/terminal_bench_2_1/data/benchmark.jsonl \
    +limit=...  # No limit for full samples
```

Expected golden patch resolve rates:
```bash

```


# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?
