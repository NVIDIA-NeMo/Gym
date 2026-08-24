# IdeGYM Sandbox Provider — Design Decisions

A record of the questions this integration raised, what was chosen, and why. It exists so a reviewer
can see which choices were deliberate and which are open, and so the next person to touch this
provider does not have to re-derive the constraints from the IdeGYM and NeMo Gym sources.

Written against **NeMo Gym** at `2251ef7` and **idegym 0.11.1** (`idegym-client` 0.11.1 on PyPI).

## Questions put to the maintainer

Three decisions were genuinely the maintainer's to make, because different answers would have meant
materially different work. They were asked before implementation.

### 1. How should `SandboxSpec.image` be resolved?

The constraint: an IdeGYM sandbox is a pod running the **IdeGYM server**, and that server is what
exposes the API the orchestrator forwards commands to. A plain benchmark image does not boot one, so
the image cannot simply be a SWE-bench tag.

| Option | Decision |
| --- | --- |
| **Pass pre-built IdeGYM images through; document `image_rewrites` for mapping upstream benchmark images onto them** | **Chosen** |
| Integrate IdeGYM's image builder so the provider can build and push a wrapped image on demand | Rejected for a first milestone: needs a registry, cluster build permissions, and a build-cache story |
| Validate the image against an allow-prefix and refuse anything else | Rejected as too rigid — registry layouts vary |

Consequence: `spec.image` is normalized (`docker://` stripped) and validated against IdeGYM's OCI
reference pattern, and nothing more. The failure mode for a non-IdeGYM image is a readiness-probe
failure, which is why `create()` probes at all instead of trusting "pod is running".

### 2. What should `close()` do?

IdeGYM distinguishes `stop_server` (delete the pod and its Kubernetes resources) from
`finish_server` (mark the server free, keep the pod warm for reuse).

| Option | Decision |
| --- | --- |
| `stop_server` only | **Chosen** |
| `stop` by default with `finish` + reuse available behind a config knob | Not implemented in this milestone |
| `finish` by default | Rejected: keeps pods warm but leaks them whenever a run dies |

Consequence: `close()` always deletes. Reuse is still reachable — pinning
`provider_options.server_name` and setting `reuse_strategy` opts into IdeGYM's own server matching —
but the provider does not manage a warm pool, and `operations` has no `close_action` knob to get
wrong. Pod startup cost is therefore paid per task; if that becomes the bottleneck on long
benchmarks, a `finish`-based warm pool is the follow-up.

### 3. Where does the SDK dependency live?

| Option | Decision |
| --- | --- |
| Opt-in extra `nemo-gym[idegym]`, imported lazily, with an actionable `ModuleNotFoundError` | **Chosen** |
| Core dependency alongside `opensandbox` / `daytona` / `boto3` | Rejected: IdeGYM is a third-party control plane, and its SDK should not be in installs that will never talk to one |

This follows the `nemo-gym[openshell]` precedent. CI installs the extra alongside the other sandbox
extras so the SDK-gated tests run rather than skip.

**One consequence worth a maintainer's attention.** `uv.lock` resolves all extras together, so adding
this one raises two locked versions for every install, not just for IdeGYM users:

| Package | Before | After | Cause |
| --- | --- | --- | --- |
| `hydra-core` | 1.3.2 | 1.3.5 | `idegym-common-utils` requires `hydra-core>=1.3.4` |
| `pyyaml` | 6.0.2 | 6.0.3 | pulled in by that resolution |

Both are patch bumps inside the line `pyproject.toml` already allows (`hydra-core` is unpinned), and
the full core unit-test suite passes on them. If keeping the resolution untouched matters more, the
alternative is declaring the extra conflicting in `[tool.uv]` so it resolves separately — at the cost
of a second resolution to maintain.

## Decisions taken without asking

These follow from reading the two codebases; they are recorded because each has a plausible
alternative that a reviewer might expect instead.

### One registered IdeGYM client per process, not per sandbox

IdeGYM's unit of ownership is a **client**: a registered, heartbeating entity that owns N server
pods, carries the resource quota (matched by regex against its *name*), and terminates all of its
servers when it stops.

Registering one per sandbox looked simpler but does not scale to a benchmark fan-out: the
orchestrator takes `LOCK TABLE clients IN EXCLUSIVE MODE` per registration, heartbeats multiply by
the concurrency, and one job's pods end up scattered across many owners in the dashboard. So a
session is shared by every sandbox with an identical `connection` config and reference-counted across
providers.

The refcount matters for correctness, not just efficiency: the SDK unregisters a client by stopping
it, which **terminates all of its servers**. Releasing early would kill live sandboxes belonging to
other providers.

### The shared client runs on a private event loop

The SDK's client owns an httpx session and a heartbeat task, so it belongs to the loop that created
it. NeMo Gym drives sandboxes from two directions: `AsyncSandbox` on the caller's loop, and the sync
`Sandbox` facade — which gives *every sandbox its own* loop in its own thread, and is what
`mini_swe_agent_2` uses. Binding the shared client to whichever loop happened to create it first
would tie one registration's lifetime to one sandbox's.

Alternative considered: key sessions on `(connection, running loop)`. That is simpler, and correct
for the async case, but degenerates to one registration per sandbox under the sync facade — exactly
the case that motivated sharing.

### Tracing is off unless configured

`IdeGYMClient` ships a default OTLP endpoint, so constructing it with no `otel_config` sends traces
off-box to a third party. The session always passes an explicit `OTELConfig`, and
`connection.tracing_endpoint` is `null` by default. Opting in is a config change.

### The SDK's httpx transport is replaced

`AGENTS.md` requires async HTTP to go through aiohttp — httpx's connection pool is O(n²) at high
concurrency — and says to swap the transport when wrapping a library that uses httpx internally. The
IdeGYM client builds its own `httpx.AsyncClient` and exposes no hook, so the swap reaches into
`client._http_client._transport`. That private access is isolated in one function, degrades to a
warning plus the SDK's own transport if the shape changes, and is guarded by a test. Pool limits and
`connection.max_connections` bounds orchestrator load either way.

### Files move as base64 over the bash tool

IdeGYM's server does have a filesystem API, but not one reachable here: its read endpoint streams raw
bytes while the orchestrator forwards requests by storing and replaying them as JSON *text*, so
binary content does not survive; and its typed file endpoints only write UTF-8. base64 over the bash
tool is binary-safe in both directions and needs nothing but coreutils.

Both directions are chunked, for different reasons, and the numbers are not arbitrary:

- **Uploads** embed the payload in the script, and the IdeGYM executor passes the script as a single
  `execve()` argument. Linux caps that at `MAX_ARG_STRLEN` (128 KiB) and fails with `E2BIG`, so a raw
  chunk has to stay under ~96 KiB once base64 inflates it by 4/3. The default is 48 KiB, and the
  config rejects anything that could not fit.
- **Downloads** come back as a command's stdout, which the orchestrator persists as the text result
  of an async operation, so a whole-file read would put an inflated copy of the file through its
  database.

The download chunk script deliberately does *not* set `pipefail`: `head -c` closes the pipe as soon
as it has its bytes, so `tail` normally dies of SIGPIPE and would make a perfectly good chunk look
like a failure. The decoded-length check is the real guard.

### Command context is expressed in the script

IdeGYM's bash tool takes a command, a timeout and a graceful-termination timeout — no
working-directory, environment or user arguments. So `cwd` becomes `cd -- ... || exit 1`, `env`
becomes `export`, and the whole thing is wrapped in one `{ ... }` group because the executor runs
`source <bash-integration> && <script>` and `&&` binds only to the first statement of a
multi-statement script.

`user` has no equivalent at all. Rather than silently dropping it, `exec.user_mode` makes the
behavior explicit: `ignore` (default) warns once per provider, while `runuser` and `su` wrap the
script for images that ship those tools. `create.run_as_root` defaults to `true` in the shipped
config, which makes `exec(user="root")` — what `mini_swe_agent_2` passes — a no-op rather than a lie.

### `create()` verifies before returning

Two checks, both because their absence produces a confusing failure much later:

- The readiness probe, because a scheduled pod is not yet a sandbox that can run commands.
- `spec.workdir` exists, because otherwise every later command fails on `cd` and reads like a broken
  agent rather than a mis-set path.

### Capabilities stands in for a status endpoint

IdeGYM has no per-server status endpoint. `list_capabilities` is the cheapest call that validates the
server's database record *and* reaches the pod, which is exactly the liveness question `status()`
asks. Its answers map cleanly: 404 (unknown client/server) and 410 (terminal state) mean the sandbox
is gone; anything else that fails leaves the status `unknown`.

### HTTP status is recovered by parsing exception text

The SDK collapses every HTTP failure into a plain `RuntimeError` whose message embeds the status and
body. Telling "the sandbox is gone" from "the control plane is busy" from "the command timed out"
requires that status, so the parsing lives in `errors.py` — one module, both message shapes, covered
by tests — rather than being spread across the provider. If the SDK ever raises typed errors, that is
the only place to change.

### Transport failures are classified by httpx's hierarchy, not Python's

The SDK lets httpx's exceptions through unwrapped, and none of them are builtins: `httpx.ConnectError`
is *not* a `ConnectionError` and `httpx.ReadTimeout` is *not* a `TimeoutError`. Classifying on the
builtins alone would leave the most common transient failure — the orchestrator being briefly
unreachable — unretried, so `is_retryable` checks `httpx.TransportError` explicitly. `ConnectTimeout`
is excluded from the *command*-timeout check: failing to reach the orchestrator is a connectivity
problem, not a slow command.

### `ready_timeout_s` is a deadline for the whole create, not per attempt

`_start_server` tracks a deadline across retries and hands each attempt only the remaining budget,
because the config documents `ready_timeout_s` as bounding the whole call. Per-attempt budgets would
let a *late* retryable failure — a 503 after 1100s of a 1200s wait — start a fresh full-length
attempt, up to 80 minutes on the shipped default. A bare `TimeoutError` is additionally never
retried: the SDK raises it only after spending the budget it was given.

### Concurrency is bounded by the HTTP pool, not by serializing operations

An earlier draft held a semaphore for the duration of each session operation. That bounds *operations*
rather than requests, and an IdeGYM create polls for as long as the readiness timeout — so with the
shipped 32-slot pool and `mini_swe_agent_2`'s concurrency of 64, provisioning would have blocked the
exec calls of sandboxes that were already running, and teardown behind provisioning. The limit now
lives where it belongs, on `connection.max_connections`, which both transports honor (the aiohttp
bridge maps it onto the connector's `limit`).

### Failure paths get the same fidelity as success paths

Three places where the SDK's or the shell's failure behavior is not the obvious one, each covered by a
test:

- `stop_server` reports a failed delete by *returning* an `ErrorResponse` rather than raising, so the
  return value is checked; an unchecked one records a live pod as stopped.
- A server is dropped from the session's bookkeeping only once it is really stopped. Dropping it on
  failure would make the caller's own retry look like "already stopped" and leave the pod running.
- The generated script opens with `:` so the brace group is never empty: a blank or comment-only
  command would otherwise fail with a bash syntax error attributed to the caller.

### Failures are classified on structure, not on substring

An orchestrator error relays the sandbox's own output in its `body`, so both the status and the
timeout patterns are anchored to the surrounding message. Searching the whole string would let tool
output like `curl failed: status=404` mark a healthy sandbox as gone, or `pytest ... timed out after
30s` read as a command timeout.

### Server names are not made unique

The provider sends `<prefix>-<metadata hints>` and stops there. IdeGYM appends its own autoincrement
id to derive `generated_name`, and that is the name Kubernetes sees and the only one the database
keeps unique (`server_name` is an unconstrained column). Adding a random suffix on top would buy no
uniqueness, cost 9 characters of the 63-character budget that instance ids need, and make pod names
harder to read.

It would also mislead: a suffix looks like it prevents unintended reuse, but reuse is skipped
entirely unless `reuse_strategy` is `RESTART`/`RESET`, and even then only servers left `FINISHED`
qualify — while this provider always *stops* the servers it creates, which marks them `STOPPED`.

The same reasoning applies to retries, which keep the name rather than re-rolling it: an attempt whose
response was lost may have landed anyway, but that orphan gets its own `generated_name` and is reaped
by the watcher once the client stops heartbeating. Renaming would only break the name matching that a
pinned `provider_options.server_name` exists to drive.

### Metadata cannot become labels

On OpenSandbox, `SandboxSpec.metadata` becomes queryable Kubernetes labels. IdeGYM has no per-server
label API — `pod_overrides` is a partial `V1PodSpec`, not object metadata — so metadata is used for
naming instead: `create.server_name_metadata_keys` folds selected keys (default `instance_id`) into
the pod name, and job attribution names the registered client, which is what the dashboard groups by
and what quota rules match. The run id is deliberately excluded from the client name: it changes per
launch, and a name that changes per launch defeats both quota matching and dashboard grouping.

## Capabilities deliberately not claimed

| Capability | Why not |
| --- | --- |
| `SupportsSandboxEndpoint` (`endpoint()`) | The orchestrator forwards API requests through an async-operation indirection; it does not route raw TCP to arbitrary container ports, so there is no honest URL to hand back. |
| `SupportsSandboxPty` | IdeGYM has no terminal API. |
| `ConnectableProvider` (`serialize()` / `connect()`) | A server is only reachable through the registered client that owns it. Attaching from another process would mean a client with no registration and no heartbeat, and tearing one down cleanly is not expressible through the SDK's public surface without also stopping the client — which would kill the original process's sandboxes. Sharing a sandbox across processes is possible by fronting this provider with a sandbox server. |

## Verified and not verified

Verified locally:

- The generated bash scripts — quoting, `cd`, `export`, base64 chunking at unaligned boundaries,
  empty files, filenames with dashes and quotes — by executing them with real `bash`.
- Every session call shape binds against the published `idegym-client` 0.11.1 signatures, and the
  plain-dict start-server request validates into the SDK's own pydantic models.
- The documented launch command resolves end-to-end:
  `gym env resolve --config <mini_swe_agent_2> --config <idegym> --config <vllm_model>`.
- The full `-m sandbox` suite and the core unit tests, with no new failures.
  (`test_enroot_provider.py::test_create_start_timeout_cleans_up` fails on a clean checkout too.)

**Not** verified: nothing has been run against a real IdeGYM orchestrator or Kubernetes cluster. The
end-to-end acceptance criterion — `mini_swe_agent_2` completing a small evaluation on IdeGYM — needs a
reachable orchestrator and IdeGYM-wrapped benchmark images, and is the next step rather than a
finished claim.

## One fix outside the provider

`mini_swe_agent_2.yaml` declared `datasets: []` twice, which the strict YAML loader rejects, so
`gym env resolve` and `gym env start` failed for *every* sandbox provider — the launch command
documented across all the provider configs. The duplicate is removed here because the provider's own
quick-start depends on it.
