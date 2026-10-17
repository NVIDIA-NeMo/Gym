# Modal Sandbox Provider

The `modal` provider runs NeMo Gym sandboxes through the Modal Python SDK. It creates each
sandbox directly from an OCI image and implements command execution, streaming file transfer,
lifecycle status, HTTPS tunnels, and cross-process reconnection through the provider-neutral
NeMo Gym sandbox API.

## Setup

Install NeMo Gym's sandbox dependencies and authenticate the Modal SDK:

```bash
uv sync --extra sandbox
modal token new
```

For a package install, use `pip install "nemo-gym[sandbox]"`. The provider requires
`modal>=1.5.5,<2.0.0`. Existing `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` environment variables
also work; credentials are read by the SDK and do not belong in the provider config.

The shipped config is `nemo_gym/sandbox/providers/modal/configs/modal.yaml`. Add it beside the
agent and model configs:

```bash
gym env start \
  --config responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml \
  --config nemo_gym/sandbox/providers/modal/configs/modal.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml
```

Set `MODAL_ENVIRONMENT` when the target Modal environment must be explicit. Otherwise the SDK
uses the active profile or workspace default. `NEMO_GYM_MODAL_APP` selects the Modal App that
owns the sandboxes and defaults to `nemo-gym-sandboxes`.

## `SandboxSpec` Mapping

| Field | Modal behavior |
| --- | --- |
| `image` | Required OCI image reference passed to `modal.Image.from_registry`. Pin a digest for reproducible runs. |
| `entrypoint` | Overrides the image command. Without one, the provider starts a POSIX-shell keepalive process. |
| `ttl_s` | Hard sandbox lifetime; must be positive and at most 24 hours. |
| `ready_timeout_s` | Budget for App lookup, allocation, and readiness; cleanup has its own bound. |
| `workdir` | Sandbox working directory and the NeMo Gym facade's default command directory. |
| `env` | Environment variables injected when the sandbox is created. |
| `files` | Uploaded by the NeMo Gym facade after readiness and before `start()` returns. |
| `metadata` | Merged into Modal sandbox tags. Modal permits at most ten combined tags. |
| `resources.cpu` | Modal CPU request in physical cores. |
| `resources.memory_mib` | Modal memory request in MiB. |
| `resources.gpu` / `gpu_type` | Converted to a Modal GPU string such as `H100:2`. |
| `resources.disk_gib` | Unsupported by the Modal Sandbox create API; warns, or raises with `create.strict_resources`. |
| `ports` | Exposed with Modal TLS termination by default; resolved through `sandbox.endpoint(port)`. |

CPU and memory values are scalar Modal resource requests, not hard usage caps. Modal supports
request/limit pairs, but this provider does not yet expose them; workloads requiring enforced
per-sandbox CPU or memory limits need that capability added before using this provider.

When `SandboxSpec.entrypoint` is omitted, the provider clears the registry image's ENTRYPOINT
before supplying its POSIX keepalive command. An explicit `SandboxSpec.entrypoint` retains the
image ENTRYPOINT and replaces its CMD, following Modal's command semantics.

## Provider Options

Set backend-specific values under `SandboxSpec.provider_options`:

| Option | Purpose |
| --- | --- |
| `gpu` | Explicit Modal GPU string, overriding neutral GPU fields. |
| `cloud` / `region` | Per-sandbox placement overrides. |
| `secrets` | Modal Secret names injected into the sandbox. |
| `volumes` | Mapping of sandbox mount paths to Modal Volume names. |
| `block_network` | Blocks outbound network access. Modal does not combine this with exposed ports. |
| `idle_timeout_s` | Terminates a sandbox after Modal considers it idle. |
| `image_secret` | Modal Secret containing private-registry credentials. |
| `name` | Modal sandbox name, unique within the App. |
| `tags` | Additional tags merged after default tags and neutral metadata. |

Unknown options and invalid combinations are rejected before sandbox allocation.

## Readiness, Timeouts, and Cleanup

The provider does not add retries to create or exec requests: a new request could allocate a
second sandbox or repeat a side effect. The Modal SDK may retry transient create failures
internally using the same request idempotency key. Provider retries are limited to recognized
transient failures on idempotent control-plane operations such as reconnecting, resolving
tunnels, and cleanup. `status()` polls once and returns unknown on transient failures.

`ready_timeout_s` covers App lookup, image preparation/pull, scheduling, and the readiness probe.
It overrides `probe.deadline_s` (180 seconds by default). `probe.command: null` disables only
the exec probe. A failed probe reports the last completed attempt's stdout, stderr, and return
code. Invalid exec configuration and sandbox errors confirmed stopped by a status poll fail
immediately. Running or unknown sandboxes may retry transient probe failures. With
`probe.deadline_s: null`, the first failed probe raises rather than polling indefinitely.

Failed or cancelled creation attempts cleanup before returning. Cleanup can extend the readiness
deadline: each close attempt is bounded by `operations.close_timeout_s` (60 seconds by default).
When allocation is still in flight, the provider allows that same cleanup budget to obtain its
id and terminate it. If no id arrives in time, it cancels the local request and logs the App name;
a remote allocation may still exist and its TTL is the final cleanup backstop. The TTL also
protects against the client process being killed; it does not replace normal `stop()` calls.

SDK timeout exceptions and the SDK's `-1` exit sentinel return
`SandboxExecResult(error_type="timeout", return_code=125)`. Worker-side deadlines can instead
surface as signal exits (for example, 137). The provider classifies signal-range exit codes
129–192 as timeouts when dispatch-to-wait-completion time meets the timeout passed to Modal
(rounded up to whole seconds). This is a heuristic: a coincident explicit exit, OOM, unrelated
signal, or delayed dispatch can be indistinguishable from a deadline. Live validation of this
worker-timeout path is still pending. Early signals, signals without a deadline, and ordinary
nonzero exits retain their process result. Slow output draining does not affect classification.

A sandbox that disappears, rejects exec, or loses its client/transport returns
`error_type="sandbox"` with code 125, without replaying the command. Both error types retain
captured stdout and stderr, decoded with UTF-8 replacement. Exec sends EOF and drains stdin
concurrently with output and process waiting, so stdin readers can finish. Expected stdin-close
errors after confirmed process exit do not replace the exit result.

Modal limits the combined exec argv to 65,536 characters, including the shell wrapper and any
user rewrite. An oversized command returns a sandbox error before dispatch; upload a script
and execute its path for large heredocs or generated programs.

Modal has no public per-exec kill API. Client cancellation (including `asyncio.timeout`) and
interrupted file transfers attempt termination only for handles created by this provider.
Handles obtained with `connect()` are borrowed: interruption logs a warning and preserves the
sandbox so its owner can collect diagnostics and run verification before stopping it. A command
on a borrowed handle may continue until its exec timeout; file writes may also remain in flight.
Use an explicit exec timeout and let the owner coordinate cleanup before reusing the sandbox.
On an owned handle, cancellation can remove logs; use exec's `timeout_s` to retain the sandbox
and its files after a command deadline. Cleanup failures preserve the original interruption.


File operations use Modal's streaming filesystem API. Reads have the SDK's 5 GB limit. Missing
remote files become `FileNotFoundError`; other SDK filesystem errors are preserved.

`close()` waits for remote termination and detaches the SDK connection within its cleanup
budget. The handle is retained if cleanup fails so it can be retried. Serialized handles
reconnect through `modal.Sandbox.from_id`.

## Files, Ports, and Reconnection

Uploads and downloads use Modal's streaming filesystem APIs. Uploads create missing parent
directories; downloads use the SDK's atomic local replacement behavior. A missing remote path
becomes `FileNotFoundError`, while permission and other filesystem errors retain their SDK
types.

The default `create.port_mode: encrypted` places Modal TLS termination in front of a plaintext
service inside the sandbox and returns an HTTPS URL. `h2` enables Modal's HTTP/2 tunnel.
`unencrypted` returns the public raw TCP socket represented as an `http://` endpoint and should
only be used when that protocol is appropriate.

The provider implements NeMo Gym's `ConnectableProvider` capability. Serialized handles carry
the Modal sandbox ID, declared ports, image, and port mode; another configured provider instance
can reconnect through `modal.Sandbox.from_id`.

## Security

Modal supplies the sandbox isolation boundary. Outbound internet access is enabled by default
for compatibility with tasks that install dependencies. Set `create.block_network: true` or a
per-sandbox override for offline workloads without exposed ports.

Only pass sandbox Secrets and Volumes that the evaluated workload needs. Provider credentials
stay in the host SDK configuration. `exec.allow_user_rewrite` is disabled by default; enabling
it wraps commands with `su` and therefore requires a named account and a compatible image.

Commands with `user=None`, `user="root"`, or `user=0` use the container's default root user and
need no `su` binary. Non-root users require `exec.allow_user_rewrite: true`, a named account, and
`su` plus `/bin/sh` in the image; non-root numeric UIDs are unsupported. The full configured
`exec.shell` argv is preserved, including login-shell flags such as `["/bin/bash", "-lc"]`.

Registry images must target `linux/amd64`. `image_secret` supports static
`REGISTRY_USERNAME` / `REGISTRY_PASSWORD` credentials. Private Amazon ECR and Google Artifact
Registry authentication require Modal's dedicated image constructors, which this provider's
`Image.from_registry` path does not expose; mirror those images to a registry with supported
static credentials before using them here.
