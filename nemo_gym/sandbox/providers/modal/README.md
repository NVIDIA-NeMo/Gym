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
| `ready_timeout_s` | Overall deadline for the configured readiness probe. |
| `workdir` | Sandbox working directory and the NeMo Gym facade's default command directory. |
| `env` | Environment variables injected when the sandbox is created. |
| `files` | Uploaded by the NeMo Gym facade after readiness and before `start()` returns. |
| `metadata` | Merged into Modal sandbox tags. Modal permits at most ten combined tags. |
| `resources.cpu` | Modal CPU request in physical cores. |
| `resources.memory_mib` | Modal memory request in MiB. |
| `resources.gpu` / `gpu_type` | Converted to a Modal GPU string such as `H100:2`. |
| `resources.disk_gib` | Unsupported by the Modal Sandbox create API; warns, or raises with `create.strict_resources`. |
| `ports` | Exposed with Modal TLS termination by default; resolved through `sandbox.endpoint(port)`. |

CPU and memory values are Modal resource requests rather than hard usage caps. Configure limits
outside this provider if the workload needs a stricter ceiling.

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

Creation is submitted once. The provider does not retry an ambiguous create failure because the
first request may already have allocated a billable sandbox. The default readiness command is
`printf ok`; it can validate exact stdout and multiple consecutive successes. Set
`probe.command: null` to disable this provider-level check.

`ready_timeout_s` bounds the entire readiness operation, including a stalled command stream.
If readiness fails or the caller cancels after allocation, the provider terminates the sandbox
before propagating the failure. Confirmed termination may take longer than the readiness
deadline. The remote TTL is the final cleanup backstop if the client process itself is killed.

Command execution is also submitted once. Modal 1.5.5 returns `-1` from `wait()` when its exec
deadline expires; the provider converts that sentinel, or a direct `ExecTimeoutError`, to
`TimeoutError`. Signal exits remain ordinary process results (`128 + signal`). Output is read as
bytes and decoded with UTF-8 replacement, so non-UTF-8 task output does not crash the agent.

Modal does not expose a public operation to kill only an individual exec process. If the caller
cancels an exec, the provider closes the sandbox so a late process cannot mutate a later
evaluation step. The same rule applies to a cancelled or timed-out file transfer.

`close()` retries only recognized transient control-plane failures, waits for remote
termination, then detaches the local SDK connection. It keeps the handle intact when cleanup
fails so callers can retry. Creation and command execution are never retried.

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
