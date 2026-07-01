# Modal sandbox provider

The `modal` provider backs NeMo Gym's provider-neutral `AsyncSandbox` API with
[Modal Sandboxes](https://modal.com/docs/guide/sandbox). It creates a sandbox from a registry image, executes
commands, transfers individual files through `Sandbox.filesystem`, polls status, and terminates the sandbox.

## Install and authenticate

Install the sandbox extra and configure Modal through its normal profile or environment variables:

```bash
uv sync --extra sandbox
modal token set --token-id ... --token-secret ...
```

Do not put Modal tokens in Gym YAML. The provider deliberately has no token fields. `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` are also supported by the Modal SDK.

Include the shipped provider block when starting Gym:

```bash
ng_run "+config_paths=[responses_api_agents/my_agent/configs/my_agent.yaml,nemo_gym/sandbox/providers/modal/configs/modal.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

The agent refers to that block with `sandbox_provider: sandbox`.

## Provider configuration

The complete provider block, including all defaults, is:

```yaml
sandbox:
  default_metadata:
    sandbox-api: modal-sdk
  modal:
    connection:
      app_name: nemo-gym-sandbox
      environment_name: null
      create_if_missing: true
    create:
      default_timeout_s: 21600
      ready_timeout_s: 1200
      cleanup_timeout_s: 30
      idle_timeout_s: null
      exec_stdout_limit_bytes: 16777216
      exec_stderr_limit_bytes: 16777216
      exec_combined_output_limit_bytes: 33554432
    probe:
      command: printf modal-sandbox-ready
      expected_stdout: modal-sandbox-ready
      timeout_s: 60
```

### `connection`

| Field | Meaning |
|---|---|
| `app_name` | Modal App used to own the sandboxes. |
| `environment_name` | Optional Modal Environment for App and named Secret lookup. |
| `create_if_missing` | Let `App.lookup` create the App when it does not already exist. |

Authentication is always resolved by the Modal SDK, independently of the selected `environment_name`.

### `create`

| Field | Meaning |
|---|---|
| `default_timeout_s` | Sandbox lifetime used when `SandboxSpec.ttl_s` is omitted. |
| `ready_timeout_s` | Client-side creation deadline used when `SandboxSpec.ready_timeout_s` is omitted. |
| `cleanup_timeout_s` | Maximum time a caller waits for one tracked termination or for `aclose()` to drain cleanup. |
| `idle_timeout_s` | Optional default Modal idle timeout. A `provider_options.idle_timeout_s` value takes precedence. |
| `exec_stdout_limit_bytes` | Maximum raw bytes accepted from stdout for one `exec` call. |
| `exec_stderr_limit_bytes` | Maximum raw bytes accepted from stderr for one `exec` call. |
| `exec_combined_output_limit_bytes` | Maximum combined raw bytes accepted from stdout and stderr for one `exec` call. |

Timeouts and output limits must be positive. Output limits must be integers; boolean values are rejected.

### `probe`

The provider runs the probe after Modal returns a sandbox and before `create()` succeeds. `expected_stdout` is a
substring check. Set `expected_stdout: null` to check only the process return code, or set `command: null` to
disable the probe entirely. `timeout_s` must remain set to a positive value even when the probe is disabled.

## Sandbox spec

```python
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec

spec = SandboxSpec(
    image="public.ecr.aws/example/deep-swe-task:latest",
    ttl_s=7200,
    ready_timeout_s=1200,
    workdir="/repo",
    resources=SandboxResources(cpu=4, memory_mib=16384),
    entrypoint=["sleep", "infinity"],
    metadata={"benchmark": "deep-swe", "task": "example-task"},
    provider_options={
        "secret_names": ["deep-swe-model-api"],
        "network_allowlist": ["api.example.com"],
        "image_setup_steps": [
            {
                "run": "npm install -g @anthropic-ai/claude-code@2.1.153",
                "user": "root",
            }
        ],
    },
)

async with AsyncSandbox({"modal": {}}, spec) as sandbox:
    await sandbox.start()
    result = await sandbox.exec("git status --short", timeout_s=60)
    await sandbox.download("/repo/result.json", "result.json", max_bytes=64 * 1024 * 1024)
```

`entrypoint` controls the sandbox's long-running process. Set it explicitly when the registry image's default
command exits immediately.

## Provider options

Unknown options fail validation instead of being ignored.

| Option | Meaning |
|---|---|
| `name` | Modal sandbox name within the configured app. |
| `secret_names` | Modal Secret names injected into the runtime. Secret values never pass through Gym config. |
| `registry_secret_name` | Modal Secret containing private registry credentials for `Image.from_registry`. |
| `block_network` | Block runtime network access completely. Cannot be combined with an inbound or outbound allowlist. |
| `network_allowlist` | Convenience alias for `outbound_domain_allowlist`. Entries are domain names, including Modal-supported wildcard prefixes. |
| `outbound_domain_allowlist` | Domains the runtime may reach. Mutually exclusive with `network_allowlist`. |
| `outbound_cidr_allowlist` | CIDRs the runtime may reach. |
| `inbound_cidr_allowlist` | CIDRs allowed to connect to exposed sandbox endpoints. |
| `cloud` | Modal cloud placement selector. |
| `region` | One region string or a list of acceptable regions. |
| `idle_timeout_s` | Per-sandbox idle timeout override. |
| `cpu_limit` | Optional CPU ceiling. Combined with `resources.cpu` as Modal's `(request, limit)` tuple. |
| `memory_limit_mib` | Optional memory ceiling. Combined with `resources.memory_mib` as `(request, limit)`. |
| `image_setup_steps` | Cached image-build steps, described below. |

### Image setup before runtime isolation

`image_setup_steps` extends the registry image using Modal's cached `Image.run_commands` layers before the
sandbox starts. This is useful for installing an agent while build-time egress is available, then applying a
runtime domain allowlist or `block_network`.

Each step accepts:

- `run`: a command string or list of command strings;
- `user`: `root`, uid `0`, or a username (implemented with `su` inside the build container);
- `shell`: absolute shell path, default `/bin/sh`; use `/bin/bash` for commands that require `pipefail`;
- `env`: non-secret build environment values;
- `secret_names`: Modal Secret names injected only into that build step.

The provider never logs setup commands, environment values, or secret names during creation failures. Prefer
`secret_names` over literal credentials in `env`.

## Bounded command output and cleanup

The provider requests binary output from Modal and consumes stdout and stderr incrementally instead of calling
`read()`. Limits count raw bytes and apply independently to each stream and to their combined output. Bounded tails
are decoded as UTF-8 only after collection; invalid bytes and partial code points at a tail boundary are omitted.
This guarantees that the UTF-8 encoding of each returned string remains within its configured byte limit. Valid
UTF-8 split across provider chunks is preserved.

When a limit is exceeded, `exec()`:

- cancels all local stream and wait tasks;
- starts terminating the enclosing Modal sandbox immediately, because Modal does not expose a per-command
  termination method;
- drains cancelled tasks and closes stream iterators concurrently with termination, bounded by
  `cleanup_timeout_s`;
- returns bounded, UTF-8-safe stdout and stderr tails with `return_code: -1`; and
- sets `error_type` to `output_limit_stdout`, `output_limit_stderr`, or `output_limit_combined`.

If the bounded termination or stream-drain wait expires, `_cleanup_timeout` is appended to `error_type`. If either
cleanup task fails, `_cleanup_error` is appended. Call cancellation follows the same concurrent, bounded cleanup
path before propagating `CancelledError`.

Sandbox termination is attempted up to three times with 0.25-second and 0.5-second backoffs. A sandbox already
reported as missing or terminated counts as successfully closed. `close()` raises `TimeoutError` when its bounded
cleanup wait expires and `ModalCleanupError` when all termination attempts fail. Timed-out termination and local
stream draining remain tracked; `aclose()` drains tracked cleanup and raises `TimeoutError` if it still cannot
finish, or `ModalCleanupError` for a delayed termination failure.

## Bounded file downloads

`AsyncSandbox.download(..., max_bytes=N)` is the security boundary for files
whose size is controlled by sandbox code. Modal streams the remote file through
a binary `Sandbox.exec` process into an owner-created temporary file, stops and
terminates the sandbox as soon as the limit is exceeded, and atomically replaces
the requested local path only after a clean zero exit. Overflow, cancellation,
and provider failure remove the temporary file and preserve any existing target.
An overflow raises `SandboxDownloadLimitExceeded`.

Omitting `max_bytes` retains the ordinary SDK `copy_to_local` behavior for
backward compatibility. That path has no Gym-enforced byte ceiling and should
be used only when the remote file size is already trusted. Providers that cannot
enforce a requested ceiling may reject the bounded operation.

## Resource and operation mapping

| Gym field or operation | Modal mapping |
|---|---|
| `image` | `Image.from_registry` |
| `ttl_s` | `Sandbox.create(timeout=...)` |
| `ready_timeout_s` | Client deadline around `Sandbox.create` |
| `workdir`, `env`, `metadata` | `workdir`, ephemeral `Secret.from_dict`, and `tags` |
| CPU / memory / GPU | Modal `cpu`, `memory`, and `gpu` requests |
| `exec` | `/bin/sh -c` through binary `Sandbox.exec.aio` streams; per-command env uses an ephemeral secret |
| upload / unbounded download | `Sandbox.filesystem.copy_from_local/copy_to_local` |
| bounded download | Binary `Sandbox.exec` stream into an atomic local temporary file |
| status | `Sandbox.poll.aio` |
| close | `Sandbox.terminate.aio(wait=True)` |

Modal does not expose an ephemeral-disk-size request, so a non-null `SandboxResources.disk_gib` is rejected.
Numeric non-root UIDs are also rejected because Modal has no neutral UID execution field; named users use `su`.
