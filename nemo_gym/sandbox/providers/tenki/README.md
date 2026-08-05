# Tenki sandbox provider

Runs NeMo Gym sandboxes on [Tenki](https://tenki.cloud/) through Tenki's asynchronous Python SDK.
The provider creates an ephemeral sandbox, waits until it can execute commands, and terminates it
when the Gym sandbox closes.

## Install and authenticate

The Tenki SDK is included in the `nemo-gym[sandbox]` extra:

```bash
uv sync --extra sandbox
export TENKI_API_KEY="..."
```

The hosted API and gateway are the SDK defaults. For another deployment, set `TENKI_BASE_URL` and
`TENKI_GATEWAY_URL`, or configure `connection.base_url` and `connection.gateway_url`.

## Use with `ng_run`

Add the bundled provider config to the normal config stack:

```bash
ng_run "+config_paths=[$AGENT, nemo_gym/sandbox/providers/tenki/configs/tenki.yaml, $MODEL]"
```

## Use from Python

```python
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec

sandbox = AsyncSandbox({"tenki": {}}, SandboxSpec(ttl_s=900))
await sandbox.start()
try:
    result = await sandbox.exec("python --version")
    print(result.stdout)
finally:
    await sandbox.stop()
```

Tenki's default managed base image is sufficient for the integration. A template, snapshot, or
volume is not required. To select a Tenki registry image, set `SandboxSpec.image` to its managed
registry reference. Tenki does not pull arbitrary public OCI references at sandbox-create time;
import or build the image in Tenki's registry first.

## Field mapping

| `SandboxSpec` field                   | Tenki behavior                                                            |
| ------------------------------------- | ------------------------------------------------------------------------- |
| `image`                               | Tenki managed registry image; unset uses Tenki's default base image       |
| `ttl_s`                               | `max_duration`; defaults to the provider's finite `create.max_duration_s` |
| `ready_timeout_s`                     | SDK create wait deadline                                                  |
| `workdir`                             | Default `cwd` for commands and relative file transfers                    |
| `env`                                 | Sandbox environment at creation; per-command values are also supported    |
| `metadata`                            | Tenki metadata, merged with Gym team/user/workload/run attribution        |
| `resources.cpu`                       | Rounded up to integer `cpu_cores`                                         |
| `resources.memory_mib`                | `memory_mb`                                                               |
| `resources.disk_gib`                  | `disk_size_gb`                                                            |
| `resources.gpu`, `resources.gpu_type` | Unsupported; raises instead of silently ignoring the request              |
| `files`                               | Uploaded by Gym after readiness; arbitrary absolute paths are supported   |
| `ports`                               | Exposed during creation and resolved through `sandbox.endpoint(port)`     |
| `entrypoint`                          | Unsupported because Tenki owns the sandbox runtime entrypoint             |

`provider_options` accepts:

- `workspace_id`, `name`, `allow_inbound`, `allow_outbound`, and `idle_timeout_minutes`
- `tags`, `sticky`, and `wait_for_runtime`
- `template`, `snapshot_id`, or `volumes` for optional Tenki-native provisioning

Only one of `SandboxSpec.image`, `provider_options.template`, and
`provider_options.snapshot_id` may be set. Unknown or incorrectly typed options raise before
creation.

Provider-specific options do not carry across backends. For example, the bundled mini-SWE config
contains OpenSandbox's `resource_requests` option; remove or override that mapping when selecting
Tenki. Its common `resources` block already maps to Tenki's CPU, memory, and disk fields.

## Lifecycle and permissions

Every sandbox receives a server-enforced maximum duration, including when `ttl_s` is omitted.
Normal shutdown requests termination and waits until Tenki reports a terminal state. A sandbox
admitted before a readiness, probe, port, or initial-file failure is terminated on the failure
path as well.

Commands run as the image's `tenki` user by default. Pass `user="root"` for Tenki's privileged
execution mode. File upload and download use the SDK filesystem API under `/home/tenki`; transfers
to other paths use a temporary staging file and a privileged copy so Gym workloads can keep their
existing absolute paths.

Tenki sandboxes are addressable by session ID, so Gym can serialize a handle and reconnect from a
second process through the provider's standard `ConnectableProvider` capability. The receiving
process needs the same Tenki authentication and connection settings.
