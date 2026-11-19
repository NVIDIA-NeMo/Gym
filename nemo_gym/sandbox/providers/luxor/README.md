# Luxor sandbox provider

Runs NeMo Gym sandboxes on [Luxor](https://luxor.ai/) through the asynchronous Luxor AI Python SDK
(`luxor-ai`, formerly published as `tenki`).
The provider creates an ephemeral sandbox, waits until it can execute commands, and terminates it
when the Gym sandbox closes.

## Install and authenticate

The SDK is included in the `nemo-gym[sandbox]` extra:

```bash
uv sync --extra sandbox
export LUXOR_API_KEY="..."
```

The hosted API and gateway are the SDK defaults. For another deployment, set `LUXOR_API_URL` and
`LUXOR_SANDBOX_GATEWAY_URL`, or configure `connection.base_url` and `connection.gateway_url`.

## Use with `ng_run`

Add the bundled provider config to the normal config stack:

```bash
ng_run "+config_paths=[$AGENT, nemo_gym/sandbox/providers/luxor/configs/luxor.yaml, $MODEL]"
```

## Use from Python

```python
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec

sandbox = AsyncSandbox({"luxor": {}}, SandboxSpec(ttl_s=900))
await sandbox.start()
try:
    result = await sandbox.exec("python --version")
    print(result.stdout)
finally:
    await sandbox.stop()
```

Luxor's default managed base image is sufficient for the integration. A template, snapshot, or
volume is not required. To select a Luxor registry image, set `SandboxSpec.image` to its managed
registry reference. Luxor does not pull arbitrary public OCI references at sandbox-create time;
build the image in Luxor's registry first and map it with `create.image_map`.

## Images built from OCI references

A Luxor sandbox is a microVM on Luxor's base OS, not a container, so an OCI image cannot be booted
as-is. The bundled helper builds one Luxor image per OCI image by downloading the image's
filesystem and moving chosen directories onto the base image:

```bash
python -m nemo_gym.sandbox.providers.luxor.build \
  --images-file images.txt \
  --output image_map.yaml
```

`--path` (repeatable) names the directories to import and defaults to `/testbed` and
`/opt/miniconda3`, the SWE-bench layout. System packages installed elsewhere in the OCI image are
not carried over, so this suits task images whose environment lives in a few directories. Paste
the generated mapping under the provider's `create` block:

```yaml
sandbox:
  luxor:
    create:
      image_map:
        "docker.io/acme/task:1.0": "workspace/gym-task-1-0-f00ee9d593d9@sha256:..."
```

`LuxorProvider.create()` looks `SandboxSpec.image` up in `image_map` and otherwise passes it
through as a Luxor registry reference. Image names are deterministic for the OCI reference and the
imported paths, and a ready image is reused unless `--rebuild` is passed. Builds are provisioning
operations that create Luxor templates and images; the provider never runs them implicitly. A
local build timeout stops waiting but cannot cancel a build Luxor has already accepted. In a batch,
the first failure prevents queued builds from starting; pass `--continue-on-error` to keep going.

## Running SWE-bench with `mini_swe_agent_2`

Build the task images, then add the provider config and the generated mapping to the agent's
config stack. `run_golden=true` applies the reference patch instead of calling the model, which
checks the images before a model run:

```bash
gym env start \
  --config responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml \
  --config nemo_gym/sandbox/providers/luxor/configs/luxor.yaml \
  --config luxor_image_map.yaml \
  --model-type vllm_model \
  ++mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.run_golden=true
```

## Field mapping

| `SandboxSpec` field                   | Luxor behavior                                                            |
| ------------------------------------- | ------------------------------------------------------------------------- |
| `image`                               | Luxor managed registry image; unset uses Luxor's default base image       |
| `ttl_s`                               | `max_duration`; defaults to the provider's finite `create.max_duration_s` |
| `ready_timeout_s`                     | SDK create wait deadline                                                  |
| `workdir`                             | Default `cwd` for commands and relative file transfers                    |
| `env`                                 | Sandbox environment at creation; per-command values are also supported    |
| `metadata`                            | Luxor metadata, merged with Gym team/user/workload/run attribution        |
| `resources.cpu`                       | Rounded up to integer `cpu_cores`                                         |
| `resources.memory_mib`                | `memory_mb`                                                               |
| `resources.disk_gib`                  | `disk_size_gb`                                                            |
| `resources.gpu`, `resources.gpu_type` | Unsupported; raises instead of silently ignoring the request              |
| `files`                               | Uploaded by Gym after readiness; arbitrary absolute paths are supported   |
| `ports`                               | Exposed during creation and resolved through `sandbox.endpoint(port)`     |
| `entrypoint`                          | Unsupported because Luxor owns the sandbox runtime entrypoint             |

`provider_options` accepts:

- `workspace_id`, `name`, `allow_inbound`, `allow_outbound`, and `idle_timeout_minutes`
- `tags`, `sticky`, and `wait_for_runtime`
- `template`, `snapshot_id`, or `volumes` for optional Luxor-native provisioning
- `resource_requests`, accepted for compatibility with OpenSandbox-oriented agent configs and
  ignored: a Luxor sandbox receives `SandboxSpec.resources` in full

Only one of `SandboxSpec.image`, `provider_options.template`, and
`provider_options.snapshot_id` may be set. Unknown or incorrectly typed options raise before
creation.

Other provider-specific options do not carry across backends and raise before creation. The
bundled mini-SWE config works unchanged: its common `resources` block maps to Luxor's CPU, memory,
and disk fields, and its `resource_requests` option is ignored.

## Lifecycle and permissions

Every sandbox receives a server-enforced maximum duration, including when `ttl_s` is omitted.
Normal shutdown requests termination and waits until Luxor reports a terminal state. A sandbox
admitted before a readiness, probe, port, or initial-file failure is terminated on the failure
path as well.

Commands run as the image's `tenki` user by default. Pass `user="root"` for Luxor's privileged
execution mode. File upload and download use the SDK filesystem API under `/home/tenki`; transfers
to other paths use a temporary staging file and a privileged copy so Gym workloads can keep their
existing absolute paths.

Luxor sandboxes are addressable by session ID, so Gym can serialize a handle and reconnect from a
second process through the provider's standard `ConnectableProvider` capability. The receiving
process needs the same Luxor authentication and connection settings.
