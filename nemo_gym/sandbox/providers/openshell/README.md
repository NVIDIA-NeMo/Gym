# OpenShell sandbox provider

Runs NeMo Gym sandboxes on an [OpenShell](https://github.com/NVIDIA/OpenShell) gateway. OpenShell
is a safe, private runtime for autonomous AI agents: the gateway provisions isolated
container/MicroVM sandboxes through a compute driver (Docker, Podman, Kubernetes, MicroVM) and
enforces policy-based egress on every outbound connection.

The provider talks to the gateway's gRPC control plane through the synchronous
[`openshell`](https://pypi.org/project/openshell/) SDK (installed with the `nemo-gym[sandbox]`
extra). Blocking SDK calls run on a thread pool bounded by `exec.concurrency`.

## Local quickstart (Docker compute driver)

Run a local gateway with OpenShell's compose setup, which uses prebuilt GHCR images and a
plaintext (no auth) control plane on `localhost:8080`:

```bash
git clone https://github.com/NVIDIA/OpenShell
cd OpenShell/deploy/docker
docker compose up -d
curl -sf http://localhost:8081/healthz   # gateway health endpoint
```

Then point the provider at it (the shipped config already defaults to `localhost:8080`):

```bash
ng_run "+config_paths=[$AGENT, nemo_gym/sandbox/providers/openshell/configs/openshell.yaml, $MODEL]"
```

Or use it directly:

```python
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxSpec

sandbox = AsyncSandbox({"openshell": {"connection": {"endpoint": "localhost:8080"}}})
await sandbox.start(SandboxSpec(image="python:3.12-slim"))
result = await sandbox.exec("echo hello")
await sandbox.stop()
```

When `spec.image` is unset, the gateway's configured default image is used
(`ghcr.io/nvidia/openshell-community/sandboxes/base:latest` in the compose setup).

## Spec mapping and caveats

| `SandboxSpec` field | OpenShell behavior |
|---|---|
| `image` | `SandboxTemplate.image` (unset -> gateway default image) |
| `env` | `SandboxSpec.environment`, also re-applied per exec |
| `metadata` | gateway sandbox labels |
| `workdir` | default `workdir` for every exec (no create-time equivalent) |
| `resources.gpu` | `ResourceRequirements.gpu.count` |
| `resources.cpu/memory_mib/disk_gib/gpu_type` | not supported by the gateway API; ignored with a warning |
| `ttl_s` | not enforced (sandboxes live until `close()`); logs a warning |
| `entrypoint` | unsupported; raises (the OpenShell supervisor owns the entrypoint) |
| `provider_options.providers` | OpenShell credential-provider names attached to the sandbox |

- `exec(user=...)` is ignored with a warning: the OpenShell exec API has no user field, so
  commands run as the sandbox's default user (non-root in the default sandbox image).
- The SDK has no file-transfer API. `upload_file` streams bytes through exec stdin
  (`mkdir -p && cat > target`); `download_file` round-trips through `base64` on stdout, so the
  sandbox image must provide `base64` (coreutils/busybox both do). Because uploads run as the
  sandbox default user, target paths must be writable by that user (e.g. under `/tmp` or the
  user's home in the default image).
- Exec timeouts are enforced by the gateway (`timeout_seconds`); the SDK extends its gRPC
  deadline past the command timeout automatically.

## Authenticated gateways

For OIDC gateways set `connection.bearer_token` (e.g. via `OPENSHELL_BEARER_TOKEN`); for
TLS/mTLS set `connection.tls_ca_path` / `tls_cert_path` / `tls_key_path`. All unset means a
plaintext channel, matching the local compose gateway (`disable_tls = true`).
