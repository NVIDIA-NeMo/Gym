# OpenShell sandbox provider

Runs NeMo Gym sandboxes on an [OpenShell](https://github.com/NVIDIA/OpenShell) gateway. OpenShell
is a safe, private runtime for autonomous AI agents: the gateway provisions isolated
container/MicroVM sandboxes through a compute driver (Docker, Podman, Kubernetes, MicroVM) and
enforces policy-based egress on every outbound connection.

The provider talks to the gateway's gRPC control plane through the synchronous
[`openshell`](https://pypi.org/project/openshell/) SDK (installed with the `nemo-gym[sandbox]`
extra). Blocking SDK calls run on a thread pool bounded by `exec.concurrency`; providers built
from identical connection configs share one gRPC channel and one pool, so per-sandbox provider
instances (the `AsyncSandbox` pattern) do not multiply threads or channels. Sandboxes live in
the gateway workspace set by `connection.workspace` (`default` unless overridden).

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
| `resources.cpu/memory_mib/disk_gib/gpu_type` | not mapped by this provider; OpenShell exposes driver-specific limits through `SandboxTemplate.resources` — pass `provider_options.template_resources` (a warning notes the redirection) |
| `resource_requests` | scheduling hint ignored; OpenShell does not expose separate request/limit semantics through this provider |
| `ttl_s` | not enforced (sandboxes live until `close()`); logs a warning |
| `entrypoint` | unsupported; raises (the OpenShell supervisor owns the entrypoint) |
| `provider_options.providers` | OpenShell credential-provider names attached to the sandbox |
| `provider_options.policy` | OpenShell `SandboxPolicy` as a mapping or a policy YAML path; unset falls back to the policy baked into the sandbox image |
| `provider_options.template_resources` | free-form driver resource passthrough (`SandboxTemplate.resources` Struct); which keys are honored is up to the gateway's compute driver |
| `provider_options.driver_config` | free-form driver config passthrough (`SandboxTemplate.driver_config` Struct) |

Unknown `provider_options` keys raise `ValueError`; wrong-typed values raise `TypeError`.

- `exec(user=...)` is ignored with a warning: the OpenShell exec API has no user field, so
  commands run as the sandbox's default user (non-root in the default sandbox image).
- The SDK has no file-transfer API. `upload_file` streams bytes through exec stdin
  (`mkdir -p && cat > target`, then `cat >>` appends), chunked at `exec.upload_chunk_bytes`
  (512 KiB default) because each chunk is one gRPC message that must stay under the gateway's
  max decode size (1 MiB on current gateway builds). `download_file` round-trips through `base64` on stdout,
  so the sandbox image must provide `base64` (coreutils/busybox both do); the whole file is
  buffered in memory (inflated 4/3 by base64), so prefer it for small-to-medium artifacts
  rather than large archives. Because uploads run as the sandbox default user, target paths
  must be writable by that user (e.g. under `/tmp` or the user's home in the default image).
- Exec timeouts are enforced by the gateway (`timeout_seconds`); the SDK extends its gRPC
  deadline past the command timeout automatically.
- Transient `CreateSandbox` failures (`UNAVAILABLE`, `RESOURCE_EXHAUSTED`, ...) are retried
  with backoff (`create.retries`); an attempt that actually committed is recovered via
  `ALREADY_EXISTS` -> `GetSandbox`. Exec is deliberately not retried: commands are not
  idempotent, and infrastructure failures are distinguishable by
  `SandboxExecResult.error_type` (`"sandbox"` / `"timeout"`).

## Authenticated gateways

For OIDC gateways set `connection.bearer_token` (e.g. via `OPENSHELL_BEARER_TOKEN`); for
TLS/mTLS set `connection.tls_ca_path` / `tls_cert_path` / `tls_key_path`. All unset means a
plaintext channel, matching the local compose gateway (`disable_tls = true`).
