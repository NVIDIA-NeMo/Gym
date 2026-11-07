# OpenShell sandbox provider

Runs NeMo Gym sandboxes on an [OpenShell](https://github.com/NVIDIA/OpenShell) gateway. OpenShell
is a safe, private runtime for autonomous AI agents: the gateway provisions isolated
container/MicroVM sandboxes through a compute driver (Docker, Podman, Kubernetes, MicroVM) and
enforces policy-based egress on every outbound connection.

The provider talks to the gateway's gRPC control plane through the synchronous
[`openshell`](https://pypi.org/project/openshell/) SDK (installed with the `nemo-gym[openshell]`
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
| `metadata` | gateway sandbox labels (values are sanitized client-side so the gateway never rejects them: invalid characters become `_`, at most 63 characters; keys pass through, and the gateway rejects an invalid key with `INVALID_ARGUMENT`) |
| `workdir` | default `workdir` for every exec (no create-time equivalent) |
| `resources.gpu` | `ResourceRequirements.gpu.count` |
| `resources.cpu/memory_mib/disk_gib/gpu_type` | not mapped by this provider; OpenShell exposes driver-specific limits through `SandboxTemplate.resources` — pass `provider_options.template_resources` (a warning notes the redirection) |
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
- Exec timeouts end the gateway exec stream (`timeout_seconds`); they do not kill the process, which may keep
  running in the sandbox. The SDK extends its gRPC deadline past the command timeout automatically. The 0.1.x
  exec stream has no timeout flag: the timeout arrives as exit code 124. The provider reports
  `error_type="timeout"` (code 124) only when exit code 124 arrives after about the full timeout; a command that
  exits 124 quickly stays a plain exit code.
- `close()` does not block. `DeleteSandbox` takes the sandbox pod's termination grace period (~31s on a
  Kubernetes-backed gateway), so `close()` hands one blocking delete function to a delete worker thread and
  returns. The worker deletes on its own client with a long deadline (`operations.delete_rpc_timeout_s`, default
  120s) and `allow_missing`. A client deadline does not cancel a delete the gateway has started, so after a
  deadline the worker waits for NOT_FOUND and re-issues the delete (up to 3 deadlines) only if the sandbox is
  still there. Transient failures (`UNAVAILABLE`, `ABORTED`, ...) and `FAILED_PRECONDITION` (a create's driver
  operation is still pending) are retried until `operations.close_timeout_s`. After the delete the worker polls
  until NOT_FOUND (`operations.close_wait_deleted`, `operations.close_timeout_s`). Every outcome is logged with
  the sandbox name; a delete that fails for good is logged as possibly leaked and then forgotten. The calling
  `AsyncSandbox` is already closed, so nothing retries it later.
  - The delete workers (`operations.delete_concurrency`, default 32, separate from `exec.concurrency`) live for
    the whole process and `aclose()` never waits for them or cancels them. Nothing bounds the queue: a warning
    logs the queue depth when it exceeds the workers, and the sandboxes stay allocated until it drains, at roughly
    one delete per second at 32 workers.
  - CPython runs already-queued work and joins non-daemon worker threads at interpreter exit, so the delete
    finishes after `sys.exit()`, after the event loop is gone, and after a normal `asyncio.run()` end. Normal
    interpreter exit waits for all queued deletes, so its delay depends on the backlog and the worker count. Each
    job can use up to three delete RPC deadlines, plus retry delays and the deletion wait (an in-flight polling
    RPC can also run past `close_timeout_s`). Hard termination (SIGKILL, `os._exit`) bypasses pending cleanup,
    and OpenShell has no TTL, so a failed or skipped cleanup can leave a sandbox allocated.
  - `gym env` stops servers with SIGKILL one second after the shutdown signal by default, so deletes queued by
    the last episodes of a run are lost. Raise `server_shutdown_timeout_seconds` to cover the delete budget when
    you use OpenShell.
  - At `aclose()`, when this process has no sandbox live or being created, it reaps this process's orphaned
    sandboxes: it lists the gateway sandboxes labelled `nemo-gym.nvidia.com/process=<this process>` and
    `nemo-gym.sandbox=1` (needs `attribution.enabled`) and deletes those it does not track, for example a
    create that was cancelled after the gateway committed it. A create's name is tracked before the RPC is
    issued, so the reap never takes a sandbox another episode is still creating, and a create cancelled
    during the RPC is deleted when the RPC returns. The per-process label (written last, so caller metadata
    cannot replace it) keeps the reap from touching sibling processes of the same job, which share the run id.
- A sandbox that never becomes READY (for example an unpullable image) fails after `create.ready_timeout_s`.
  The SDK exposes no pull-specific signal, so the provider does not guess; the error message includes the
  conditions the gateway reports (`SandboxStatus.conditions`) for diagnosis.
- Transient `CreateSandbox` failures (`UNAVAILABLE`, `RESOURCE_EXHAUSTED`, ...) are retried
  with backoff (`create.retries`); an attempt that actually committed is recovered via
  `ALREADY_EXISTS` -> `GetSandbox`. Exec is deliberately not retried: commands are not
  idempotent, and infrastructure failures are distinguishable by
  `SandboxExecResult.error_type` (`"sandbox"` / `"timeout"`).

## Authenticated gateways

For OIDC gateways set `connection.bearer_token` (e.g. via `OPENSHELL_BEARER_TOKEN`); for
TLS/mTLS set `connection.tls_ca_path` / `tls_cert_path` / `tls_key_path`. All unset means a
plaintext channel, matching the local compose gateway (`disable_tls = true`).

## Attribution labels

Every sandbox also gets job attribution labels (`team`, `user`, `workload`, `cluster`, `run`), prefixed by
`attribution.key_prefix` (default `nemo-gym.nvidia.com/`), so sandboxes can be listed or cleaned up per
run, for example `kubectl get pods -l nemo-gym.nvidia.com/run=<id>`. Unset fields auto-detect from
`NEMO_GYM_*` / Slurm environment variables; see `nemo_gym.sandbox.attribution.AttributionConfig`.
Explicit `SandboxSpec.metadata` keys win over attribution keys, and the `nemo-gym.sandbox=1` marker
label is always set. Set `attribution.enabled: false` to turn attribution off. The OpenShell SDK
and protos expose no label validation of their own; labels are sanitized client-side because the
gateway stores them as Kubernetes labels.

## Network access and egress policy

OpenShell denies all egress for a sandbox that has no network policy, and a gateway can be configured
without a global policy. In that case a Gym sandbox reaches nothing, not even the model endpoint, until
a policy is passed at create time.

- `network_access: model_only | model_and_tools` is not supported for this provider: `restricted_network_policy`
  raises for it. Set `provider_options.policy` to a full `SandboxPolicy` mapping or YAML path instead.
  `network_policies` is a map keyed by rule name; this rule allows exactly those `host:port` endpoints for any
  binary (`/**` matches every absolute path) and nothing else:

  ```yaml
  provider_options:
    policy:
      version: 1
      network_policies:
        model_endpoint:
          name: model_endpoint
          endpoints:
            - {host: model.example.com, port: 443}
            - {host: 10.1.2.3, port: 8000, allowed_ips: ["10.1.2.3/32"]}
          binaries:
            - {path: "/**"}
  ```

  DNS and TLS use the OpenShell defaults: names are resolved by the sandbox supervisor, and
  the supervisor proxy terminates TLS with its own CA, which it installs in the sandbox.
- Exact hostnames may resolve to private addresses. A private IP-literal endpoint should also list `allowed_ips` for that
  address (a `/32` or `/128`). Loopback, link-local and
  metadata addresses stay blocked by OpenShell regardless.
- Endpoints must be an FQDN or an IP address. A short Kubernetes Service name such as `service.namespace` may not
  resolve inside the sandbox; use `service.namespace.svc.cluster.local` or the IP.
- The supervisor proxy terminates TLS with its own CA and exposes it to the sandbox through `SSL_CERT_FILE`,
  `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `GIT_SSL_CAINFO`, `NODE_EXTRA_CA_CERTS` and `DENO_CERT`. Tools inside the sandbox must keep
  those variables (and must not use a bundled CA store only), or certificate verification fails.
- Extra hosts (such as a package index) and L7 rules go in the same `provider_options.policy`.
