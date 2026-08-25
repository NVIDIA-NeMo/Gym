# IdeGYM Sandbox Provider

Runs each NeMo Gym sandbox as an [IdeGYM](https://github.com/JetBrains-Research/idegym) *server*: a
Kubernetes pod provisioned through the IdeGYM orchestrator, driven over the HTTP API the orchestrator
forwards to it. Use it when your evaluation environments already live in IdeGYM — SWE-bench-style
repository tasks, IDE-backed environments, or anything else IdeGYM images provide — and you want to
run them through NeMo Gym's agents and benchmarks.

The provider implements the provider-neutral `SandboxProvider` contract, so any sandbox-backed agent
(for example `mini_swe_agent_2`) can use it by pointing at this provider's config.

## Requirements

- A reachable IdeGYM orchestrator, and Basic auth credentials for it.
- The client SDK: `uv pip install 'nemo-gym[idegym]'` (pulls `idegym-client`, `idegym-api`,
  `idegym-common-utils`).
- **An IdeGYM server image.** This is the one requirement that is easy to miss: the pod must run the
  IdeGYM server, because that server is what exposes the API the orchestrator forwards commands to. A
  plain benchmark image (`swebench/sweb.eval.x86_64.*`, say) does not boot one, and starting it will
  fail readiness. See [Images](#images) below.
- `base64`, `wc`, `tail` and `head` in the image — coreutils, present in practically every Linux base
  image. File transfer uses them.

## Quick start

```bash
export IDEGYM_ORCHESTRATOR_URL=idegym.example.com
export IDEGYM_NAMESPACE=idegym
export IDEGYM_AUTH_USERNAME=...
export IDEGYM_AUTH_PASSWORD=...

AGENT=responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml
MODEL=responses_api_models/vllm_model/configs/vllm_model.yaml
gym env start \
    --config $AGENT \
    --config nemo_gym/sandbox/providers/idegym/configs/idegym.yaml \
    --config $MODEL
```

Every shipped provider config binds the same instance name `sandbox`, so switching a benchmark from
OpenSandbox or Docker to IdeGYM is swapping that one config path — no agent edit.

Directly from Python:

```python
import asyncio

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec

provider_config = {
    "idegym": {
        "connection": {"orchestrator_url": "idegym.example.com", "namespace": "idegym"},
        "create": {"run_as_root": True},
    }
}

spec = SandboxSpec(
    image="registry.example.com/idegym/my-env:latest",
    workdir="/testbed",
    resources={"cpu": 2, "memory_mib": 8192, "disk_gib": 30},
    provider_options={"resource_requests": {"cpu": 0.5, "memory_mib": 2048}},
    metadata={"instance_id": "django__django-11099"},
)


async def main() -> None:
    async with AsyncSandbox(provider_config, spec) as sandbox:
        await sandbox.start()
        result = await sandbox.exec("python -m pytest -q", timeout_s=600)
        print(result.stdout or result.stderr)


asyncio.run(main())
# leaving the `async with` block stops the server and deletes its Kubernetes resources
```

Credentials come from `IDEGYM_AUTH_USERNAME` / `IDEGYM_AUTH_PASSWORD` when `connection.username` and
`connection.password` are unset, which is how they should normally be passed.

## Configuration

`configs/idegym.yaml` is the shipped config block, with every knob commented. The provider constructor
takes one section per concern:

| Section | Purpose |
| --- | --- |
| `connection` | Orchestrator URL, namespace, auth, client name, HTTP pooling, tracing. |
| `create` | Readiness budget, retries, generated server names, pod defaults (`run_as_root`, ports), orchestrator polling backoff. |
| `exec` | Default command timeout, client-side overhead, how `exec(user=...)` is honored. |
| `verify` | The post-start working-directory check. |
| `files` | Upload/download chunk sizes and the download size cap. |
| `operations` | Status and teardown timeouts and retries. |
| `attribution` | How the registered IdeGYM client name is derived when it is not pinned. |

### `provider_options`

Per-sandbox IdeGYM options with no neutral equivalent, validated before a pod is allocated:

| Option | Notes |
| --- | --- |
| `resource_requests` | Kubernetes *requests*, paired with `spec.resources` as the limits. |
| `runtime_class_name` | e.g. `gvisor` for a sandboxed runtime. |
| `run_as_root` | Overrides `create.run_as_root` for this sandbox. |
| `node_selector` | Node labels for scheduling. |
| `volumes`, `volume_mounts` | Native Kubernetes shapes. |
| `env_from` | ConfigMap/Secret env import — the only way to put environment on the *pod*. |
| `service_account_name` | ServiceAccount for the pod. |
| `pod_overrides` | Partial `V1PodSpec` deep-merged into the generated pod spec (tolerations, affinity, ...). |
| `reuse_strategy`, `server_kind`, `snapshot`, `max_restarts` | Passed through to IdeGYM. |
| `server_name` | Pins the server name IdeGYM's reuse lookup matches on. Otherwise it is derived from the name prefix and metadata hints. |
| `service_port`, `container_port` | Override the configured ports. |

### `SandboxSpec` mapping

| Field | Mapping |
| --- | --- |
| `image` | The IdeGYM server image. `docker://` is stripped; the rest must be a valid lowercase OCI reference. |
| `workdir` | Not a pod setting — each command `cd`s into it, and create fails early if it does not exist (`verify.check_workdir`). |
| `env` | Exported per command, since IdeGYM can only put ConfigMap/Secret env on the pod (`provider_options.env_from`). |
| `resources` | Kubernetes **limits**. `cpu` → cores/millicores, `memory_mib` → `Mi`, `disk_gib` → `ephemeral-storage`. |
| `files` | Uploaded after start by the sandbox API, through this provider's `upload_file`. |
| `metadata` | Not labels — IdeGYM has no per-server label API. `create.server_name_metadata_keys` folds selected keys into the pod name; the rest is kept for diagnostics. |
| `ready_timeout_s` | Overrides `create.ready_timeout_s`. |
| `ttl_s` | Not enforced; warns. IdeGYM servers live until stopped (its watcher reaps servers whose client stops heartbeating). |
| `ports` | Ignored; warns. An IdeGYM pod exposes only its own API port. |
| `entrypoint` | Rejected. The entrypoint starts the IdeGYM server. |
| `resources.gpu`, `resources.gpu_type` | Not mapped; warns. Request accelerators via `node_selector` or `pod_overrides`. |

Note that IdeGYM's resource quota accounting reads **limits** first, so `spec.resources` is what
counts against your quota rule, not the smaller `resource_requests`.

## How it works

```
AsyncSandbox / Sandbox
        │
   IdeGymProvider ──── spec.py      SandboxSpec  → start-server request
        │         ──── naming.py    RFC-1035 server names, client names
        │         ──── shell.py     cwd / env / user  → one bash script
        │         ──── transfer.py  files → base64 over the bash tool
        │         ──── errors.py    SDK exception → "gone" / "busy" / "timed out"
        │
   IdeGymSession  ←── the only module that imports the idegym SDK
        │
   IdeGYM orchestrator ── forwards ──→ server pod (IdeGYM server + your project)
```

**One registered client per process.** IdeGYM's unit of ownership is a *client*: a registered,
heartbeating entity that owns N server pods, carries the resource quota keyed on its name, and
terminates all of its servers when it stops. Every sandbox created from an identical `connection`
config shares one registered client, reference-counted across providers; the last release
unregisters it, which also cleans up any server that was never closed.

**That client runs on a private event loop** in a daemon thread. The SDK's client is loop-bound (an
httpx session plus a heartbeat task), while NeMo Gym drives sandboxes both from a caller's loop
(`AsyncSandbox`) and from one loop per sandbox (the sync `Sandbox` facade the mini-swe-agent harness
uses). A private loop lets one registration serve every sandbox regardless of who asked.

**Commands are shaped into a script.** IdeGYM's bash tool takes a command and nothing else — each
call is a fresh `bash -c` in the server's project directory with a cleaned environment — so `cwd`,
`env` and `user` are expressed inside the generated script. The script is emitted as one `{ ... }`
group, because IdeGYM runs `source <bash-integration> && <script>` and `&&` binds only to the first
statement.

**Create trusts IdeGYM's readiness wait.** The orchestrator reports a server only once its pod
passes the Kubernetes readiness probe against the server's health endpoint, so `create()` adds just
one check of its own: that `spec.workdir` exists. Anything that fails after provisioning stops the
server.

**Files ride base64 over bash.** IdeGYM's filesystem API is not usable through the orchestrator: its
read endpoint streams raw bytes while the orchestrator forwards requests as JSON text, and its typed
file endpoints only write UTF-8. Uploads are chunked to stay under the sandbox shell's 128 KiB
argument limit; downloads are chunked because each chunk's stdout is persisted as an
async-operation result.

**Status** uses the capabilities call — the cheapest orchestrator call that validates the server's
record *and* reaches the pod. A 404 or 410 means the sandbox is gone; anything else that fails leaves
the status `unknown` rather than guessing.

## Images

`spec.image` must be an image that runs the IdeGYM server. Build those with IdeGYM's own image
builder, publish them to a registry your cluster can pull from, then map upstream benchmark images
onto them with the sandbox spec's `image_rewrites`:

```yaml
mini_swe_agent_2:
  responses_api_agents:
    mini_swe_agent_2:
      sandbox_provider: sandbox
      sandbox_spec:
        image_rewrites:
          - from: "swebench/"
            to: "registry.example.com/idegym-swebench/"
```

Rewrites are ordered and the first matching prefix wins. The provider does not build images.

## Limitations

- **No `endpoint()`.** The orchestrator forwards API requests rather than routing raw TCP to
  arbitrary container ports, so a long-lived service inside the sandbox cannot be reached this way.
- **No PTY sessions.** IdeGYM has no terminal API, so `sandbox.pty` is unavailable.
- **No `serialize()` / `connect()`.** A server is only reachable through the registered client that
  owns it, so sharing one across processes needs a sandbox server in front of this provider.
- **No `user` switching by default.** IdeGYM's bash tool has no user field; commands run as the
  container's user, which `provider_options.run_as_root` controls at pod level. `exec.user_mode` can
  wrap commands in `runuser` or `su` for images that ship them.
- **Command and output sizes.** A command may not exceed ~100 KiB once shaped into a script, and the
  IdeGYM executor strips leading and trailing whitespace from stdout and stderr.
- **`ttl_s` is not enforced**, so a crashed process leaks pods until its client stops heartbeating
  and IdeGYM's watcher reaps them.

## Development

```bash
uv pip install 'nemo-gym[idegym]'
pytest tests/unit_tests/test_idegym_provider.py tests/unit_tests/test_idegym_session.py
```

The provider tests need no orchestrator: they run against a fake session, and the file-transfer and
command-shaping tests execute the generated scripts with local `bash`. The session tests that bind
against the real SDK signatures skip when `idegym-client` is not installed.

See [DESIGN-DECISIONS.md](DESIGN-DECISIONS.md) for the choices behind this layout and the
alternatives that were rejected.
