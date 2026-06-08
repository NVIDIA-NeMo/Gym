# ECS Fargate sandbox provider

Runs each `nemo_gym.sandbox` sandbox as an AWS ECS Fargate task behind an SSH
sidecar. It implements the provider-neutral `SandboxProvider` contract, so any
sandbox-backed agent (not just mini-swe-agent) can use it by setting
`sandbox_provider.ecs_fargate` in its config.

## Prerequisites

- **Infrastructure** provisioned in the target account/region. The reference
  Terraform stack publishes its outputs to SSM at
  `/<ssm_project>/ecs-sandbox/config` (`ssm_project` defaults to `harbor`):
  cluster, subnets, security groups, task/execution roles, ECR mirror, EFS, and
  the SSH-sidecar key ARNs. A missing parameter raises an actionable error.
- **Credentials**: `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` (or an instance
  role) plus `AWS_REGION`.
- **Network**: the host must reach each task's SSH sidecar port (`52222`) — run
  inside the sandbox VPC/peered network, or allow the host IP on the sidecar
  security group. Exec, file transfer, and model reverse-tunnels ride this SSH
  connection.

## Configuration

Region-only is enough; everything else auto-discovers from SSM (explicit YAML
always wins):

```yaml
sandbox_provider:
  ecs_fargate:
    region: ${oc.env:AWS_REGION}
    cpu: "2048"
    memory: "8192"
    ephemeral_storage_gib: 50
```

Common fields:

| Field | Default | Purpose |
| --- | --- | --- |
| `region` | — | AWS region; enables SSM auto-discovery when `cluster` is omitted |
| `cpu` / `memory` | `"4096"` / `"8192"` | Fargate task size (CPU units / MiB) |
| `ephemeral_storage_gib` | task default | Task ephemeral disk |
| `auto_mirror` | `true` | Pull a missing public image into the ECR mirror on demand (see below) |
| `ssm_project` | `harbor` | SSM namespace for auto-discovery |
| `environment_dir` | — | Build a task image from a Dockerfile dir via CodeBuild instead of using a prebuilt image |

Per-sandbox `ready_timeout_s`, `env`, `files`, `metadata`, and
`provider_options` (e.g. `platform`, `outside_endpoints`) come from the
`SandboxSpec`.

## Images and on-demand mirroring

ECS pulls task images from the account ECR mirror, not their origin registry. A
bare/public image (e.g. `docker.io/swebench/sweb.eval.x86_64.<id>:latest`)
resolves to the mirror tag `<ecr_repository>:<sanitized-name>`. Resolution order:

1. `environment_dir` set → build the image via CodeBuild and use it.
2. Image is already an ECR reference → use verbatim (never re-mirrored).
3. Bare/public name + `auto_mirror=true` → mirror into ECR on demand (CodeBuild
   pull → retag → push) during `create`, then launch. Concurrent tasks for the
   same image de-duplicate onto one build.

Set `auto_mirror: false` to require a pre-populated mirror and fail fast on a
miss. The first task for a new image waits on a one-time build (~1–3 min for
typical SWE-bench images); later tasks hit the ECR cache.

## Lifecycle

`create` launches the task + SSH sidecar and returns once the exec server is
healthy. `exec`, `upload_file`, `download_file`, `status`, and `close` delegate
to the per-sandbox engine over the SSH tunnel. `outside_endpoints` (via
`spec.provider_options`) open reverse tunnels so an in-sandbox process can reach
a host-side endpoint (e.g. a model server).

## Security

The sidecar security group in the reference stack allows `0.0.0.0/0` on `52222`
for convenience. Restrict it to the orchestrator's egress IP (or move to a
private/Teleport path) before non-smoke use.
