# Substrate sandbox provider

Runs sandboxes as [Substrate](https://github.com/agent-substrate/substrate)
actors. Substrate can pause or suspend an actor and bring it back on demand, so
an idle sandbox holds no worker; for a benchmark run that spends most of its
wall-clock waiting on a policy model, that is the reason to use this backend.

## How it works

```
provider ── gRPC ──────> ateapi          Create / Pause / Suspend / Delete actor
         └─ HTTP ──────> atenet router ─> sandbox server (inside the actor)
```

Control operations go to `ateapi`. Commands and file transfers go to a small
HTTP server inside the actor, addressed through the `atenet` router by the
`ate-target-actor: <atespace>/<actor>` header. The router resumes a paused or
suspended actor before routing to it, so nothing has to resume it explicitly.

The in-actor server is Substrate's `demos/sandbox` binary. It is static, so an
`ActorTemplate` mounts its image as a read-only image volume into the benchmark
image, which runs untouched, and sets it as the container command.

## Setup

Build the environment the usual way, `uv sync --extra dev --extra sandbox`,
which is also what selects a supported interpreter. Building it by hand can
pick up a Python that `requires-python` admits but the dependencies do not: on
3.14, Hydra fails while assembling its argument parser (`ValueError: badly
formed help string`) and the servers exit before reading any configuration.

The cluster's nodes must be **amd64**: SWE-bench publishes x86_64 images only
and gVisor does not emulate, so on arm64 an actor fails to start with
`runsc start`: exit status 128.

The cluster needs one `ActorTemplate` per image. An `ActorTemplate` spec is
immutable and an actor names a template rather than an image, so the set is
built ahead of a run rather than on demand. Substrate ships a generator; pass
each image exactly as NeMo Gym will ask for it, because the template name is
derived from that string on both sides:

```bash
# in a Substrate checkout, against the cluster kubectl points at
BUCKET_NAME=<snapshot bucket> demos/sandbox/gym-templates.sh \
  swebench/sweb.eval.x86_64.django_1776_django-11099:latest
```

That derivation is pinned by a unit test; if it changes, the generator must
change with it.

Then expose the two endpoints and run:

```bash
kubectl port-forward -n ate-system svc/api 8080:443 &
kubectl port-forward -n ate-system svc/atenet-router 8000:80 &

gym env start \
  --config responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml \
  --config nemo_gym/sandbox/providers/substrate/configs/substrate.yaml \
  --config <model>.yaml
```

## Authentication

`ateapi` rejects unauthenticated calls. The provider mints a ServiceAccount
token through the Kubernetes `TokenRequest` API, the same way `kubectl-ate`
authenticates, using the local kubeconfig or in-cluster credentials. The
serving CA is read from the cluster's `ClusterTrustBundle`.

Credentials are resolved inside the provider rather than through `${oc.env:...}`
interpolation in the config block: the per-instance config is written to disk,
so a token placed there would be written in plaintext, and it would not reach a
remote worker anyway.

## Configuration

```yaml
sandbox:
  substrate:
    connection:
      ateapi_addr: localhost:8080
      atenet_addr: localhost:8000
      atespace: nemo-gym            # actors are created here
      template_namespace: nemo-gym  # templates are looked up here
    create:
      suspend_policy: pause_after_exec   # never | after_exec | pause_after_exec
    exec:
      default_timeout_s: 1800
      concurrency: 32
```

### `suspend_policy`

`pause_after_exec` calls `PauseActor` once each step finishes: the actor's
worker is freed and its snapshot stays on the node, so the next step resumes
without a download but has to land on the same node.

`after_exec` calls `SuspendActor` instead: the snapshot is pushed to object
storage and any node can resume it. `SuspendActor` is synchronous, so this pays
a checkpoint upload at the end of every step and a restore download at the
start of the next, both on the critical path.

`never` leaves the actor running and is the control to measure against.

## What this provider does not honor

Declined rather than faked:

| Spec field | Why |
| --- | --- |
| `resources` | Actors are sized by their `WorkerPool`, not per sandbox. |
| `ttl_s` | Substrate has no TTL; a paused actor already costs almost nothing. |
| `entrypoint` | The container command runs the sandbox server. |

`user` must be `root` (or unset). Every container in an actor shares one fixed
identity, so there is no user to switch to; anything else is refused rather than
silently ignored.

Commands run under `bash` when the image has it, else `sh`: `mini_swe_agent_2`
sources conda's `conda.sh`, which is not portable to dash.

## Regenerating the gRPC stubs

`_ateapi/` holds stubs generated from Substrate's `ateapi.proto`:

```bash
scripts/generate_ateapi_stubs.sh /path/to/substrate
```
