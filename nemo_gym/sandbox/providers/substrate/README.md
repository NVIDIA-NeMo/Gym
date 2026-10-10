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
When no worker has room the router parks the request for its budget and then
answers 503; the provider waits and sends the request again, because it never
reached the actor. A 503 for an actor mid-transition or a control-plane
conflict is retried for about a minute; one that names a crashed or deleted
actor, or that the provider cannot explain, fails the step at once.

The in-actor server is Substrate's `demos/sandbox` binary. It is static, so an
`ActorTemplate` mounts its image as a read-only image volume into the benchmark
image, which runs untouched, and sets it as the container command.

## Setup

### Cluster

Any Substrate install works; two settings matter for this workload.

- **Router timeouts.** `atenet-router` bounds one request by `--route-timeout`
  (default 5 min) and holds a request waiting for a worker for
  `--parked-request-budget` (default 5 s). Set the first above the longest
  exec you allow (`exec.default_timeout_s`, and mini-swe-agent's
  `eval_timeout`) and the second to how long a step may wait for a slot, for
  example `--route-timeout=1830s --parked-request-budget=120s`.
- **Pool sizing.** A worker hosts several actors at once; each actor subtracts
  its template's cpu and memory limits from the worker's capacity. Size the
  `WorkerPool` and the templates together: SWE-bench test suites peak at 2 to
  4 GiB, so a 12 GiB worker holds three 4 GiB actors, and the sandboxes beyond
  that wait in the router until one is paused.

### Templates

The cluster needs one `ActorTemplate` per image. An `ActorTemplate` spec is
immutable and an actor names a template rather than an image, so the set is
built ahead of a run rather than on demand. Substrate ships a generator; pass
each image as NeMo Gym will ask for it, with or without the `docker.io/`
prefix, since both sides drop it before deriving the name:

```bash
# in a Substrate checkout, against the cluster kubectl points at
KO_DOCKER_REPO=<registry> BUCKET_NAME=<snapshot bucket> WORKLOAD=<pool label> \
  ACTOR_CPU=1 ACTOR_MEMORY=4Gi demos/sandbox/gym-templates.sh \
  swebench/sweb.eval.x86_64.django_1776_django-11099:latest
```

The provider derives a template's name from the image it is asked for, so the
two sides agree without a shared lookup table. That derivation is pinned by a
unit test for a real SWE-bench image; if it changes, the generator must change
with it.

### Run

Build the environment with `uv sync --extra dev --extra sandbox`, then start the
servers from the activated venv. Running them through `uv run` makes Ray's
`uv run` hook demand a `pyproject.toml` inside the agent's directory and the
`mini_swe_agent_2` server exits at startup.

```bash
kubectl port-forward -n ate-system svc/api 8080:443 &
kubectl port-forward -n ate-system svc/atenet-router 8000:80 &

source .venv/bin/activate
gym env start "+config_paths=[responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml,nemo_gym/sandbox/providers/substrate/configs/substrate.yaml,<model>.yaml]"
```

The cluster's nodes must be **amd64**: SWE-bench publishes x86_64 images only
and gVisor does not emulate, so on arm64 an actor fails to start with
`runsc start`: exit status 128.

## Authentication and trust

`ateapi` rejects unauthenticated calls. The provider mints a ServiceAccount
token through the Kubernetes `TokenRequest` API, the same way `kubectl-ate`
authenticates, using the local kubeconfig or in-cluster credentials. The
serving CA is read from the cluster's `ClusterTrustBundle`.

Credentials are resolved inside the provider rather than through `${oc.env:...}`
interpolation in the config block: the per-instance config is written to disk,
so a token placed there would be written in plaintext, and it would not reach a
remote worker anyway.

The data path is different. The router's ingress port is unauthenticated today
and the in-actor server runs whatever it is sent, as root. Anyone who can reach
the router Service and knows `<atespace>/<actor>` can run commands in that
sandbox. Keep the router unexposed (a port-forward, or a NetworkPolicy that
admits only the Gym servers) until Substrate's ingress authentication lands.

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
      ready_timeout_s: 1200         # create() and resume() wait this long for a worker slot
      suspend_policy: pause_after_exec   # never | after_exec | pause_after_exec
      idle_before_suspend_s: 1.0
    exec:
      default_timeout_s: 1800
      concurrency: 32             # per provider instance; Gym makes one per sandbox
      capacity_retry_delay_s: 5
    probe:
      timeout_s: 150              # above the router's --parked-request-budget
```

### `suspend_policy`

`pause_after_exec` calls `PauseActor` once the actor has been idle for
`idle_before_suspend_s` after a step: the actor's worker is freed and its
snapshot stays on the node, so the next step resumes without a download but has
to land on the same node. The delay keeps back-to-back steps, such as an eval's
test runs, from each paying a checkpoint and a restore; an exec that arrives
during the delay cancels the pause, and an actor is never paused while a request
to it is in flight. An exec that arrives while a pause is already under way
waits for it to finish, because the router cannot resume an actor that is still
pausing.

`after_exec` calls `SuspendActor` instead: the snapshot is pushed to object
storage and any node can resume it. This pays a checkpoint upload per idle
period and a restore download at the start of the next step.

`never` leaves the actor running and is the control to measure against.

Whatever the policy, an actor whose worker is being evicted is suspended to
storage once idle, since a node-local pause would go away with the node: the
in-actor server reports the worker's `SIGTERM` on its next `/exec` reply, once,
and the provider suspends at the next idle moment (or in `pause()`). A paused
actor has no process to signal, so it gets no notice; its snapshot stays under
its node's `/var/lib/ate`, and Substrate does not move it when that node goes.
Until it does, suspend every Gym actor (`kubectl ate suspend`) before rolling a
worker pool or draining a node.

### Pause, resume and status

The provider implements `SupportsSandboxPauseResume`. `pause()` waits for
requests in flight to finish (a pause would checkpoint a running command and
cut its reply), calls `PauseActor` (or `SuspendActor` when the worker is being
evicted) and returns once the control plane reports the actor parked; it raises
if requests are still in flight after `connection.grpc_timeout_s`. `resume()`
sends a readiness probe through the router, which resumes the actor and parks
the probe until a worker has room. `status()` reports a paused, pausing,
suspended or suspending actor as `PAUSED`, a resuming or reverting one as
`STARTING`, a crashed one as `ERROR`, and a deleted one as `STOPPED`.

## What this provider does not honor

Declined rather than faked:

| Spec field | Why |
| --- | --- |
| `resources` | Actors are sized by their `ActorTemplate`, not per sandbox. |
| `ttl_s` | Substrate has no TTL; a paused actor already costs almost nothing. |
| `entrypoint` | The container command runs the sandbox server. |

`user` must be `root` (or unset). Every container in an actor shares one fixed
identity, so there is no user to switch to; anything else is refused rather than
silently ignored.

Commands run under `bash` when the image has it, else `sh`: `mini_swe_agent_2`
sources conda's `conda.sh`, which is not portable to dash. `spec.env` is applied
to each command, not to the container, so background processes a command leaves
behind do not see it. Uploads and downloads move one whole file per request
through memory; they are meant for seed files and results, not repositories.

## Regenerating the gRPC stubs

`_ateapi/` holds stubs generated from Substrate's `ateapi.proto`; the Substrate
commit they came from is recorded in `_ateapi/__init__.py`:

```bash
scripts/generate_ateapi_stubs.sh /path/to/substrate
```
