# Substrate sandbox provider

Runs sandboxes as [Substrate](https://github.com/agent-substrate/substrate)
actors. Substrate can suspend an actor to storage and resume it elsewhere, so an
idle sandbox holds no worker; for a benchmark run that spends most of its
wall-clock waiting on a policy model, that is the reason to use this backend.

## How it works

```
provider ── gRPC ──────> ateapi          Create / Resume / Suspend / Delete actor
         └─ HTTP ──────> atenet router ─> ate-sandbox-agent (inside the actor)
```

Control operations go to `ateapi`. Commands and file transfers go to an agent
inside the actor, addressed through the `atenet` router by the actor's mesh name
(`<actor>.<atespace>.actors.resources.substrate.ate.dev`). The router resumes a
suspended actor before routing to it, so nothing has to resume it explicitly.

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
built ahead of a run rather than on demand. Substrate ships a generator:

```bash
# in a Substrate checkout
hack/swebench-templates.sh django__django-11099 astropy__astropy-12907
```

Each template pairs the benchmark image, untouched, with `ate-sandbox-agent`
delivered as an image volume and set as the container command. The provider
derives a template's name from the image it is asked for, so the two sides agree
without a shared lookup table. That derivation is pinned by a unit test; if it
changes, the generator must change with it.

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
authenticates, using the local kubeconfig or in-cluster credentials. Install the
cluster with `--ateapi-client-auth=token`.

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
      atespace: swebench            # actors are created here
      template_namespace: ate-swebench
    create:
      suspend_policy: after_exec    # never | after_exec
    exec:
      default_timeout_s: 1800
      concurrency: 32
```

### `suspend_policy`

`after_exec` suspends the actor once each step finishes, freeing its worker
while the policy model thinks. The next step goes through the router, which
resumes the actor before forwarding.

`SuspendActor` is synchronous: it returns only once the snapshot has been pushed
to storage and the worker freed. So `after_exec` pays a checkpoint at the end of
every step and a restore at the start of the next, both on the critical path,
including across gaps too short to be worth reclaiming. `never` is the control
to measure it against.

## What this provider does not honor

Declined rather than faked:

| Spec field | Why |
| --- | --- |
| `resources` | Actors are sized by their `WorkerPool`, not per sandbox. |
| `ttl_s` | Substrate has no TTL; a suspended actor already costs almost nothing. |
| `entrypoint` | The container command runs the sandbox agent. |

`user` must be `root` (or unset). Every container in an actor shares one fixed
identity, so there is no user to switch to; anything else is refused rather than
silently ignored.

Commands run under `bash`, not `sh`: `mini_swe_agent_2` sources conda's
`conda.sh`, which is not portable to dash. Every SWE-bench image ships bash, but
a minimal image might not.
