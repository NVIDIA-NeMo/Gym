# vibench_agent

The build half of the [ViBench](https://github.com/ViBench/vibench-public) environment. It
takes a product requirements document, builds a working web application in its own sandbox,
and copies the result out as a tarball for
[`resources_servers/vibench`](../../resources_servers/vibench/README.md) to stand up and grade.

Everything about installing and driving the OpenCode harness is inherited from
`opencode_sandboxed_agent`. Only two things are overridden: how the sandbox is acquired, and
how the finished app is harvested.

## Shape

```
POST /seed_session   PRD text + asset dirs (no sandbox handle)
create sandbox       ViBench's app-bench-base image, WORKDIR /app
stage PRD + assets   via SandboxSpec.files, before the harness starts
run OpenCode         inherited from OpenCodeSandboxedAgent
harvest /app         tarball written into the shared artifact_dir
POST /verify         resources server unpacks and grades it
```

## Why the agent owns the sandbox

The sandbox is never shared. Reaching into the agent's box the way `swebench` does requires
`serialize()`/`connect()`, which only the OpenSandbox provider implements — so that shape
cannot run on Docker, Apptainer or enroot at all. Handing the built app over as a tarball
keeps every provider viable. See [#2082](https://github.com/NVIDIA-NeMo/Gym/issues/2082) for
the design discussion.

`artifact_dir` is a filesystem path shared with the resources server. That is not a new
constraint: grading already shells into a local Docker daemon, so both processes are on one
host regardless.

Harvesting skips symlinks that resolve outside `/app`. The tarball comes from a box the model
controlled, so the verifier refuses escaping links — harvesting one would make it reject the
whole archive.

## Networking

Use `configs/docker.yaml` here instead of the stock
`nemo_gym/sandbox/providers/docker/configs/docker.yaml`. Two things differ, and both are
load-bearing:

| Stock behaviour | Problem | What this config does |
| --- | --- | --- |
| 180s exec timeout | kills long `npm`/`pip` installs mid-build | raises the timeout |
| harness told the model is at `http://127.0.0.1:<port>` | inside a bridged container that is *the container itself*, so the harness makes **zero** LLM calls and exports an empty app | keeps the default bridge, adds `host.docker.internal` via host-gateway, and the agent rewrites loopback model URLs to it |

**Do not "fix" the second row with `network: host`.** That puts model-written code on the host
network namespace — the same class of harness breakout as mounting the Docker socket.

Only `policy_model` binds `0.0.0.0`, because on Linux Docker Engine a `127.0.0.1` bind is not
visible from the bridge gateway. The resources server, agent and head server stay on loopback,
since grader credentials live in those processes.

> **Exposure:** `0.0.0.0` publishes the model server on *every* host interface, not just the
> Docker bridge — including the run's token-capture path (`/ng-rollout/<id>`), which accepts a
> dummy API key. Run this on a single-tenant box, or firewall the port.

OpenSandbox sandboxes have their own address and do not need this file; pass
`sandbox_model_base_url` on the agent instead.

## Run

The second `--config` is required — `sandbox_provider: sandbox` is a reference the provider
config binds, so without it startup fails with *"Sandbox provider reference 'sandbox' is not
defined in the merged config"*. Swapping that one path moves you to another provider without
editing anything else.

```bash
gym env start \
    --config resources_servers/vibench/configs/vibench.yaml \
    --config responses_api_agents/vibench_agent/configs/docker.yaml \
    --model-type openai_model

gym eval run --no-serve \
    --agent vibench_opencode_agent \
    --input resources_servers/vibench/data/example.jsonl \
    --output results/vibench_rollouts.jsonl \
    --limit 1 \
    --num-repeats 1
```

Start with `--limit 1`. One rollout builds an app and then runs a full compose stack per test
plan; wall-clock is tens of minutes.

Setup, task-row generation, reward definition and grading live in the
[resources server README](../../resources_servers/vibench/README.md).

## Tests

```bash
pytest responses_api_agents/vibench_agent/tests/
```

## Licensing

ViBench is Apache 2.0. This agent contains no ViBench data of its own; PRDs, test plans and
the runner harness come from a local ViBench checkout.
