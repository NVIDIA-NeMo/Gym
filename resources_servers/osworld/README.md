# OSWorld Resources Server

This Gym resources server owns stateful OSWorld `DesktopEnv` sessions and
places them on a pool of remote Docker/QEMU workers. The agent and resources
server may run in one deployment today, but their process and HTTP contracts
remain separate:

```text
Gym agent -> OSWorld resources server -> remote Docker provider -> OSWorld VM
```

Each signed Gym session owns at most one live environment. The server selects
a worker by normalized load, makes step operations idempotent by
`operation_id`, evaluates before final cleanup, reaps idle sessions, and
persists non-secret placement metadata for diagnostics.

## Configuration

Start from `configs/osworld.yaml`. Static workers use `direct_http` and name an
SSH-controlled Docker host plus the host that exposes the VM ports. Dynamically
registered workers use `http_control`; the server discovers ready
`osworld_worker` services from the Gym head.

Production-like deployments should set:

- `OSWORLD_RESOURCES_TOKEN` when `require_auth=true`;
- `NEMO_GYM_REGISTRATION_TOKEN` when worker registration is protected;
- the same `deployment_id` in the Resources Server config and
  `OSWORLD_DEPLOYMENT_ID` on every worker;
- a private `proxy_config_file` only when proxy tasks are explicitly enabled.

Tokens, SSH keys, proxy credentials, and host-specific runtime state must not
be committed. `num_workers` must remain one because live `DesktopEnv` objects
are process-local.

The dynamically registered worker is Gym control-plane code, not an OSWorld
provider module. Start it on each Docker/KVM worker with:

```bash
python -m resources_servers.osworld.worker
```

It reads the `OSWORLD_WORKER_*`, `NEMO_GYM_HEAD_URL`, and
`NEMO_GYM_REGISTRATION_TOKEN` runtime settings, publishes the OSWorld service
ports, and registers capacity plus the non-secret VM resource contract with
the Gym head. Discovery and create requests are scoped by deployment ID; a
worker rejects requests for another deployment. The pinned OSWorld dependency
contains only the provider client and generic `DesktopEnv` contracts; it does
not depend on Gym's worker registry.

## Tests and example data

The unit tests use an in-memory fake environment and never contact Docker,
OSWorld VMs, SSH hosts, or the worker registry:

```bash
PYTHONPATH="$PWD" python -m pytest -q resources_servers/osworld/tests
```

`data/example.jsonl` and `data/example_rollouts.jsonl` mirror the five
committed OSWorld benchmark smoke fixtures. They exist for Gym's server data
contract; running the resources-server unit tests does not replay those tasks.

## Licensing

- Server code: Apache-2.0.
- OSWorld task fixtures and dependency: see the upstream OSWorld repository.
