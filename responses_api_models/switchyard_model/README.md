# Description

Serves Gym model calls through a [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) routing
proxy. Switchyard decides, per request, which upstream model should carry the work.

Because this is a model server rather than an agent integration, any benchmark Gym already supports
can be run against a router without changing harness code:

```bash
gym eval run --benchmark <name> --model-type switchyard_model
```

## Gym runs the proxy for you

Point `routing_profiles` at your Switchyard routing config; the proxy is started with this server
and stopped when it exits. You never install or manage it — `nemo-switchyard` is a dependency of
this server, so Gym installs it into the server's virtual environment on startup.

```bash
gym eval run --benchmark <name> --model-type switchyard_model \
  ++policy_model.responses_api_models.switchyard_model.routing_profiles=/path/to/routes.yaml
```

### The first startup builds Switchyard from source

While `nemo-switchyard` is pinned to a git commit rather than a release, installing it compiles
Switchyard's Rust extension instead of using its published wheels. This needs no setup on your
part — maturin bootstraps a Rust toolchain if the machine has none — but it does mean the first
`gym env start` for this server spends an extra minute or two building, and that the build reaches
`static.rust-lang.org` and `crates.io` on top of PyPI and GitHub. Behind a restrictive egress
policy those two hosts are the ones to allow.

Neither applies once the pin moves to a released wheel, and neither applies to the self-managed
mode below, which installs nothing.

**Attaching instead.** Set `switchyard_base_url` to use a proxy you already run — worth doing when
an eval needs to pin a specific Switchyard build, or when several servers should share one
instance (routing strategies that use session or agent affinity are stateful, so replicas each
running their own proxy would not route the way a single deployed proxy does).

```bash
switchyard --routing-profiles routes.yaml -- serve --port 4000

gym eval run --benchmark <name> --model-type switchyard_model \
  ++policy_model.responses_api_models.switchyard_model.switchyard_base_url=http://127.0.0.1:4000/v1
```

## Rollout correlation

Gym's rollout-attempt id is forwarded as Switchyard's session id (`proxy_x_session_id`), so
proxy-side routing decisions and costs can be joined back to the rollout that produced them.

Note that `switchyard_model` is the *route* name, not a provider model id — Switchyard maps it to a
concrete target. Which model actually served a call comes back on the response, and is recorded per
call when `observability_enabled` is set. Be aware that some agents overwrite the top-level
response `model` with the configured policy model name (`harbor_agent` does), so routing
attribution should be read from the model-call capture records rather than the rollout response.

## Dependency direction

Gym knows Switchyard; Switchyard does not know Gym.

`nemo-switchyard` is a dependency of this server only, not of Gym's core — so only runs that route
through Switchyard pay for it. No Gym code imports Switchyard; this server speaks
OpenAI-compatible HTTP to the proxy and drives the CLI to launch one.

The dependency is pinned to `==0.1.0` with a `[tool.uv] override-dependencies` entry relaxing that
release's stale `openai>=2.34.0` floor. See the comments in `pyproject.toml` — both should be
removed once a Switchyard release carrying the widened floor ships.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- nemo-switchyard: Apache 2.0
