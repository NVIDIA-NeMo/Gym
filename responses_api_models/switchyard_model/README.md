# Description

Serves Gym model calls through a [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) routing
proxy. Switchyard decides, per request, which upstream model should carry the work.

Because this is a model server rather than an agent integration, any benchmark Gym already supports
can be run against a router without changing harness code:

```bash
gym eval run --benchmark <name> --model-type switchyard_model
```

## Gym hosts the proxy for you

Point `deployment` at a native Switchyard TOML deployment — the `llm_clients`, `targets`, and
`routes` the proxy serves. The proxy is Switchyard's native server, hosted inside this server's
process: it starts with the server, binds a loopback port, and stops when the server exits. You
never install or manage a separate process — `nemo-switchyard` is a dependency of this server, so
Gym installs its published wheel into the server's virtual environment on startup.

```bash
gym eval run --benchmark <name> --model-type switchyard_model \
  ++policy_model.responses_api_models.switchyard_model.deployment=/path/to/routes.toml
```

The deployment file is the routing condition the eval runs under — an explicit, diffable artifact.
Running the same benchmark against two deployment files is how routed runs are compared. A bad
deployment fails at startup with Switchyard's validation error, not as a timeout mid-run.

**Attaching instead.** Set `switchyard_base_url` to use a proxy you already run — worth doing when
an eval needs to pin a specific Switchyard build, or when several servers should share one
instance (routing strategies that use session or agent affinity are stateful, so replicas each
hosting their own proxy would not route the way a single deployed proxy does).

```bash
switchyard-server --config routes.toml --port 4000

gym eval run --benchmark <name> --model-type switchyard_model \
  ++policy_model.responses_api_models.switchyard_model.switchyard_base_url=http://127.0.0.1:4000/v1
```

## Rollout correlation

Gym's rollout-attempt id is forwarded as Switchyard's session id (`x-switchyard-session-id`), so
proxy-side routing decisions and costs — request logs, spans, session-affinity state, and the
aggregates on `/v1/stats` — can be joined back to the rollout that produced them.

The id arrives on the `/ng-rollout/<id>/` URL prefix, which agents add only when model-call
capture (`observability_enabled` plus a capture directory) or training-token capture is enabled.
The server checks this at startup: it warns when `forward_session_id` is merely the default, and
refuses to start when forwarding was requested explicitly but no capture is on.

For per-session snapshots on `/v1/routing/session-stats`, run `switchyard-server` yourself with
`--routing-log-file`, attach with `switchyard_base_url`, and add `proxy_x_session_id` to
`session_id_headers` — the name the 0.2.0 routing log keys on. Add names knowingly: Switchyard
forwards headers it does not recognize (and, at 0.2.0, the session header itself) to the upstream
provider.

Note that `switchyard_model` is the *route* name, not a provider model id — Switchyard maps it to a
concrete target. Which model actually served a call comes back on the response, and is recorded per
call when `observability_enabled` is set. Be aware that some agents overwrite the top-level
response `model` with the configured policy model name (`harbor_agent` does), so routing
attribution should be read from the model-call capture records rather than the rollout response.

## Routing-condition record

Set `condition_dir` (one directory per run) and the server records the routing condition the run
served under: `switchyard-condition.json` at startup — route, mode, deployment SHA-256 and
archived contents (inline `api_key` and all `extra_headers` values redacted), `nemo-switchyard`
version when hosting — and `switchyard-stats.json` at shutdown, wrapping the proxy's `/v1/stats`
(per-target requests, errors, tokens, latency, and classifier-side usage), which for a hosted
proxy would otherwise die with the process. The snapshot's `scope` field says what the counters
cover: a hosted proxy lives for exactly the run, while an attached proxy aggregates everything it
has served. An attached proxy is also a build Gym cannot identify, so its manifest takes identity
from the caller-supplied `proxy_provenance` mapping instead of the local wheel version. This is
what makes two routed runs comparable and one routed run reproducible after the fact; the
documented comparison workflow in the docs page builds on it.

## Dependency direction

Gym knows Switchyard; Switchyard does not know Gym.

`nemo-switchyard` is a dependency of this server only, not of Gym's core — so only runs that route
through Switchyard pay for it. No Gym code imports Switchyard on Gym's side of the boundary; this
server speaks OpenAI-compatible HTTP to the proxy and hosts the native server when asked to.

The dependency is pinned exactly (`nemo-switchyard==0.2.0`) because this server hosts the pinned
code in-process and depends on its wire behavior — the TOML deployment schema, the session header,
and the endpoint set are all version-coupled, and Switchyard is evolving quickly. Upgrades should
be deliberate, tested events; see the comment in `pyproject.toml`.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- nemo-switchyard: Apache 2.0
