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
and stopped when it exits. You never manage it.

```bash
pip install 'nemo-switchyard[server]'   # launching drives the `switchyard` CLI

gym eval run --benchmark <name> --model-type switchyard_model \
  ++policy_model.responses_api_models.switchyard_model.routing_profiles=/path/to/routes.yaml
```

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

This server takes no dependency on `nemo-switchyard` and never imports it — it speaks
OpenAI-compatible HTTP to the proxy, and launch mode drives the CLI. Installing Switchyard is left
to whoever wants launch mode, which also lets an eval pin the exact Switchyard build its numbers
are reported against rather than inheriting Gym's.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0

Optional, installed separately for `launch_proxy` mode
- nemo-switchyard: Apache 2.0
