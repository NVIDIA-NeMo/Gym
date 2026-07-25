# Description

Serves Gym model calls through a [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) routing
proxy. Switchyard decides, per request, which upstream model should carry the work.

Because this is a model server rather than an agent integration, any benchmark Gym already supports
can be run against a router without changing harness code:

```bash
gym eval run --benchmark <name> --model-type switchyard_model
```

## Modes

**Attach (default).** You run the proxy; Gym points at it. Keeping the proxy outside Gym is what
lets an eval pin a specific Switchyard build and compare against scaled-evals runs of the same
commit.

```bash
switchyard --routing-profiles routes.yaml -- serve --port 4000

gym eval run --benchmark <name> --model-type switchyard_model \
  ++policy_model.responses_api_models.switchyard_model.switchyard_base_url=http://127.0.0.1:4000/v1
```

**Launch.** Gym starts the proxy itself and shuts it down at exit, for a single-command run.

```bash
gym eval run --benchmark <name> --model-type switchyard_model \
  ++policy_model.responses_api_models.switchyard_model.launch_proxy=true \
  ++policy_model.responses_api_models.switchyard_model.routing_profiles=/path/to/routes.yaml
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

Gym knows Switchyard; Switchyard does not know Gym. The `nemo-switchyard` dependency is scoped to
this package rather than Gym's core dependencies, so only runs that route through Switchyard pay
for it. Attach mode needs no dependency at all — the proxy is a plain OpenAI-compatible endpoint.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- nemo-switchyard: Apache 2.0
