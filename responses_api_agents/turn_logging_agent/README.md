# Turn Logging Agent

A drop-in variant of `simple_agent` that records **per-turn telemetry** while running the
identical agent loop (same max-steps, tool-calling, and termination behavior — validated
live against `simple_agent` with no outcome drift; see
`resources_servers/enterpriseops_gym/PARITY.md` §3, run C).

## Why

The stock `simple_agent` aggregates usage across the whole rollout, which discards
per-turn timings and zeroes `cached_tokens` / reasoning-token details. Eval workflows that
analyze turn-level behavior (latency breakdowns, prefix-cache hit rates, tool-call timing)
need those numbers preserved per model call.

## What it records

For every agent turn: timestamp, wall-clock duration, input/output/**cached**/reasoning
token counts, tool names called, and the output-slice indices into the final response.
Turn records are correlated to model calls via a private Responses `metadata` key
(`_turn_log_id`, stripped before the request reaches the model server) and attached as
`turns` on the verify response, so they land in collected rollout rows.

`resources_servers/enterpriseops_gym/export_eval_telemetry.py` shows a downstream
consumer: it flattens `turns` into a per-turn JSONL telemetry schema.

## Usage

Wire it like `simple_agent` — it is environment-agnostic. Example config
(`resources_servers/enterpriseops_gym/configs/enterpriseops_gym_turnlog.yaml` adds it
alongside a standard stack):

```yaml
my_turn_logging_agent:
  responses_api_agents:
    turn_logging_agent:
      entrypoint: app.py
      max_steps: 50
      resources_server:
        type: resources_servers
        name: my_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: example
        type: example
        jsonl_fpath: resources_servers/my_resources_server/data/example.jsonl
```

Collect rollouts with `+agent_name=my_turn_logging_agent`; each output row's verify
response carries the `turns` list.

## Tests

```bash
gym env test +entrypoint=responses_api_agents/turn_logging_agent
```
