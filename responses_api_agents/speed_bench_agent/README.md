# speed_bench_agent

Multi-turn fixed-replay agent for SPEED-Bench.

## Why a new agent

NeMo-Gym's `simple_agent` performs at most one model call per task: given
`responses_create_params.input = [user_1, user_2, ...]`, it sends the
entire message list to the model in a single `/v1/responses` call.

NeMo-Skills' `SpecdecGenerationTask.process_single_datapoint` replays the
turn list one turn at a time:

1. Send `[user_1]`, receive `assistant_1`.
2. Send `[user_1, assistant_1, user_2]`, receive `assistant_2`.
3. …

This matters for spec-decode parity: the per-turn prompt distribution
the model sees affects the speculative-decoding acceptance rate.
Collapsing all turns into a single call would give a different prompt
mixture and therefore different headline AL / AR numbers.

`speed_bench_agent` reproduces Skills' replay behaviour. Single-turn rows
collapse to a single call (same shape as `simple_agent`).

## Behaviour

- Reads `body.input` as the entire dialogue.
- Treats every *trailing* user message as a turn boundary (so a user can
  optionally pre-seed assistant turns; only the trailing user messages
  trigger fresh model calls).
- For each trailing user turn, sends the running dialogue (preamble +
  prior turns + accumulated assistants + this turn) to the model, gets
  the assistant reply, appends it.
- Aggregates `usage.input_tokens` / `output_tokens` / `total_tokens`
  across all turn calls into a single `usage` on the returned response.

The returned `NeMoGymResponse.output` contains every assistant
output-message item across all turns (concatenated in order). The
`speed_bench` resources server only reads `response.usage.output_tokens`
to record per-task token counts.

## Configuration

```yaml
speed_bench_simple_agent:
  responses_api_agents:
    speed_bench_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: speed_bench_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
```
