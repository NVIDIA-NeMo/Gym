# Haystack Agent

A NeMo Gym agent harness that runs a serialized [Haystack](https://haystack.deepset.ai/)
`Pipeline` as its rollout loop. The pipeline contains a Haystack `Agent` whose
`chat_generator` is a `NeMoGymResponsesChatGenerator` — a Haystack `ChatGenerator` that calls a
native NeMo Gym model server's `/v1/responses` endpoint (resolved by `server_name`). Haystack's
`Agent` drives the repeated tool-calling loop; tools live entirely on the Haystack side.

> [!WARNING]
> **Env-side (resources-server) tools are not supported yet.** Tools must be defined in the
> Haystack pipeline as executable Haystack `Tool`/`Toolset` objects. Tools served by the resources
> server — and any `tools` passed in the request body — are ignored: the request's `tools` field is
> not forwarded to the model, and the agent cannot invoke tools it doesn't hold locally. If your
> environment relies on resources-server tools, this harness won't run them.

The pipeline is deserialized and warmed up **once at startup** (in `model_post_init`) and shared
across all requests, so expensive component/tool initialization is paid a single time rather than on
every rollout. Concurrent rollouts are safe: Haystack's `Agent`/`Pipeline` keep all per-run state in
locals, and the generator's per-request session state (cookies, usage, last response) is isolated
via `contextvars`.

## Layout

- `chat_generator.py` — `NeMoGymResponsesChatGenerator` + Haystack `ChatMessage` ⇄ Responses-API
  conversion helpers. Serializable, so it can be declared in a pipeline YAML by `type:`.
- `app.py` — `HaystackAgent`; `responses()` loads `pipeline_yaml`, runs it, and returns the
  trajectory as a `NeMoGymResponse`. `run()` seeds the resources-server session and verifies.
- `configs/pipeline.yaml` — example Haystack `Agent` pipeline (one trivial `get_weather` tool).
  Regenerate with Haystack's `Pipeline.dumps()`; swap in your own `Tool`/`PipelineTool`.
- `configs/haystack_agent.yaml` — Gym config wiring resources server, model server, and
  `pipeline_yaml` together.
- `example_tools.py` — example Haystack-side tool referenced by `configs/pipeline.yaml`.

## Run

```bash
gym env start \
    --resources-server <your_resources_server> \
    --responses-api-agent haystack_agent \
    --model-type vllm_model
```

## Test

```bash
gym env test --responses-api-agent haystack_agent
```

## Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- haystack-ai: Apache 2.0
