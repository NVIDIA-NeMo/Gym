# Haystack Agent

A NeMo Gym agent harness that runs a serialized [Haystack](https://haystack.deepset.ai/)
`Pipeline` as its rollout loop. The pipeline contains a Haystack `Agent` whose
`chat_generator` is a `NeMoGymResponsesChatGenerator` — a Haystack `ChatGenerator` that calls a
native NeMo Gym model server's `/v1/responses` endpoint (resolved by `server_name`). Haystack's
`Agent` drives the repeated tool-calling loop. It can use Haystack-local tools and, through a
`ContextAwareMCPToolset`, environment tools exposed by a Gym Resources Server over MCP.

> [!NOTE]
> Dataset-row `responses_create_params.tools` are ignored. To use Resources Server tools,
> configure the server with `expose_tools_over_mcp: true` and include a
> `ContextAwareMCPToolset` in the Haystack pipeline. Gym seeds the Resources Server for each
> rollout and places its signed MCP session token in request-local context for tool calls.

The pipeline and MCP tool schemas are deserialized and warmed up **once at startup**. Each rollout
opens one token-authenticated MCP connection lazily on its first tool call and closes it when the
rollout ends. Concurrent rollouts remain isolated.

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
- mcp-haystack: Apache 2.0
