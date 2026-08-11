# Haystack Agent

A NeMo Gym agent harness that runs a serialized [Haystack](https://haystack.deepset.ai/)
`Pipeline` as its rollout loop. The pipeline contains a Haystack `Agent` whose
`chat_generator` is a `NeMoGymResponsesChatGenerator` — a Haystack `ChatGenerator` that calls a
native NeMo Gym model server's `/v1/responses` endpoint (resolved by `server_name`). Haystack's
`Agent` drives the repeated tool-calling loop. It can use Haystack-local tools, request-supplied
HTTP environment tools, and `ContextAwareMCPToolset` environment tools exposed over MCP.

> [!NOTE]
> Function tools in `responses_create_params.tools` become request-scoped Haystack tools and
> dispatch to the Resources Server's `POST /{tool_name}` routes. A configured MCP tool with the
> same name takes precedence; otherwise the request tool overrides a same-named local pipeline tool.

The pipeline is deserialized once at startup and warmed by Haystack on its first use. Each rollout
opens a token-authenticated MCP connection lazily on its first MCP tool call and closes it when the
rollout ends. Concurrent rollouts remain isolated.

## Layout

- `chat_generator.py` — `NeMoGymResponsesChatGenerator` + Haystack `ChatMessage` ⇄ Responses-API
  conversion helpers. Serializable, so it can be declared in a pipeline YAML by `type:`.
- `http_tool.py` — request-scoped direct HTTP Resources Server tool adapter.
- `app.py` — `HaystackAgent`; `responses()` loads `pipeline_yaml`, runs it, and returns the
  trajectory as a `NeMoGymResponse`. `run()` seeds the resources-server session and verifies.
- `configs/pipeline.yaml` — minimal Haystack `Agent` pipeline. Regenerate with Haystack's
  `Pipeline.dumps()`; add local `Tool`/`PipelineTool` instances as needed.
- `configs/haystack_agent.yaml` — Gym config wiring resources server, model server, and
  `pipeline_yaml` together.

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
