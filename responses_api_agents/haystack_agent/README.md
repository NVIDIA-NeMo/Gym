# Haystack Agent

A NeMo Gym agent harness that runs a serialized [Haystack](https://haystack.deepset.ai/)
`Pipeline` as its rollout loop. The pipeline contains a Haystack `Agent` whose
`chat_generator` is a `NeMoGymResponsesChatGenerator` — a Haystack `ChatGenerator` that calls a
native NeMo Gym model server's `/v1/responses` endpoint (resolved by `server_name`). Haystack's
`Agent` drives the repeated tool-calling loop. It can use Haystack-local tools, request-supplied
HTTP environment tools, and `ContextAwareMCPToolset` environment tools exposed over MCP.

> [!NOTE]
> Function tools in `responses_create_params.tools` become request-scoped Haystack tools and
> dispatch to the Resources Server's `POST /{tool_name}` routes. An MCP tool available and selected
> for the rollout takes precedence over a same-named request tool. Collisions with local pipeline
> tools raise an error; rename or remove the conflicting definitions.

The pipeline is deserialized once at startup and warmed by Haystack on its first use. Each rollout
opens a token-authenticated MCP connection to discover its available tools before inference, reuses
that connection for tool calls, and closes it when the rollout ends. Discovery results belong only
to that rollout, including when the pipeline configures `eager_connect: true`.
MCP rollout copies support Haystack's standard toolset collection APIs. Local toolsets retain their
`spawn()` lifecycle and can expose new tools during the run; Haystack checks names again each turn
and rejects collisions with existing tools.

MCP `tool_names: null` includes every tool returned by authenticated discovery. An explicit list
selects only matching available tools. If it excludes any discovered tools, the agent raises an
error unless its config sets `allow_mcp_tool_filtering: true`. This selection applies only to MCP;
HTTP tools remain independently supplied by the request.

## Pipeline and request contract

The configured component must be a Haystack `Agent` backed by `NeMoGymResponsesChatGenerator`.
`Agent.system_prompt` is supported, but `Agent.user_prompt` must be unset: the incoming Responses
request already supplies the complete user context, and Haystack appends `user_prompt` after that
context, making it impossible to distinguish from generated output when the rollout is reconstructed.

The request's `input` is converted to Haystack messages and request `tools` become request-scoped
HTTP tools. Other explicitly supplied generation settings, including `temperature`, `top_p`,
`max_output_tokens`, and `tool_choice`, are forwarded to every model call. They override the
generator's static `generation_kwargs` from the pipeline YAML. `instructions` is ignored in favor
of the pipeline's `system_prompt`; streaming is unsupported.

The harness is text-only. It rejects image, file, refusal, and other non-text content parts rather
than silently changing the trajectory. Generated token IDs, logprobs, routed-expert metadata, and
per-call usage are retained in `ChatMessage.meta` while Haystack executes tools, then restored in
the final Responses trajectory. Each model turn's token stream is attached once to that turn's
terminal function-call or assistant-message output item.

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
