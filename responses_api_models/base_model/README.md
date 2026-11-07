# base_model

Model server for raw **base** (pretrained, non-instruct) checkpoints. A base model has no chat template and no tool
calling, so the server flattens the conversation into a `Role: content` transcript, sends it to the upstream's
`/v1/completions` endpoint (for example vLLM), and parses the action back out of the generated text.

Three strings shape every prompt: a **steer** after the transcript that names the action format and the repository
root, a **primer** that opens the model's turn, and a **prefill** that pre-opens the action so the model continues it
instead of writing prose. Their defaults depend on the wire:

| `wire` | the model writes | returned as |
|---|---|---|
| `backticks` (default) | one bash command in a ```` ```mswea_bash_command ```` fence (mini-swe-agent's text protocol) | assistant text |
| `function_calling` | a JSON array of `bash` tool calls, following the schema in `tool_definitions_json` | `tool_calls` on the assistant message |

The wire must match the one the trajectories were captured on, not the benchmark: crossed, every turn is a well-formed
action the agent cannot execute. On `function_calling` no stop list is sent, because tool-call arguments routinely
contain text such as `\nUser:`. The server also cuts each completion at the first stop string itself: some servers
match stop strings on token boundaries and miss one whose newline shares a token with the text before it, letting the
model run on into turns it invents.

The default steers name `/testbed` (backticks, SWE-bench images) and `/app` (function calling, DeepSWE images) as the
repository root. Override `steer` for any other layout: a steer naming the wrong root produces well-formed commands
against a path that does not exist, which scores as a quiet 0.

## Configuration

| field | default | notes |
|---|---|---|
| `openai_base_url`, `openai_model` | required | the upstream endpoint and the model `id` it serves (read it from `/v1/models`) |
| `openai_api_key` | required | self-hosted vLLM ignores it |
| `wire` | `backticks` | sets the defaults of `steer`, `primer`, `prefill` and `stop` |
| `steer`, `primer`, `prefill`, `stop` | per wire | each can be overridden; `stop: []` sends no stop list |
| `tool_definitions_json` | none | the tool schema as a JSON string, rendered verbatim into the prompt |
| `max_tool_calls_per_turn` | 3 | calls beyond this in one turn are dropped |
| `max_tokens`, `temperature`, `top_p` | 1024, 0.6, none | sampling |

`configs/base_model.yaml` defines `policy_model`, which reads `policy_base_url` and `policy_model_name` from
`env.yaml`. The prefix pass@K benchmark (`benchmarks/prefix_pass_k/`) serves checkpoints through this server on both
wires.

## Tests

```bash
gym env test +entrypoint=responses_api_models/base_model
```

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
