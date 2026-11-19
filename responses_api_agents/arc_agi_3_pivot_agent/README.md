# ARC-AGI-3 pivot agent

Overview of all the ARC-AGI-3 environments and agents, with a smoke run: [`resources_servers/arc_agi_3/OVERVIEW.md`](../../resources_servers/arc_agi_3/OVERVIEW.md).

Plays one pivot window of [`resources_servers/arc_agi_3_pivot`](../../resources_servers/arc_agi_3_pivot/README.md):

1. `/seed_session`: the server restores the pivot and returns the harness's request (messages, tools, chat template
   kwargs).
2. Until the window is done: one chat completion on the policy model server with the request's messages, then `/step`
   with the reply. The server returns the harness's next request, whose messages the next call sends.
3. `/verify`.

Every call extends the previous one: the server ends the window when the harness's next request does not (compaction,
a helper request), so training-token capture records one chain per rollout. Model calls use `url_path_for_run`, the
rollout's capture and sampling route.

## Chat format

- The Gym model server keeps OpenAI fields only: `top_k` goes in `metadata.extra_body`, the chat template kwargs in
  `metadata.chat_template_kwargs`, and tool calls are sent with `id`, `type` and `function.{name, arguments}` only
  (recorded pivots carry engine extras such as a streaming `index`).
- Assistant history reasoning is sent as a `<think>...</think>` content prefix; a reply's reasoning comes back in
  `reasoning_content` (or as a `<think>` prefix, which is split out).

## Configuration

| Field | Default | |
|---|---|---|
| `temperature`, `top_p`, `top_k`, `max_tokens` | 1.0, 0.95, 20, 12288 | the recorded pivot requests' sampling; a training framework's per-rollout values replace temperature, top_p and the output cap |
| `max_calls` | 64 | safety net; the window's own rules end it first |

A failed model call ends the window; the server masks the rollout (`model_request_error`). A pivot that could not be
restored goes straight to `/verify` (masked).

## Tests

```bash
pytest responses_api_agents/arc_agi_3_pivot_agent/tests
```
