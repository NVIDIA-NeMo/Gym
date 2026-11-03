# decision_engine_model

Model server for **typed decision engines**: servers that take a `state` and typed `questions` and return one
answer per question with a probability for every option, in one read (no text generation). Agents call it through
`/v1/responses` like any model server.

- **Request.** The last user message must be the JSON `{"state": ..., "questions": {...}}`. Other messages (for
  example an LLM answer-format instruction in the same dataset) are not sent to the engine.
- **Endpoints.** `endpoint: systemone` (default) posts `{"model", "state", "questions", **extra_body}` to
  `<base_url>/v1/systemone`. `endpoint: decisions` posts the OpenAI Decisions request shape to `/v1/decisions`
  (`state` as one `input_json` part, `choice` -> `choice`, `noul` -> `predicate`) and maps the answers back.
- **Reply.** One assistant message whose text is `{"di_engine": {"status", "response", "error", "wall_ms"}}`.
  `status` is `ok`, `unsupported` (HTTP 400/413/422 naming a capacity limit, as the Decision Index kit's `http`
  engine treats it) or `error` (other 4xx, or 5xx/429/transport errors after `max_attempts`). The server never
  raises for an engine failure, so one bad request does not stop a run; the resources server decides what it means.
- **Usage.** `input_tokens` is the engine's own count (`usage.input_tokens`; unknown when the engine does not report
  it). An engine reads one answer position per question and generates no text, so `output_tokens` is the number of
  answers.

```yaml
policy_model:
  responses_api_models:
    decision_engine_model:
      entrypoint: app.py
      base_url: ${oc.env:DECISION_ENGINE_BASE_URL,${decision_engine_base_url}}   # required, no default
      api_key: ${oc.env:DECISION_ENGINE_API_KEY,${decision_engine_api_key}}
      model: ${oc.env:DECISION_ENGINE_MODEL,${decision_engine_model_name}}
      endpoint: systemone
      extra_body: {}
      max_concurrent_requests: 32
```

The endpoint is required (`decision_engine_base_url: ???`): serve the engine locally and point it at `http://127.0.0.1:<port>`. The key is optional and redacted in `resolved_config.yaml`. `/v1/chat/completions` is not served.
