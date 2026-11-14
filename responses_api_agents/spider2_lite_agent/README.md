# Description

Agent for the [Spider 2.0-Lite](../../resources_servers/spider2_lite/README.md) text-to-SQL
resources server. It is `simple_agent` with one extra behavior: when the model server rejects a
request because the prompt exceeds the model's context window, the agent returns an empty,
`failed` response instead of raising. The resources server then scores that row as 0, so one
oversized prompt does not abort a long evaluation.

This matters mostly for BigQuery tasks, where the schema in the prompt can be very large.

Only HTTP 400 responses whose body names an input-size limit are handled (for example
`context_length_exceeded`, `maximum context length`, or `prompt is too long`). Every other
upstream error, including a 400 about anything else, still propagates unchanged. To cover a new
provider's wording, extend `_CONTEXT_ERROR_MARKERS` in `app.py`.

Use it through the Spider 2.0-Lite config, which already wires the agent to the resources server:

```bash
gym env test +entrypoint=responses_api_agents/spider2_lite_agent
```

See `resources_servers/spider2_lite/configs/spider2_lite.yaml` and
`benchmarks/spider2_lite/README.md` for running the benchmark end to end.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
