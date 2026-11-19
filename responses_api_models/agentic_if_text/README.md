# Recorded single-turn text model adapter

The default benchmark profile now uses Gym's standard `vllm_model`; see
[`config.yaml`](../../benchmarks/agentic_if_realworld/config.yaml). This adapter
remains available through `raw_receipt.yaml` for the earlier transport.

This adapter serves the real-world IF benchmark's frozen adaptive delivery
protocol. It sends the exact one-user-message wrapper, requests streamed usage,
records bounded transport attempts privately, and rejects native actions and
incomplete output before accepting a selected answer. It preserves provider token
counts; absent counts remain unknown. Gym's standard model-call capture records
its Responses interface when enabled by the run.

`reasoning_effort: max` forwards `chat_template_kwargs.reasoning_effort` to a
compatible vLLM endpoint. The default `null` sends no effort override. The driver
records the chosen setting and rejects resume after it changes; the judge's
reasoning setting is independent.

Use the [benchmark runbook](../../benchmarks/agentic_if_realworld/README.md).
No real task data, model output, credentials or captures belong in this directory.
