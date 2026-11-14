# Opt-in vLLM tool media

This extends the stock `vllm_model` only at its converter hook. Images and videos
a tool returns stay structured parts of that `tool` message, with its
`tool_call_id`. It adds no extra user message, explanatory prompt, call sorter,
model loop or custom context-admission check. Other model components use the
generic shared converter.

The core schema (`nemo_gym.openai_utils`) permits tool-return video parts; that
alone does not enable video support on an arbitrary backend. Select this model
component only with a validated vLLM model/processor and tool-role media
support. Media preparation, pixel budgets and native context limits are separate
concerns.

CPU converter tests use invented data URLs; they do not replace a real GPU and
sandbox acceptance run. No dataset, checkpoint or credential is included.
