# vllm_model_with_guards

`vllm_model` plus opt-in guards that end a conversation from the model server. For models that keep calling tools after
the harness tells them to stop: the SFT checkpoints evaluated under opencode ignore its "maximum steps reached, answer
with text only" notice and loop on no-op commands, so the harness's own step limit never ends the run. Use
`vllm_model` for everything else.

| Config field | Effect |
|---|---|
| `guard_text_only_after_assistant_turns: N` | From N assistant turns on, every request is sent with `tool_choice: "none"`, so the model can only answer in text and the run ends. The engine is still called. |
| `guard_max_assistant_turns: N` | At N assistant turns, answer with a final text message and no tool call. The engine is not called. |
| `guard_identical_tool_calls: N` | When the last N assistant turns made the identical tool call (same tool, same arguments), answer as above. Minimum 2. |

All three default to unset (off); with none set this server behaves like `vllm_model`. The guards act in
`chat_completions`, which `/v1/chat/completions` and the converted `/v1/responses` path both use, so the server does
not support `is_responses_native: true`. A guard-ended turn is recorded as a local response
(`local_response_reason: harness_guard`).

- `guard_text_only_after_assistant_turns` needs `use_completions_api: false` (`/v1/completions` has no `tool_choice`) and
  no `tool_choice` entry in `sampling_overrides` other than `"none"` (a pin is applied after the guard). Both are refused
  at startup.
- A trailing harness step-limit notice (OpenCode's) is not counted as a model turn, and an OpenCode compaction-summary
  request goes to the engine unchanged.
- Enabling a guard changes how episodes end and can change evaluation results.

See "End a run from the model server" in the [vLLM guide](../../fern/versions/latest/pages/model-server/vllm.mdx).

Config: `configs/vllm_model_with_guards.yaml`.
