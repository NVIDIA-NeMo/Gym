# Synthetic Tool-Use Agent

This custom Responses API agent runs full synthetic customer-service conversations.

The complete workflow and shared design documents are indexed with the synthetic tool-use pipeline:

- [Complete synthetic tool-use pipeline](../../resources_servers/synthetic_tool_use/README.md)
- [Generation workflow](../../resources_servers/synthetic_tool_use/docs/generation.md)
- [Rollout behavior](../../resources_servers/synthetic_tool_use/docs/rollout.md)

It differs from `simple_agent` in one critical way: when the policy model emits a normal assistant message, the rollout does not stop. The agent records the assistant message with the synthetic simulation resource server, asks the server for `/next_user_message`, appends that user turn, and continues until a stop marker, transfer marker, max step limit, incomplete response, or verification failure.

`max_agent_steps` counts generated policy-model responses, not individual user/tool trajectory messages, and defaults
to `50`. At the final step, a text response is followed by one final user-simulator turn so the user can emit a stop
or transfer marker. A final response containing tool calls is recorded and terminated without executing those tools or
adding dummy function-call outputs.

When `parallel_tool_calls=false`, only the first emitted function call is selected. Any additional function-call
output items from a nonconforming provider response are removed before the response is stored, sent to the resource
server, or included in the next policy-model request. Non-function output items, including assistant content and
reasoning, are preserved.

Tool results are returned to the policy as the raw tool-simulator text. Resource-server diagnostics such as
`schema_valid`, `error`, `should_continue`, and `terminal_state` control the rollout and remain available in the
internal trajectory, but are not embedded in the policy-visible tool message.

The resource server owns the user simulator and tool simulator. The agent owns only the policy-model loop and the transcript that is returned as the Responses API output.

For Gym rollouts, `/run` canonicalizes the returned row the same way tau2 does: leading user/system/developer items
before the first assistant, reasoning, or function-call item are placed in `responses_create_params.input`, and the
remaining generated continuation stays in `response.output`. The resource server's internal `SyntheticMessage`
trajectory is not stored in `response.output`; it is returned from `/verify` as the sidecar `result` dict.

Transient infrastructure failures from session seeding, rollout generation, or verification follow Gym's failure
sidecar contract. The agent returns `_ng_failure_class: transient`, so the collector writes the attempt to
`<output_stem>_failures.jsonl` and a resumed run can retry it up to `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`. These failures
do not enter the scored rollout JSONL. Semantic policy-agent failures remain ordinary verified rollouts and are not
retried through this mechanism.
