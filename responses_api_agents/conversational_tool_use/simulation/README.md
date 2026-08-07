# Conversational Tool-Use Agent

This custom Responses API agent runs full simulated customer-service conversations.

The complete workflow and shared design documents live with the simulation resource server:

- [Component overview](../../../resources_servers/conversational_tool_use_simulation/README.md)
- [Domain generation](../domain_generation/README.md)
- [Policy/tool generation](../policy_tool_generation/README.md)
- [Scenario generation](../scenario_generation/README.md)
- [Rollout behavior](../../../resources_servers/conversational_tool_use_simulation/docs/rollout.md)

## Prepare Assets

Prepare the shared assets before running this agent:

```bash
python -m resources_servers.conversational_tool_use_simulation.prepare
```

This downloads all runtime prompts and policy/tool golden references from
[`nvidia/NeMo-Gym-Conversational-Tool-Use-Assets`](https://huggingface.co/datasets/nvidia/NeMo-Gym-Conversational-Tool-Use-Assets).
Add `--include-prompt-history` for the optional prompt history. JSON schemas and example JSONL files remain in Git.

## Configuration

[`configs/conversational_tool_use_simulation.yaml`](configs/conversational_tool_use_simulation.yaml) registers the
`conversational_tool_use_agent` instance and expects a resource-server instance named
`conversational_tool_use_simulation`. For a complete runnable stack, use the
[resource-server configuration](../../../resources_servers/conversational_tool_use_simulation/configs/conversational_tool_use_simulation.yaml),
which registers both instances.

Agent controls in the checked-in config:

| Setting | Default | Meaning |
|---|---:|---|
| `resources_server` | `conversational_tool_use_simulation` | Resource-server instance used for session state, user and tool simulation, and verification |
| `model_server` | `policy_model` | Model-server instance used for policy responses |
| `max_agent_steps` | `50` | Maximum policy-model responses per rollout; simulator and tool messages do not count |
| `seed_first_user_message` | `true` | Ask the resource server for the first customer turn when no conversation is prefilled |

Policy request controls such as `temperature`, `top_p`, `max_output_tokens`, `tool_choice`, and
`parallel_tool_calls` belong to each row's `responses_create_params`. Omitted sampling fields defer to the configured
policy model server; `parallel_tool_calls` defaults to `true`.

## Initial User Turn and Prefilled History

Conversation rows may omit the first customer message or provide it explicitly:

- With no prefilled user message, the default `seed_first_user_message=true` flow asks the resource server to generate
  one before the first policy-model call.
- A row may put a user message directly in `responses_create_params.input`.
- A row may instead set top-level `initial_user_message`; the agent copies it into `responses_create_params.input`
  before seeding. If both forms are present, their first user messages must match.
- A row may provide a longer user, assistant, function-call, and function-call-output history and resume from the next
  actor.

When the resource server generates the first user turn, `/v1/responses` initially returns it as a leading output item.
Before `/run` writes the scored rollout row, the agent moves that item into `responses_create_params.input` and removes
it from `response.output`. The model-generated continuation therefore begins with the first assistant reasoning,
message, or function call. See [rollout behavior](../../../resources_servers/conversational_tool_use_simulation/docs/rollout.md#initial-user-turn-and-prefilled-history)
for the trajectory-side counters and complete contract.

It differs from `simple_agent` in one critical way: when the policy model emits a normal assistant message, the rollout
does not stop. The agent records the assistant message with the simulation resource server, asks the server for
`/next_user_message`, appends that user turn, and continues until a stop marker, transfer marker, max step limit,
incomplete response, or verification failure.

`max_agent_steps` counts generated policy-model responses, not individual user/tool trajectory messages, and defaults
to `50`. At the final step, a text response is followed by one final user-simulator turn so the user can emit a stop
or transfer marker. A final response containing tool calls is recorded and terminated without executing those tools or
adding dummy function-call outputs.

When `parallel_tool_calls=false`, only the first emitted function call is selected. Any additional function-call
output items from a nonconforming provider response are removed before the response is stored, sent to the resource
server, or included in the next policy-model request. Non-function output items, including assistant content and
reasoning, are preserved.

Before the rollout starts, the agent checks that the materialized policy prompt and model-visible tool definitions are
the canonical renderings of the top-level policy and simulator tools. A mismatched row is rejected instead of letting
the policy model and verifier see different contracts. If a provider emits assistant text and function calls in one
response, the agent records the selected items in provider order before executing any call so both transcript views
remain aligned. Parallel tool results are appended only after all selected response items.

For the simulator trajectory, each selected output item carries its source `response_id` and
`response_output_index`. The resource server stores the complete raw policy response only once, on the first selected
item, rather than repeating it on every decomposed message.

Tool results are returned to the policy as the raw tool-simulator text. Resource-server diagnostics such as
`schema_valid`, `error`, `should_continue`, and `terminal_state` control the rollout and remain available in the
internal trajectory, but are not embedded in the policy-visible tool message.

The resource server owns the user simulator and tool simulator. The agent owns only the policy-model loop and the transcript that is returned as the Responses API output.

For Gym rollouts, `/run` canonicalizes the returned row the same way tau2 does: leading user/system/developer items
before the first assistant, reasoning, or function-call item are placed in `responses_create_params.input`, and the
remaining generated continuation stays in `response.output`. The resource server's internal `ConversationMessage`
trajectory is not stored in `response.output`; it is returned from `/verify` as the sidecar `result` dict.

Transient infrastructure failures from session seeding, rollout generation, or verification follow Gym's failure
sidecar contract. The agent returns `_ng_failure_class: transient`, so the collector writes the attempt to
`<output_stem>_failures.jsonl` and a resumed run can retry it up to `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`. These failures
do not enter the scored rollout JSONL. Semantic policy-agent failures remain ordinary verified rollouts and are not
retried through this mechanism.

Gym's rollout prefix is propagated through the agent self-call and downstream policy-model call. The seed request also
passes the same correlation ID to the resource server, which applies it to simulator and judge calls.

If a seeded `/run` exits before successful verification, the agent makes an idempotent `/discard_session` call.
Rollout errors, verification errors, and cancelled attempts therefore cannot leave abandoned session state behind.

## Tests

```bash
gym env test \
  +entrypoint=responses_api_agents/conversational_tool_use/simulation \
  +should_validate_data=false
```
