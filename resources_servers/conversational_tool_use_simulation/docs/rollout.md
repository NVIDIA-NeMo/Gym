# Conversational Tool-Use Rollouts

One rollout joins a generated policy, tool set, and customer scenario with four runtime roles:

- the policy model acts as the customer-service agent
- the user simulator produces customer turns
- the tool simulator produces results for generated tool schemas
- the judge model scores completed trajectories

The policy loop runs in `responses_api_agents/conversational_tool_use_agent`. Session state, simulation, validation, and
verification run in `resources_servers/conversational_tool_use_simulation`.

## Session Setup

`/seed_session` loads the policy, tools, scenario, generation profile, source lineage, and Gym rollout-correlation ID
for one rollout. Model-server references, retry settings, and verification settings come from the environment config.
Most rows start without conversation history, so the resource server generates the first customer turn. Rows may also
contain prefilled Responses history; supported user, assistant, function-call, and function-call-output items are
hydrated into session state before generation continues.

`/session_tools` exposes the generated tools as Responses API function tools for the policy model.

## Conversation Loop

The agent repeats the following state transitions:

1. A customer message is followed by a policy-model response.
2. Assistant messages and function calls are recorded together through `/record_agent_outputs`, in provider order.
3. Recorded function calls are validated and sent to the matching generated tool route.
4. The tool simulator returns a JSON value, which is appended as a function-call output before the policy model runs
   again.
5. Stop markers, transfer markers, validation failures, generation failures, and the policy-step cap terminate the
   loop.

`max_agent_steps` counts policy-model responses. If the final response contains text, the user simulator receives one
last turn so it can terminate the conversation. If the final response contains tool calls, the calls are recorded but
not executed.

When `parallel_tool_calls=false`, only the first function call is retained and executed. When it is true, every emitted
function call is retained and calls are executed in provider order until one result terminates the trajectory. Calls
after that terminal result remain recorded, but receive no fabricated function-call output.

If one model response contains both assistant text and function calls, every selected item is recorded in its original
order before any call is executed. Parallel function-call outputs are appended after all response items, matching the
policy-visible Responses transcript even for a nonconforming mixed response.

## Validation

Validation occurs as messages enter session state:

- the first customer message cannot immediately terminate the conversation
- function names must exist in the generated tool set
- function arguments must parse as JSON and satisfy the parameter schema
- tool results must parse as JSON and satisfy the return schema
- tool results wrapped in a `json` Markdown code fence are unwrapped for validation

The raw tool-simulator text is retained in the trajectory and returned to the policy model. Validation uses a parsed
copy and does not rewrite the stored output.

## Verification

`/verify` evaluates the completed trajectory. In message-level mode, user and tool-result messages are checked first,
then agent messages, followed by an agent-conversation judgment when earlier checks leave a positive reward. Combined
mode sends the full conversation to one judge request.

The verifier reports:

- reward
- invalid-reason categories
- `user_failure`, `tool_failure`, and `agent_failure` labels
- per-role verification details
- judge diagnostics and provider errors

Successful verification finalizes and removes the in-memory session. A direct caller may retry verification after a
retryable judge-provider error because the resource server retains the session. The policy agent explicitly discards
the session when `/run` exits before successful verification.

Transfer-ground-truth enforcement can assign reward zero and skip the judge when the observed transfer behavior does
not match the scenario label.

## Retries And Failures

Simulator generation retries validate each candidate before accepting it. Judge provider retries are separate from
semantic judge retries: transient HTTP, connection, and timeout failures use bounded exponential backoff, while
successfully returned but malformed judge responses consume semantic attempts.

Exhausted transient infrastructure failures follow Gym's failure-sidecar contract and can be retried by resumed
collection. Semantic policy or trajectory failures remain scored rollout results.

## Output

The returned Gym row has two complementary views:

- `responses_create_params` and `response` contain the policy-visible Responses API transcript
- `result.profile` and `result.source_artifacts` preserve generation identity
- `result.trajectory` contains simulator messages, terminal state, validation details, verification results, prefill
  counts, and continuation indexes

Top-level reward, invalid reasons, failure labels, and judge diagnostics are synchronized with the nested result
before the rollout is returned.

When observability is enabled, every policy, user-simulator, tool-simulator, and judge request uses Gym's rollout path
prefix, so standard model-call captures are correlated to this rollout.
