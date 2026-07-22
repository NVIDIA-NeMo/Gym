# Synthetic Tool-Use Rollouts

One rollout joins a generated policy, tool set, and customer scenario with three model roles:

- the policy model acts as the customer-service agent
- the user simulator produces customer turns
- the tool simulator produces results for generated tool schemas
- the judge model scores completed trajectories

The policy loop runs in `responses_api_agents/synthetic_tool_use_agent`. Session state, simulation, validation, and
verification run in `resources_servers/synthetic_tool_use_simulation`.

## Session Setup

`/seed_session` loads the policy, tools, scenario, model-server references, retry settings, and verification settings
for one rollout. Most rows start without conversation history, so the resource server generates the first customer
turn. Rows may also contain prefilled Responses history; supported user, assistant, function-call, and
function-call-output items are hydrated into session state before generation continues.

`/session_tools` exposes the generated tools as Responses API function tools for the policy model.

## Conversation Loop

The agent repeats the following state transitions:

1. A customer message is followed by a policy-model response.
2. Assistant text is recorded through `/record_agent_message`, then `/next_user_message` generates the next customer
   turn.
3. Function calls are validated and sent to the matching generated tool route.
4. The tool simulator returns a JSON value, which is appended as a function-call output before the policy model runs
   again.
5. Stop markers, transfer markers, validation failures, generation failures, and the policy-step cap terminate the
   loop.

`max_agent_steps` counts policy-model responses. If the final response contains text, the user simulator receives one
last turn so it can terminate the conversation. If the final response contains tool calls, the calls are recorded but
not executed.

When `parallel_tool_calls=false`, only the first function call is executed. When it is true, every emitted function
call is retained and executed.

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
- `result.trajectory` contains simulator messages, terminal state, validation details, verification results, prefill
  counts, and continuation indexes

Top-level reward, invalid reasons, failure labels, and judge diagnostics are synchronized with the nested result
before the rollout is returned.
