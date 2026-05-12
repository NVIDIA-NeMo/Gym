# Description
This is a resources server that is to be used to verify a single action taken by an agent that can either call a tool or send a chat message to the user as the next step in a trajectory.  For each verification request, there is an expected action that is either a tool call or a chat message.  An expected tool call is compared with a tool call issued by the agent by programmatically comparing the arguments in the tool calls.  If the expected action is a chat message, then the agent receives a positive reward if it sends a chat message, and a negative reward if it calls a tool instead.

Data links: ?

# Parallel tool calls
Rows may also set `expected_action.type` to `function_call_batch` with a `calls` list containing the same `function_call` objects used by single-call rows. The verifier matches parallel tool calls as an unordered multiset, so response output order does not matter.

By default, matching is exact: the actual response must contain the same number of tool calls as the expected batch, and every expected call must match one actual call. The resource server config can relax the cardinality check with `tool_call_comparator_config.allow_subset: true` or `tool_call_comparator_config.allow_superset: true`. Both default to `false`.

Rewards remain binary by default. Set `tool_call_comparator_config.parallel_tool_call_reward_mode: fractional` to return the matched-call fraction after the configured cardinality gate passes.

# Example usage

## Running servers
The following command can be used to run this resources server, along with the tool simulation agent and an OpenAI model:
```bash
gym env start \
    --resources-server single_step_tool_use_with_argument_comparison \
    --model-type openai_model
```

Then, rollouts can be collected using a command such as the following:
```bash
gym eval run --no-serve \
    --agent single_step_tool_use_with_argument_comparison_agent \
    --input resources_servers/single_step_tool_use_with_argument_comparison/data/example.jsonl \
    --output resources_servers/single_step_tool_use_with_argument_comparison/data/example_rollouts.jsonl
```

# Licensing information
Code: Apache 2.0<br>
Data: ?

Dependencies
- nemo_gym: Apache 2.0
