# Description
This is a resources server that is to be used to verify a single action taken by an agent that can either call a tool or send a chat message to the user as the next step in a trajectory.  For each verification request, there is an expected action that is either a tool call or a chat message.  An expected tool call is compared with a tool call issued by the agent by programmatically comparing the arguments in the tool calls.  If the expected action is a chat message, then the agent receives a positive reward if it sends a chat message, and a negative reward if it calls a tool instead.

Data links: ?

# Parallel tool calls
Rows may also set `expected_action.type` to `function_call_batch` with a `calls` list containing the same `function_call` objects used by single-call rows. The verifier matches parallel tool calls as an unordered multiset, so response output order does not matter. Because argument matching is fuzzy, one actual call can satisfy several expected calls; the verifier resolves this with a maximum bipartite matching rather than pairing calls greedily.

Scoring has two independent stages, both configured under `tool_call_comparator_config`.

## Cardinality gate: which responses are admissible

`allow_subset` admits responses that make **fewer** calls than expected, and `allow_superset` admits responses that make **more**. Both default to `false`, so by default the call count must match the expected batch exactly and anything else scores zero.

The gate applies to `function_call_batch` rows only. A plain `function_call` row asks whether the model made *that call*, not how many calls it made, so surplus calls never reduce its reward. Chat templates do not render differently for `parallel_tool_calls` — the Nemotron template never references the flag — so a model is never told that only one call is allowed, and charging it for an extra call would penalize behaviour it had no signal to avoid. This also keeps parallel tool-call support a **no-op for every existing single-call dataset**. Set `parallel_tool_call_reward_mode: f1` if you do want surplus calls charged for on single-call rows.

## Reward mode: how much credit an admissible response earns

`parallel_tool_call_reward_mode` accepts:

| Mode | Reward |
| --- | --- |
| `binary_strict` (default) | `1.0` only if every required call matched, else `0.0` |
| `fractional` | the matched fraction of the required calls |
| `f1` | `2 * matched / (expected + actual)` — the harmonic mean of precision and recall |

The two stages are complementary. Under `binary_strict` and `fractional` the gate is a *free pass*: once a shape is admitted, surplus calls cost nothing and `allow_subset` lets a response earn full credit for emitting only the easiest call. Concretely, with `allow_superset: true` a response that makes both expected calls plus twenty junk calls still scores `1.0`.

Under `f1` the gate instead decides which imperfect shapes are worth *partial* credit, and missing and surplus calls are penalized symmetrically — the same twenty-junk-call response scores `2 * 2 / (2 + 22) = 0.167`. Only an exact set of calls reaches `1.0`. Prefer `f1` for RL, where the permissive gates are otherwise reward-hacking vectors.

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
