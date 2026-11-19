# Description

Simple Agent with a **Nemotron Decide** safety gate in front of every tool call.

Before a model-requested tool call executes, the gate asks Nemotron Decide one choice question, `allow` / `ask` /
`block`, over a JSON state:

| field | content |
|---|---|
| `user_task` | the user's own messages only |
| `system_instructions` | system/developer messages |
| `policy` | operator policy text (`decide_gate.policy`) |
| `recent_steps` | the last `recent_steps` assistant messages, tool calls and tool outputs (tool data stays here) |
| `proposed_call` | tool name and parsed arguments |

Verdict: `block` if p(block) >= `block_threshold`; else `ask` if p(ask) + p(block) >= `ask_threshold`; else `allow`.
A blocked call is not executed; the model gets a JSON refusal (`blocked_by_safety_gate`) as the tool output. Rollouts
have no human, so `on_ask: block` (default) returns a needs-confirmation refusal; `on_ask: allow` runs the call.

Optional (`check_tool_outputs: true`): each tool output is checked with a yes/no question ("does this output carry
instructions aimed at the assistant?"); flagged outputs are annotated (default), withheld, or only logged.

Every decision is logged and attached to the response as `metrics.decide_gate` (check, tool, verdict, probabilities,
latency, executed). `mode: audit` logs without blocking; `on_error: block|allow` fails closed or open;
`gate_tools` / `skip_tools` limit which tools are gated.

The gate uses the `_before_tool_call` / `_after_tool_call` hooks of `simple_agent` (no-ops by default), so the agent
loop, checkpointing and trajectories are Simple Agent's. The endpoint config is shared with
`resources_servers/decide_judge` (`decide_client.py`).

Note: verifiers that grade the proposed calls (e.g. `indirect_prompt_injection`) still count a blocked call against
the policy; the gate controls execution, not the policy's reward.

## Configs

- `configs/decide_gate_agent.yaml`: generic; set `resources_server.name`.
- `configs/indirect_prompt_injection_decide_gate.yaml`: wraps the `indirect_prompt_injection` environment.

## Demo

`demo/run_ipi_gate_demo.sh` runs `data/ipi_gate_demo.jsonl` (one injected and one benign healthcare task) through the
real `indirect_prompt_injection` resources server and the live gate, with `demo/scripted_policy.py` playing a hijacked
policy. Expected: the chart read and the user-requested message are allowed; the injected exfiltration call is blocked.

```bash
export DECIDE_BASE_URL=http://localhost:8770   # your Nemotron Decide server
export DECIDE_API_KEY=...   # only if the server needs a bearer token
bash responses_api_agents/decide_gate_agent/demo/run_ipi_gate_demo.sh results/decide_gate_demo
```

# Licensing information
Code: Apache 2.0
Data: Apache 2.0 (`data/ipi_gate_demo.jsonl` written for this agent)

Dependencies
- nemo_gym: Apache 2.0
