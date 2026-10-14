# Swarm Agent

A multi-agent harness: one task, many model conversations, coordinated by spawn calls or a shared board. Every
topology is a config flavor of this agent, so a swarm runs on any benchmark whose rows carry their tools in
`responses_create_params.tools`, the same way `simple_agent` reads them.

| flavor | topology | how agents coordinate |
|---|---|---|
| `swarm_solo` | solo | one agent, no coordination tools |
| `swarm_agent` (`swarm_flat16`), `swarm_flat64` | flat | the root calls `spawn_agents(tasks)`, workers cannot spawn |
| `swarm_hier256` | hier | the root spawns leads, leads spawn workers |
| `swarm_recursive256` | recursive | sub-agents keep spawning down to depth `max_depth` |
| `swarm_board256` | board | the root calls `spawn_pool(instructions)` once, workers use `board_post`, `board_read`, `board_claim` |
| `swarm_peer16`, `swarm_peer64` | peer | `max_agents` peers start together and use `post`, `read`, `claim`. peer-0 writes the answer from the board at the end |
| `swarm_peerdyn` | peer-dyn | `peer_start` peers grow the group with `recruit(n, brief)` up to `max_agents` |

Task tools are the row's tools. Calls go to the resources server's `POST /<tool>` with the seeded session cookies, so
every agent of a rollout shares one session. A tool that declares an `_agent` parameter gets the caller's id
(`agent-<n>`) filled in. Coordination tools live in the agent: claims are check-and-set on the event loop, so they
are atomic.

## Time, turns and tokens

- `deadline_s` bounds a rollout. Unset, it is the row's `verifier_metadata.budget_s` (SwarmBench rows have one),
  else 3600 s. An agent at depth d stops at `deadline - d * child_margin_s` and makes one tool-free final call, so its
  partial answer reaches its parent in time. `child_margin_s` defaults to a twelfth of the deadline. Spawns and
  recruits that would start agents with no time left are refused.
- The root's final call gets `final_grace_s` beyond the deadline (unset: the row's `grace_s`, else 1800 s).
- Every normal call, the root's included, is capped at `max_tokens`. Only the root's final call gets
  `final_max_tokens`.
- A call slower than `call_timeout_s` is retried once, then the agent stops as if its time were up.
- A reply cut at `max_tokens` without a tool call is resampled once. If it is cut again it is not taken as the
  answer: the agent is asked for its final answer. Empty replies and tool calls the server could not parse also get
  a nudge instead of ending the agent.
- Turn caps (`root_turns`, `worker_turns`) and a context guard (`root_ctx`, `worker_ctx`) end an agent with the same
  final call.

## What a rollout returns

- `response.output` is the root's transcript (for peer modes, peer-0's transcript followed by the reporter's answer),
  and the last assistant message is what verify scores. `response.usage` sums all agents.
- `swarm` holds the tree and per-agent counters: `id`, `parent`, `depth`, `role`, `turns`, `tool_calls`,
  `input_tokens`, `output_tokens`, `length_stops`, `wall_s`, `stop_reason`, `error`, plus run totals.
- `ng_agent_observations` has one `AgentInvocation` per agent (`agent-<n>`), with its parent, the spawn call that
  started it, its full conversation and the response ids of its model calls. Gym folds these into `ng_trajectory`.

## Models and token capture

The root, the peers and the reporter use `model_server`, and leads and workers use `worker_model_server` (default:
`model_server`). Every call of every agent goes through the Gym model server on the rollout's path
(`/ng-rollout/<id>/...`), so model-call capture and training token capture record the whole tree under one rollout
id, interleaved in call order. Inline token ids in `response.output` cover only the root, so set
`token_id_capture: true` on the agent to train on the other agents' turns. Join captured calls to agents through the
response ids in `ng_agent_observations`.

## Example: SwarmBench

SwarmBench rows hold the prompt and `verifier_metadata` only, because the OpenCode configs reach the tools over MCP.
Add the track's tools to each row from the branch with `benchmarks/swarmbench`:

```python
import json
import sys

from resources_servers.swarmbench.app import mcp_tool_name, tool_specs

for line in open(sys.argv[1]):
    row = json.loads(line)
    meta = row["verifier_metadata"]
    specs = tool_specs(meta["track"], meta.get("track_version", 1))
    tools = [{"type": "function", "name": mcp_tool_name(meta["track"], t["name"]), "description": t["description"],
              "parameters": t["parameters"], "strict": False} for t in specs]
    params = row.pop("responses_create_params", None) or {"input": [{"role": "user", "content": row.pop("prompt")}]}
    print(json.dumps(row | {"responses_create_params": params | {"tools": tools}}))
```

Then bind a swarm flavor to the server:

```yaml
# swarmbench_swarm.yaml
swarmbench:
  resources_servers:
    swarmbench:
      entrypoint: app.py
      domain: agent
      verified: false
swarmbench_swarm_board256:
  responses_api_agents:
    swarm_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: swarmbench
      model_server:
        type: responses_api_models
        name: policy_model
      topology: board
      max_agents: 256
swarmbench_swarm_environment_server:
  environment_servers:
    legacy_agent:
      entrypoint: app.py
      agent_server:
        type: responses_api_agents
        name: swarmbench_swarm_board256
```

```bash
python add_tools.py benchmarks/swarmbench/data/swarmbench_benchmark.jsonl > swarmbench_swarm.jsonl
gym env start --config swarmbench_swarm.yaml --model-type vllm_model \
    --model "$POLICY_MODEL" --model-url "$POLICY_BASE_URL" --model-api-key "$POLICY_API_KEY"
gym eval run --no-serve --agent swarmbench_swarm_board256 --input swarmbench_swarm.jsonl \
    --output results/swarmbench_board256.jsonl
```

On a benchmark whose rows already carry tools, load a flavor next to it instead, for example
`--agent-type swarm_agent/swarm_peer16`, and Gym rehosts the benchmark's agent on the swarm.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
