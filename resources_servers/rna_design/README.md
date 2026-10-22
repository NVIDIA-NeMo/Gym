# RNA Design (Claude Code)

The [RNA design benchmark](../../responses_api_agents/rna_design_agent/README.md) with Claude Code as the agent.
Each rollout runs Claude Code in its own OpenSandbox box. The design tools and the scoring stay on the Gym node and
are served to the box over MCP, with a session token per rollout, so code the agent writes cannot reach the scorer.
Claude Code sees only the tools the task offers, under the same budget and reward as the other modes.

## Run

Do the setup in the benchmark README first. Then:

```bash
export OPENSANDBOX_DOMAIN=https://<your cell> OPENSANDBOX_API_KEY=<key>
gym env start --config resources_servers/rna_design/configs/rna_design.yaml --model-type vllm_model \
    ++use_absolute_ip=true ++policy_model.responses_api_models.vllm_model.port=$PORT \
    ++rna_design_claude_code_agent.responses_api_agents.harness_agent.sandbox_model_base_url=http://$NODE_IP:$PORT
gym eval run --no-serve --agent rna_design_claude_code_agent --input rows.jsonl --output rollouts.jsonl \
    ++head_server.host=$NODE_IP
```

The box must reach the Gym node, so Gym binds the node's address (`use_absolute_ip`) and Claude Code calls the
model through the Gym model server at `$NODE_IP:$PORT`. The default image is public. Any image with Python 3.12+
and Node.js works.

A rollout where the model was never reached, or where every tool call failed and nothing was designed, is masked.

## Tests

```bash
python -m pytest resources_servers/rna_design/tests
```
