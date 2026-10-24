# RNA Design

RNA inverse folding, as both a benchmark and a training environment: given a target secondary structure, design
sequences that fold into it. Each rollout is one design campaign with a fixed budget of tool calls on a published
generator (gRNAde), a folding engine (EternaFold) and the suite's scorers. When the budget is spent, the whole pool
of designs is re-scored with the suite's own metric and gates, and the reward is 1.0 if the campaign solved the
target.

- **Benchmark:** eight fixed suites built from the published [RNAInvBench](https://github.com/COLA-Laboratory/RNAInvBench)
  sets: Eterna100 v1 and v2, Rfam learn and Taneda, ArchiveII, and three anta sets (1,694 targets in all). Three
  model-free baselines run on the same budget.
- **Training environment:** the agent mode records the model's token IDs (`token_id_capture`), so the same rollouts
  train with NeMo RL.

The code is in [`responses_api_agents/rna_design_agent`](../../responses_api_agents/rna_design_agent/README.md) (the
campaign loop and tools) and [`resources_servers/rna_design`](../../resources_servers/rna_design/README.md) (the same
tools over MCP, for Claude Code).

## Setup

```bash
# gRNAde, its checkpoints (about 100 MB) and a torch env, no login needed. Runs on CPU or a GPU.
python responses_api_agents/rna_design_agent/prepare.py --setup ~/rna_env
export RNA_WORKER_PYTHON=~/rna_env/venv/bin/python GRNADE_REPO=~/rna_env/geometric-rna-design
```

EternaFold, which scores every design, and the suites are built on first start. That needs internet once, plus
`git`, `make` and a C++ compiler. Set `ARNIEFILE` to use an existing EternaFold build instead.

## Run

The agent mode needs no sandbox:

```bash
# 20 Eterna100 v2 targets as dataset rows
python responses_api_agents/rna_design_agent/prepare.py rnainvbench_eterna100_v2 --output rows.jsonl --limit 20

gym env start --environment rna_design/config_agent --model-type vllm_model
gym eval run --no-serve --agent rna_design_agent --input rows.jsonl --output rollouts.jsonl
```

## Modes

Every mode gets the same tools and budget, and is scored the same way. For another mode, swap `--environment` and
`--agent`.

| Mode | `--environment` | `--agent` | Who picks the tool calls |
|---|---|---|---|
| agent | `rna_design/config_agent` | `rna_design_agent` | the model, one call per round, from a fresh view of the campaign |
| fixed | `rna_design/config_fixed` | `rna_design_fixed` | a fixed cycle: generate, then score, until the budget runs out |
| fixed_deep | `rna_design/config_fixed_deep` | `rna_design_fixed_deep` | the fixed cycle plus every other scorer the suite offers |
| generator | `rna_design/config_generator` | `rna_design_generator` | one pass of the generator, then the score |
| Claude Code | `rna_design` | `rna_design_claude_code_agent` | Claude Code in its own sandbox per rollout, calling the same tools over MCP |

The model-free modes never call the model, but Gym still starts a model server, so `policy_base_url` must be set.

### Claude Code

The catalog workload (`config.yaml` and `manifest.yaml`). Each rollout runs Claude Code in its own OpenSandbox box.
The box must reach the Gym node, so Gym binds the node's address (`use_absolute_ip`) and Claude Code calls the model
through the Gym model server at `$NODE_IP:$PORT`.

```bash
export OPENSANDBOX_DOMAIN=https://<your cell> OPENSANDBOX_API_KEY=<key>
gym env start --environment rna_design --model-type vllm_model \
    ++use_absolute_ip=true ++policy_model.responses_api_models.vllm_model.port=$PORT \
    ++rna_design_claude_code_agent.responses_api_agents.harness_agent.sandbox_model_base_url=http://$NODE_IP:$PORT
gym eval run --no-serve --agent rna_design_claude_code_agent --input resources_servers/rna_design/data/example.jsonl \
    --output rollouts.jsonl ++head_server.host=$NODE_IP
```

## Reward

`1.0` if the best design that passes the gates reaches the suite's `pass_threshold` on its primary metric, else
`0.0`. Most suites need an exact fold (1.0). The pseudoknotted ones (ArchiveII, anta pseudo) need 0.8 of the
crossing pairs. A rollout that measured nothing about the model is masked instead of scored: a timeout, a crashed
worker, every tool call failing, a model that was never reached, or a model endpoint that failed partway with
nothing solved.

Long targets are slow on CPU. Raise `rollout_timeout_s` or use a GPU.

## Tests

```bash
gym env test rna_design   # the verifier fixture: one exact fold scores 1.0, one with no pairs 0.0
```

The component tests are in each component's README.

## Licensing

The code is Apache-2.0. The data comes from published sets under their own terms, and ArchiveII states none. See
[LICENSE-third-party](../../responses_api_agents/rna_design_agent/LICENSE-third-party) for each component and set.
