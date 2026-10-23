# RNA Design

An RNA inverse-folding benchmark: given a target structure, design sequences that fold into it. Each rollout is
one design campaign with a fixed budget of tool calls. The tools are a published generator (gRNAde), a folding
engine (EternaFold) and the suite's scorers. When the budget is spent, the whole pool of designs is re-scored with
the suite's own metric and gates.

Eight suites, built on first start from the published [RNAInvBench](https://github.com/COLA-Laboratory/RNAInvBench)
sets: Eterna100 v1 and v2, Rfam learn and Taneda, ArchiveII, and three anta sets (1,694 targets in all).
No dataset is redistributed here and each carries its publisher's terms, ArchiveII's unstated — see
[LICENSE-third-party](LICENSE-third-party). The build needs network access on first start.

## Setup

```bash
# gRNAde, its checkpoints (about 100 MB) and a torch env, no login needed. Runs on CPU or a GPU.
python responses_api_agents/rna_design_agent/prepare.py --setup ~/rna_env
export RNA_WORKER_PYTHON=~/rna_env/venv/bin/python GRNADE_REPO=~/rna_env/geometric-rna-design
```

EternaFold, which scores every design, and the suites in `suites/` are built on first start. That needs internet
once, plus `git`, `make` and a C++ compiler. Set `ARNIEFILE` to use an existing EternaFold build instead.

## Run

```bash
# 20 Eterna100 v2 targets as dataset rows
python responses_api_agents/rna_design_agent/prepare.py rnainvbench_eterna100_v2 --output rows.jsonl --limit 20

gym env start --config responses_api_agents/rna_design_agent/configs/rna_design_agent.yaml --model-type vllm_model
gym eval run --no-serve --agent rna_design_agent --input rows.jsonl --output rollouts.jsonl
```

## Modes

Every mode gets the same tools and budget, and is scored the same way.

| Mode | Config | Who picks the tool calls |
|---|---|---|
| `agent` | `configs/rna_design_agent.yaml` | the model, one call per round, from a fresh view of the campaign |
| `fixed` | `configs/rna_design_fixed.yaml` | a fixed cycle: generate, then score, until the budget runs out |
| `fixed_deep` | `configs/rna_design_fixed_deep.yaml` | the fixed cycle plus every other scorer the suite offers |
| `generator` | `configs/rna_design_generator.yaml` | one pass of the generator, then the score |
| Claude Code | `resources_servers/rna_design/configs/rna_design.yaml` | Claude Code in its own sandbox per rollout, calling the same tools over MCP |

For another mode, swap the config and `--agent` (`rna_design_fixed`, `rna_design_fixed_deep`,
`rna_design_generator`). The model-free modes never call the model, but Gym still starts a model server, so
`policy_base_url` must be set. Claude Code is in [`resources_servers/rna_design`](../../resources_servers/rna_design/README.md).

## Reward

`1.0` if the best design that passes the gates reaches the suite's `pass_threshold` on its primary metric, else
`0.0`. Most suites need an exact fold (1.0). The pseudoknotted ones (ArchiveII, anta pseudo) need 0.8 of the
crossing pairs. The response also reports `best`, `pool_size`, `rounds`, `tool_calls`, `tool_failures` and token
counts.

A campaign that could not run (a timeout, a crashed worker, or every tool call failing) is masked, not scored as a
bad design.

## Notes

- Each campaign runs in its own process, under `RNA_WORKER_PYTHON`, because gRNAde needs torch and the campaign
  loop sets process-wide state.
- gRNAde imports `torch_scatter` and `torch_cluster`, which have no aarch64 wheels. `rds/tools/pyg_shims/` has pure
  PyTorch versions of the one function it uses from each, used only when the real packages are missing.
- Long targets are slow on CPU. Raise `rollout_timeout_s` or use a GPU.

## Third-party components

See [LICENSE-third-party](LICENSE-third-party). ViennaRNA is not used, because its licence restricts
redistribution.

## Tests

```bash
gym env test +entrypoint=responses_api_agents/rna_design_agent
```
