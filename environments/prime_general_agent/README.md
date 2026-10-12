# Description

Self-growing toolbench environment that currently 4,418 synthetic tool-use tasks across 1,037 families and 5 difficulty tiers and can be used to generate more. 

Prime Intellect environment integrated through the verifiers library integration.

Full description: https://app.primeintellect.ai/dashboard/environments/primeintellect/general-agent

For installing additional prime environments from the Environments Hub and generating datasets, see [responses_api_agents/verifiers_agent/README.md](../../responses_api_agents/verifiers_agent/README.md).

## Notes

1. Use the local solver to start, not the default env id. Set `vf_env_id: general-agent-solver-local` in `config.yaml`. The default `general-agent` env id may route to a sandbox solver (including options with OpenCode or RLM) that requires a funded `PRIME_API_KEY`. The local solver runs in-process with no sandbox.
2. Dataset `info` must be at the row top level. The local solver reads `state["info"]["task_dir"]`. If you have a dataset where `info` is nested under `verifier_metadata`, lift it. `prepare.py` writes rows in the correct shape.
3. `agent_ref.name` must match the agent directory name (`prime_general_agent`). NeMo Gym dispatches rows to the agent by this name.
4. Corpus version must match the installed package. `info.task_dir` paths point into the installed `general-agent` package's bundled tasks/ tree. Regenerate the dataset whenever you bump the `general-agent` pin.
5. The requirements is currently pinned to verifiers main as the regular 0.1.14 pin seems too old, but this should be pinned soon. It is currently tested on verifiers>=0.1.15.dev2. 

## Quick start

```bash
ng_run "+config_paths=[environments/prime_general_agent/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

```bash
ng_collect_rollouts \
    +agent_name=prime_general_agent \
    +input_jsonl_fpath=environments/prime_general_agent/data/example.jsonl \
    +output_jsonl_fpath=results/prime_general_agent_rollouts.jsonl \
    +limit=1
```

## Prepare training data

```bash
python environments/prime_general_agent/prepare.py --split train --output environments/prime_general_agent/data/train.jsonl
```

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- verifiers: Apache 2.0
