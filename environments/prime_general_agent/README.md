# Description

Self-growing toolbench environment that currently 4,418 synthetic tool-use tasks across 1,037 families and 5 difficulty tiers and can be used to generate more. 

Prime Intellect environment integrated through the verifiers library integration.

Full description: https://app.primeintellect.ai/dashboard/environments/primeintellect/general-agent

For installing additional prime environments from the Environments Hub and generating datasets, see [responses_api_agents/verifiers_agent/README.md](../../responses_api_agents/verifiers_agent/README.md).

## Notes

1. Use the local solver to start, not the default env id. Set `vf_env_id: general-agent-solver-local` in `config.yaml`. The default `general-agent` env id may route to a sandbox solver (including options with OpenCode or RLM) that requires a funded `PRIME_API_KEY`. The local solver runs in-process with no sandbox.
2. Dataset `info` must be at the row top level. The local solver reads `state["info"]["task_dir"]`. If you have a dataset where `info` is nested under `verifier_metadata`, lift it. `prepare.py` writes rows in the correct shape.
3. `agent_ref.name` must match the agent directory name (`prime_general_agent`). NeMo Gym dispatches rows to the agent by this name.
4. Corpus version must match the installed package. `info.task_dir` is relative to the installed `general-agent` package's bundled tasks/ tree, and the agent resolves it against its own install. Regenerate the dataset whenever you bump the `general-agent` pin.
5. The agent pins verifiers v0.3.1, like `verifiers_agent`, and general-agent 0.1.3. general-agent 0.1.4 ships no tasks/, and 0.1.5 needs verifiers v1, which drops the legacy API `verifiers_agent` uses.

## Quick start

```bash
gym env start --environment prime_general_agent --model-type vllm_model
```

```bash
gym eval run --no-serve \
    --agent prime_general_agent \
    --input environments/prime_general_agent/data/example.jsonl \
    --output results/prime_general_agent_rollouts.jsonl \
    --limit 3
```

## Prepare training data

`gym env start` builds the agent venv that `prepare.py` needs.

```bash
responses_api_agents/prime_general_agent/.venv/bin/python environments/prime_general_agent/prepare.py \
    --output environments/prime_general_agent/data/train.jsonl
```

# Licensing information
Code: Apache 2.0
Data: Apache 2.0 (general-agent task corpus from Prime Intellect's [prime-envs](https://github.com/PrimeIntellect-ai/prime-envs))

Dependencies
- nemo_gym: Apache 2.0
- verifiers: MIT
- general-agent: Apache 2.0
