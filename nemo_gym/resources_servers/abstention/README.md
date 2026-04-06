# Description

This is a resources server that trains a policy model to abstain from answering when unsure rather than hallucinating. It uses a three-tier reward scheme:

- **Correct** (1.0): The model provides a correct answer
- **Abstain** (configurable, default 0.5): The model outputs `\boxed{[IDK]}` or the LLM judge grades the answer as NOT_ATTEMPTED
- **Incorrect** (0.0): The model provides an incorrect answer

Correctness is verified by an LLM judge using the OMNISCIENCE_GRADER template instead of string matching. The judge grades the model's extracted answer against the gold target as one of CORRECT, INCORRECT, or NOT_ATTEMPTED.

The dataset used is [HotPotQA](https://hotpotqa.github.io/) (fullwiki split).

# Example usage

## Running servers

```bash
config_paths="nemo_gym/responses_api_models/openai_model/configs/openai_model.yaml, \
nemo_gym/resources_servers/abstention/configs/abstention.yaml"
ng_run "+config_paths=[$config_paths]" \
    +abstention.resources_servers.abstention.judge_model_server.name=policy_model
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=abstention_simple_agent \
    +input_jsonl_fpath=nemo_gym/resources_servers/abstention/data/example.jsonl \
    +output_jsonl_fpath=results/abstention_verify_responses.jsonl \
    +limit=3
```

## Preprocessing HotPotQA data

```bash
python nemo_gym/resources_servers/abstention/dataset_preprocess.py \
    --download \
    --raw-data-dir /path/to/data/hotpotqa \
    --output-dir nemo_gym/resources_servers/abstention/data
```

# Licensing information

Code: Apache 2.0
Data:
- HotPotQA: Creative Commons Attribution-ShareAlike 4.0 International

Dependencies:
- nemo_gym: Apache 2.0
