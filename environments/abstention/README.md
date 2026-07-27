# Description

This is an environment that trains a policy model to abstain from answering when unsure rather than hallucinating. It uses a three-tier reward scheme:

- **Correct** (1.0): The model provides a correct answer
- **Abstain** (configurable, default 0.5): The model outputs `\boxed{[IDK]}` or the LLM judge grades the answer as NOT_ATTEMPTED
- **Incorrect** (0.0): The model provides an incorrect answer

Correctness is verified by an LLM judge using the OMNISCIENCE_GRADER template instead of string matching. The judge grades the model's extracted answer against the gold target as one of CORRECT, INCORRECT, or NOT_ATTEMPTED.

The dataset used is [HotPotQA](https://hotpotqa.github.io/) (fullwiki split).

# Example usage

## Running servers

```bash
gym env start \
    --environment abstention \
    --model-type openai_model \
    +abstention.resources_servers.abstention.judge_model_server.name=policy_model
```

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent abstention_simple_agent \
    --input environments/abstention/data/example.jsonl \
    --output results/abstention_verify_responses.jsonl \
    --limit 3
```

## Preprocessing HotPotQA data

```bash
python environments/abstention/prepare.py \
    --download \
    --raw-data-dir /path/to/data/hotpotqa \
    --output-dir environments/abstention/data
```

# Licensing information

Code: Apache 2.0
Data:
- HotPotQA: Creative Commons Attribution-ShareAlike 4.0 International

Dependencies:
- nemo_gym: Apache 2.0
