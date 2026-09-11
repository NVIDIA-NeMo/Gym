# Description

This is an environment that trains a policy model to abstain from answering when unsure rather than hallucinating. It uses a three-tier reward scheme:

- **Correct** (1.0): The model provides a correct answer
- **Abstain** (configurable, default 0.5): The model outputs `\boxed{[IDK]}` or the LLM judge grades the answer as NOT_ATTEMPTED
- **Incorrect** (0.0): The model provides an incorrect answer

Correctness is verified by an LLM judge using the OMNISCIENCE_GRADER template instead of string matching. The judge grades the model's extracted answer against the gold target as one of CORRECT, INCORRECT, or NOT_ATTEMPTED.

The default dataset is [NVIDIA Nemotron-RL-QA-Abstention-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-QA-Abstention-v1), with 3,150 training examples spanning HotPotQA, Go documentation, health, and law. Its public release contains one `train` split; no validation split is configured. The released JSONL already contains the questions, gold answers, agent routing, and boxed-answer/`[IDK]` prompt required by this environment, so no preprocessing is needed.

# Example usage

## Downloading the training data

Download the public dataset to the configured training path:

```bash
gym dataset download \
    --repo-id nvidia/Nemotron-RL-QA-Abstention-v1 \
    --artifact data/train.jsonl \
    --output environments/abstention/data/nemotron_qa_abstention_train.jsonl
```

This replaces the deprecated HotPotQA download and preprocessing workflow. Existing `hotpotqa_train.jsonl` and `hotpotqa_val.jsonl` files are no longer used by the default configuration.

## Quick training-set evaluation

After configuring the policy endpoint, run a small evaluation on the training split:

```bash
gym eval run \
    --environment abstention \
    --model-type openai_model \
    --split train \
    --limit 10 \
    --concurrency 2 \
    --output results/abstention_train_smoke.jsonl \
    +abstention.resources_servers.abstention.judge_model_server.name=policy_model
```

This smoke test uses the policy endpoint as the judge. For a separate judge, configure the `genrm_model` server referenced by the environment and omit the override.

## Running servers for the bundled example

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

# Licensing information

Code: Apache 2.0
Data:
- Nemotron-RL-QA-Abstention-v1: Creative Commons Attribution 4.0 International, as declared by the dataset card. Released rows also retain their source license metadata (`CC BY-SA 4.0`).

Dependencies:
- nemo_gym: Apache 2.0
