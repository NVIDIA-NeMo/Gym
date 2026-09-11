# Description

This is a resources server that trains a policy model to abstain from answering when unsure rather than hallucinating. It uses a three-tier reward scheme:

- **Correct** (1.0): The model provides a correct answer
- **Abstain** (configurable, default 0.5): The model outputs `\boxed{[IDK]}` or the LLM judge grades the answer as NOT_ATTEMPTED
- **Incorrect** (0.0): The model provides an incorrect answer

Correctness is verified by an LLM judge using the OMNISCIENCE_GRADER template instead of string matching. The judge grades the model's extracted answer against the gold target as one of CORRECT, INCORRECT, or NOT_ATTEMPTED.

The default dataset is [NVIDIA Nemotron-RL-QA-Abstention-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-QA-Abstention-v1). Its public `train` split contains 3,150 examples: 450 HotPotQA questions and 900 each from Go documentation, health, and law. The release has no validation split.

The published JSONL already includes the question, gold answer, `abstention_simple_agent` reference, and answer/abstention prompt in Gym format. No preprocessing is required. Five examples covering all four sources are included for smoke testing.

# Example usage

## Downloading training data

```bash
gym dataset download \
    --repo-id nvidia/Nemotron-RL-QA-Abstention-v1 \
    --artifact data/train.jsonl \
    --output resources_servers/abstention/data/nemotron_qa_abstention_train.jsonl
```

The dataset is public, and the explicit download above works without a token. The config also declares its Hugging Face source so Gym can download missing training data during evaluation or `gym dataset collate --download` when `hf_token` is configured.

## Running servers

Configure the policy model endpoint and credentials before starting the servers. The following smoke-test commands reuse the policy model as the judge; configure a separate `genrm_model` server and omit the override to use an independent judge.

```bash
gym env start \
    --model-type openai_model \
    --resources-server abstention \
    +abstention.resources_servers.abstention.judge_model_server.name=policy_model
```

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent abstention_simple_agent \
    --input resources_servers/abstention/data/example.jsonl \
    --output results/abstention_verify_responses.jsonl \
    --limit 3
```

## Evaluating the training split

This command starts the servers, evaluates ten training examples, and writes the scored rollouts:

```bash
gym eval run \
    --resources-server abstention \
    --model-type openai_model \
    --split train \
    --limit 10 \
    --concurrency 2 \
    --output results/abstention_train_rollouts.jsonl \
    +abstention.resources_servers.abstention.judge_model_server.name=policy_model
```

For already-running servers, use `gym eval run --no-serve` with `--input resources_servers/abstention/data/nemotron_qa_abstention_train.jsonl` instead.

# Licensing information

Code: Apache 2.0
Data:
- Nemotron-RL-QA-Abstention-v1: Creative Commons Attribution 4.0 International, as declared by the dataset card. Published row-level license and provenance fields are preserved in the examples and downloaded data.

Dependencies:
- nemo_gym: Apache 2.0
