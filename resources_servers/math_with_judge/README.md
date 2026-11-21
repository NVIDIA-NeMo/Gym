# Description
This is a resources server for verifying the correctness of answers to mathematical problems.  It uses a combination of the Hugging Face Math-Verify library and an LLM as a judge.

The problems in the OpenMathReasoning dataset are taken from the
[OpenMathReasoning dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-math-OpenMathReasoning)
on Hugging Face.


# Example usage

## Per-task output format

A task may supply `answer_format_regex` with a named `answer` capture, for example
`"\\s*<answer>(?P<answer>[^<>\\r\\n]+)</answer>\\s*"`. The regex must match the entire
text of the last assistant message (earlier messages and reasoning items are excluded). Missing or empty captures
score zero without calling the mathematical verifier or judge. A passing match
is graded using only the captured answer; the original response is retained.
The result retains `answer_format_regex` and includes `answer_format_valid`.
Nested boxes or unmatched braces bypass symbolic grading to avoid selecting a
fragment; the judge receives the complete capture (or the reward is zero when
the judge is disabled). Without a contract, existing answer
extraction and result fields are unchanged. Regexes are authored task settings;
use bounded, simple patterns appropriate to the template.

This separates formatting compliance from mathematical equivalence. The judge's
format-tolerant equivalence prompt cannot override a failed format contract.

## Running servers
The following are example commands for running this resources server, along with the simple agent and an OpenAI model:
```bash
gym env start \
    --model-type openai_model \
    --resources-server math_with_judge \
    +math_with_judge.resources_servers.math_with_judge.judge_model_server.name=policy_model
```

The OpenMathReasoning download below and `--download` under "Prepare for trajectory collection" fetch from the
GitLab model registry. They need the `mlflow` extra (`uv sync --extra mlflow`) and `mlflow_tracking_uri` and
`mlflow_tracking_token` in `env.yaml`.

To download the OpenMathReasoning dataset, the following command can be run:
```bash
gym dataset download --storage gitlab \
    --name math_open_math_reasoning \
    --revision 0.0.1 \
    --artifact open_math_reasoning_problems.jsonl \
    --output data/open_math_reasoning_problems.jsonl
```

Then, rollouts can be collected using a command such as the following:
```bash
gym eval run --no-serve \
    --agent math_with_judge_simple_agent \
    --input data/open_math_reasoning_problems.jsonl \
    --output results/example_open_math_reasoning_verify_responses.jsonl \
    --limit 5
```

## Prepare for trajectory collection
This writes `validation.jsonl` with each DAPO17k train problem repeated 16 times. NeMo RL reads only the
validation split when `is_trajectory_collection: true`. `--download` fetches `dapo17k` 0.0.1 from the GitLab
model registry.
```bash
gym dataset collate \
    --config resources_servers/math_with_judge/configs/dapo17k_trajectory_collection.yaml \
    --config responses_api_models/openai_model/configs/openai_model.yaml \
    --output-dir data/dapo17k_trajectory_collection \
    --mode train_preparation \
    --download
```

# Licensing information
Code: Apache 2.0<br>
Data:
- OpenMathReasoning: Creative Commons Attribution 4.0 International
- Math Stack Overflow: Creative Commons Attribution-ShareAlike 4.0 International

Dependencies
- nemo_gym: Apache 2.0
- math-verify: [Apache 2.0](https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/LICENCE)
