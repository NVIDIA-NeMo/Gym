# Proof Judge Resources Server

## Overview

This environment trains or evaluates theorem-proving models that produce both a
solution and a self-evaluation.

The policy response is expected to contain:

- `## Solution`
- `## Self Evaluation`
- a final boxed score in the self-evaluation

The server parses the proof, parses the self-reported score, and then calls an
external or Gym-managed verifier model. Depending on configuration, it can also
call a meta-verifier on the self-analysis.

Reward is:

```text
alpha * r_y + beta * r_z
```

where:

- `r_y` is the verifier score on the generated proof
- `r_z` is a self-consistency term gated by the meta-verifier

## Input Schema

Required fields:

- `responses_create_params`: OpenAI Responses create params for the policy model
- `problem`: The theorem-proving prompt

## Data Preparation

Convert raw proof-problem JSONL into Gym examples with:

```bash
python resources_servers/proof_judge/prepare_data.py \
    --input /path/to/raw.jsonl \
    --output resources_servers/proof_judge/data/example.jsonl
```

By default the converter reads the `problem` field and routes examples to the
`proof_simple_agent`.

## Notes

- The verifier can run through Gym's `/v1/responses` path or through external
  OpenAI-compatible judge servers exposed with `JUDGE_SERVER_ARGS`.
- `alpha` and `beta` in
  [proof_judge.yaml](/Users/smahdavi/Desktop/ls/Gym/resources_servers/proof_judge/configs/proof_judge.yaml)
  control the weight of verifier reward versus self-evaluation consistency.

## Licensing Information

Code: Apache 2.0

Prompt templates and example files in this directory: Apache 2.0 unless noted otherwise.

External datasets converted with `prepare_data.py`: use according to the upstream
dataset and model licenses.
