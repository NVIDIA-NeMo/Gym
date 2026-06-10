# Proof Verification Resources Server

## Overview

This environment scores a model that evaluates the quality of a proof or solution.
The policy is given a problem, a candidate proof, and instructions to write a
detailed evaluation ending with a boxed score in `{0, 0.5, 1}`.

The server then:

1. Parses the model's predicted score from the response.
2. Compares it against `ground_truth_verify_score`.
3. Sends the candidate judgement to a meta-verifier judge model together with the
   reference judgement.

The final reward is:

```text
reward = (1 - abs(predicted_score - ground_truth_verify_score)) * r_meta
```

where `r_meta` is the boxed score returned by the meta-verifier.

## Input Schema

Required fields:

- `responses_create_params`: OpenAI Responses create params for the policy model
- `problem`: Original proof problem
- `proof`: Candidate proof or solution to evaluate
- `ground_truth_judgement`: Reference evaluation text
- `ground_truth_verify_score`: Reference score in `{0, 0.5, 1}`

## Data Preparation

Convert raw JSONL rows into Gym-compatible examples with:

```bash
python resources_servers/proof_verification/prepare_data.py \
    --input /path/to/raw.jsonl \
    --output resources_servers/proof_verification/data/example.jsonl
```

The converter expects JSONL rows with `problem`, `proof`,
`ground_truth_judgement`, and `ground_truth_verify_score`.

## Notes

- The judge can run either through a Gym-managed model server or through an
  external OpenAI-compatible endpoint via `JUDGE_SERVER_ARGS`.
- Long proofs and long verification outputs are rejected by length guards in
  the server.

## Licensing Information

Code: Apache 2.0

Prompt templates and example files in this directory: Apache 2.0 unless noted otherwise.

External datasets converted with `prepare_data.py`: use according to the upstream
dataset and model licenses.
