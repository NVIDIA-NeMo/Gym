# Proof GenSelect Resources Server

## Overview

This environment presents a math problem together with two candidate proofs and
asks the model to choose the better one.

The policy must return:

```text
<best_solution>1</best_solution>
```

or:

```text
<best_solution>2</best_solution>
```

The server parses that tag and assigns reward `1.0` when the selected index
matches `correct_index`, otherwise `0.0`.

## Input Schema

Required fields:

- `responses_create_params`: OpenAI Responses create params for the policy model
- `problem`: Problem statement
- `proof_1`: First candidate proof
- `proof_2`: Second candidate proof
- `correct_index`: Correct answer, either `1` or `2`

Optional metadata:

- `score_1`
- `score_2`

## Data Preparation

Convert pairwise-selection JSONL into Gym examples with:

```bash
python resources_servers/proof_genselect/prepare_data.py \
    --input /path/to/raw.jsonl \
    --output resources_servers/proof_genselect/data/example.jsonl
```

The converter expects `problem`, `proof_1`, `proof_2`, and `correct_index`.

## Licensing Information

Code: Apache 2.0

Prompt templates and example files in this directory: Apache 2.0 unless noted otherwise.

External datasets converted with `prepare_data.py`: use according to the upstream
dataset and model licenses.
