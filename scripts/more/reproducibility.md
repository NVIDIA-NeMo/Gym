# Reproducing the published evaluation results

## What these are

## Prerequisites

- **Python 3.13.14 or newer.** Gym does not install on 3.12.
- **`uv` on your `PATH`.** Some benchmarks shell out to scripts that call it and
  fail with a bare `exit 127` without it.

## Running a recipe

## Benchmarks needing extra setup

Most recipes need nothing beyond the prerequisites above. These do:

### SciCode

Scoring needs `test_data.h5` (~1 GB) — the numeric ground-truth the test cases are
checked against. It is not downloaded automatically. Get it from the official SciCode
[Google Drive folder](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR),
save it anywhere, and point `TEST_DATA` at it:

```bash
sha256sum /path/to/test_data.h5
# 48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890

TEST_DATA=/path/to/test_data.h5 ./scicode.sh
```

It must be readable by the user running the recipe — otherwise every test case fails
and the run scores 0 with no error.

## Results

## Comparing your numbers

## Base (pretraining) model

The recipes in this directory cover the **instruct** model. Recipes for the **base**
model live in [`base/`](./base/) — 21 short-context benchmarks plus RULER for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16`.

They work differently: they are `nemo-evaluator-launcher` configs run with
`nel run --config ...`, not Gym CLI scripts, because the base benchmarks are
`lm-evaluation-harness` and `nemo-skills` tasks rather than Gym environments. See
[`base/README.md`](./base/README.md) for their prerequisites, which differ from the ones
above.
