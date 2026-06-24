# IFEval

[IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) is the original instruction-following evaluation benchmark from Google Research. It evaluates whether a model follows explicit, programmatically-verifiable constraints (length, format, keywords, etc.) embedded in a prompt. The data is the upstream `input_data.jsonl` of 541 prompts.

This benchmark chains to the existing `instruction_following` resources server, which uses [`verifiable_instructions`](https://github.com/abukharin-nv/verifiable-instructions) to perform the per-instruction checks. Per-rollout reward equals Skills' STRICT-mode accuracy under `grading_mode="binary"` (1 if all instructions pass, 0 otherwise).

## Prepare data

```bash
gym eval prepare --benchmark ifeval
```

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark ifeval
```

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent ifeval_instruction_following_simple_agent \
    --input benchmarks/ifeval/data/ifeval_benchmark.jsonl \
    --output results/ifeval_rollouts.jsonl \
    --num-repeats 4
```

## Scoring notes

* `grading_mode="binary"` — reward is 1.0 only when all instructions in the prompt are satisfied. This matches Skills' `prompt_strict_accuracy`.
* `grading_mode="fraction"` — set on the dataset row to instead get the per-instruction strict accuracy (Skills' `instruction_strict_accuracy`).
* **Loose-mode evaluation** (Skills' `prompt_loose_accuracy` / `instruction_loose_accuracy`) is not implemented in the existing `instruction_following` server. Reproducing the four-way Skills metric breakdown would require a server-side enhancement that adds the eight-perturbation loose check to `verify()` and a custom `compute_metrics()`. This migration is data-only and reuses the existing strict-mode server.
