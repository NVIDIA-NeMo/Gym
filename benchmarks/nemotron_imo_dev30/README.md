<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Nemotron IMO dev30

The 30-problem development set of
["An Open Recipe for IMO Gold"](https://arxiv.org/abs/2609.10712): 20 problems from
[Nemotron-IMO-Bench](https://huggingface.co/datasets/nvidia/Nemotron-IMO-Bench) and 10 recent
competition problems, in four tiers (3 easy, 7 medium, 10 hard, 10 unsolved; tiers come from the
paper's search run). Policies write full proofs; a reference-free jury of three frontier LLM
judges grades each proof 0-7 (see [proof_jury_judge](../../resources_servers/proof_jury_judge/README.md)).

The intended use is pass@1 regression tracking for natural-language math proofs.

## Prompt

The opening of the paper's standard proof-generation prompt (Appendix B.1), without its
self-evaluation instructions:

```
Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.

## Problem

{problem}
```

## Data preparation

```bash
gym eval prepare --benchmark nemotron_imo_dev30
```

Writes `data/nemotron_imo_dev30_benchmark.jsonl`: `problem_idx`, `tier`, `problem`,
`source_dataset`, `source_id`, `dataset_license`. Sources (revisions pinned in `prepare.py`):

| Problems | Source | License |
| --- | --- | --- |
| 20 | `nvidia/Nemotron-IMO-Bench` | CC-BY-4.0 |
| 7 | `Hwilner/imo-proofbench` | CC-BY-4.0 |
| 1 (EGMO) | `ShadenA/MathNet` | CC-BY-4.0 |
| 2 (IMO 2025 P3, P6) | `MathArena/imo_2025` | CC-BY-NC-SA-4.0 |

Three statements (`proofbench133_008`, `_130`, `_133`) differ from the copies used in the paper
in LaTeX typography only.

## Running

```bash
export JUDGE_API_KEY=...   # key for the judges' gateway
gym eval run --benchmark nemotron_imo_dev30 --split benchmark \
    --model-type vllm_model --model <policy> --model-url <base_url> --model-api-key <key> \
    --temperature 1.0 --top-p 0.95 --max-output-tokens 262144 \
    --output results/nemotron_imo_dev30/<policy>.jsonl
```

Each problem is sampled 4 times (`num_repeats: 4`). Expect about 3.5 judge calls per rollout:
three judges, plus three more when their scores differ by more than 2 points.

Use a 262,144-token output budget and always report `truncated` next to `score`: Nemotron-3-Ultra
models reason for 100-200k tokens on these problems, and proofs finishing after 256k tokens
rarely earn credit.

## Reference results (2026-10-07)

| Policy | Serving | Mean score, % of 7 (± s.e. over 4 repeats) | Full credit, % | Truncated, % |
| --- | --- | --- | --- | --- |
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | BF16, vLLM | 24.9 ± 2.2 | 10.0 | 0.0 |
| `nvidia/Nemotron-3-Labs-Ultra-Math-RL` | BF16, vLLM | 30.5 ± 1.3 | 12.5 | 10.0 |

Both at 262,144 output tokens, temperature 1.0, top-p 0.95, served with vLLM (`--reasoning-parser
nemotron_v3`).

The same Nemotron-3-Ultra model behind NVIDIA's inference API (`nvidia/nvidia/nemotron-3-ultra`,
2026-10-06) reasoned about twice as long (median 220k tokens, 53% truncated at 262k) and scored
19.6 ± 2.7. Compare models served the same way.
