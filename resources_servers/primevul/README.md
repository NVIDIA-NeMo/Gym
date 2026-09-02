# PrimeVul resources server

This resources server scores the paired binary vulnerability-classification protocol from
[PrimeVul](https://github.com/DLVulDet/PrimeVul), introduced in
[Vulnerability Detection with Code Language Models: How Far Are We?](https://arxiv.org/abs/2403.18624)
(ICSE 2025).

The runnable benchmark and its manifest are under [`benchmarks/primevul`](../../benchmarks/primevul/).
This server is stateless, exposes no tools, and is paired with `responses_api_agents/simple_agent`.

## Verification

The published PrimeVul prompt asks the model to return one binary option:

- `YES`: a security vulnerability is present.
- `NO`: no security vulnerability is present.

The parser accepts these tokens case-insensitively, including the full option text. A bare `(1)` or
`(2)` selection is also accepted. If multiple explicit YES/NO options appear, the final one is used.
Closed `<think>` and `<thinking>` blocks are removed before parsing. A response without an option
receives reward `0.0` and increments `mean/parse_error_rate`.

Each row must provide:

```json
{
  "verifier_metadata": {
    "id": "sample-id",
    "pair_id": "primevul-benchmark-0",
    "gold_is_vulnerable": true
  }
}
```

## Metrics

Per-row reward is binary classification accuracy. `/aggregate_metrics` additionally reports the
paper's four mutually exclusive pair outcomes, plus one of our own:

- `mean/paired_accuracy` (P-C): both members are correct; this is the headline metric.
- `mean/pairwise_vulnerable_rate` (P-V): both members are predicted vulnerable.
- `mean/pairwise_benign_rate` (P-B): both members are predicted benign.
- `mean/pairwise_reversed_rate` (P-R): both labels are reversed.
- `mean/pairwise_unanswered_rate`: at least one member produced no parsable verdict.

The last has no counterpart in the paper, whose setting had no unparseable replies. It exists
because P-V, P-B and P-R are claims about what a model *did*, and a pair with a missing verdict
supports no such claim — folding a truncated rollout into "reversed" would report reasoning the
model never performed. Unanswered pairs stay in the denominator, so `paired_accuracy` is unaffected
and a model cannot raise its score by declining to answer.

A rollout that was cut off before answering also sets `failure_reason` on its verify response
(`model response incomplete: max_output_tokens`). The reward is still 0, but the field marks it as
an infrastructure limit rather than a capability result. Reasoning models hit this readily: give
them a generous `--max-output-tokens`, since a small budget is spent entirely on chain of thought.

It also reports parse-error rate, binary accuracy, precision, recall, F1, confusion counts, pair
count, and rollout count. With repeated rollouts, pair metrics are computed at each explicit rollout
index and then averaged.

## Data and reproducibility

Preparation downloads the `paired` configuration of the `colin/PrimeVul` Hugging Face mirror at
immutable revision `4fd7158322872d711e90f091dbd8673ef32cb1be`. The canonical split is the upstream
`test` split: 435 vulnerable/fixed pairs, or 870 rows.

`benchmarks/primevul/data/primevul_example.jsonl` is synthetic smoke data authored for NeMo Gym;
it contains no third-party project source, which is why it is the only data file committed.
Preparation writes the canonical test split to the gitignored
`benchmarks/primevul/data/primevul_benchmark.jsonl`:

```bash
gym eval prepare --benchmark primevul
```

Preparation checks that every consecutive pair contains exactly one vulnerable and one fixed
function. `max_pairs` samples complete pairs reproducibly for local smoke runs.

## Usage

```bash
gym env validate primevul
gym env test primevul

gym env start --benchmark primevul --model-type openai_model

gym eval run --no-serve \
    --benchmark primevul \
    --output results/primevul.jsonl \
    --num-repeats 1
```

Use `--num-repeats N` when repeated sampling is desired; benchmark dataset metadata does not apply
that run-level option automatically. Avoid `--limit`, which can split pairs. Use preparation's
`max_pairs` argument for whole-pair subsets.

## Licensing and provenance

- NeMo Gym integration code and synthetic smoke data: Apache-2.0.
- PrimeVul repository and dataset release: MIT.
- Hugging Face mirror: declares MIT and is pinned above.

PrimeVul reconstructs functions from third-party C/C++ projects. Those functions retain their
original project licenses; the PrimeVul MIT grant covers the benchmark's curation and software, not
an independent relicensing of upstream project code. For that reason, canonical functions are
downloaded during preparation and are not committed to this repository.

The published prompt is adapted from PrimeVul's MIT-licensed `openai_expr/utils.py`.
