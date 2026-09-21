# CombiBench

Lean 4 combinatorics benchmark bound to the
[`combibench`](../../resources_servers/combibench/) resources server and
`simple_agent` (single-turn whole-proof generation, matching upstream's
evaluation protocol).

- **Tasks**: 100 problems — 45 fill-in-the-blank (the model supplies the answer
  in an `abbrev <name>_solution` and proves the theorem about it) and 55
  proof-only.
- **Source**: [MoonshotAI/CombiBench](https://github.com/MoonshotAI/CombiBench)
  `lean/CombiBench/*.lean` + `metadata.csv` at `c67e4213597b1477351d9ef5ca37fb622084cc78`
  (MIT). The Hugging Face dataset `AI-MO/CombiBench` at `882ba08b` is available
  via `--source hf`; see "Which upstream copy" below for why it is not the default.
- **Paper**: [arXiv:2505.03171](https://arxiv.org/abs/2505.03171).
- **Prompt**: [`prompt.yaml`](prompt.yaml), byte-identical to upstream's
  `evaluation/config/template.json5` system prompt and user template. Only the
  formal statement is shown; the informal statement is not.
- **Reward**: binary; 1.0 iff upstream's one-stage Fine-Eval accepts the output
  (see the server README for the exact rules).
- **Lean**: the server needs a Kimina Lean Server built for Lean/Mathlib
  v4.24.0, the toolchain upstream pins. See the server README.

This directory is the paper's **"without solution"** setting. The
**"with solution"** setting, where the published answers are already
substituted into the statements, is [`benchmarks/combibench_with_solution`](../combibench_with_solution/).

## Preparation

```bash
gym eval prepare --benchmark combibench
```

Downloads the pinned repository tarball, joins each `.lean` statement with its
`metadata.csv` row, strips doc comments so the prompt shape matches the
published dataset, and writes `data/combibench_test.jsonl` (100 rows, fails
closed on any other count). Rows carry `theorem_name`, `formal_statement`,
`answers` (list or null), `natural_language`, `tag`, `source`, `split`,
`dataset_source`, `dataset_revision`; prompts are applied at rollout time.

```bash
# The published dataset instead of the repository files:
python benchmarks/combibench/prepare.py --source hf
```

## Which upstream copy

Upstream's harness loads the Hugging Face dataset, last updated 2025-07-13.
The repository's Lean files were corrected afterwards (statement fixes for
`hackmath_4`, `brualdi_ch4_35` and others; answer fixes for `brualdi_ch1_5`
and `brualdi_ch2_36`) and bumped to Lean v4.24.0 on 2025-11-11. Ignoring
comments, 12 of the 100 `test` statements differ between the two copies.

Measured against Mathlib v4.24.0 with the statements' `sorry`s in place:

| Source | Statements that compile | Failing |
| --- | --- | --- |
| GitHub `c67e4213` | 100 / 100 | — |
| Hugging Face `882ba08b` | 94 / 100 | `hackmath_6`, `imo_2008_p5`, `imo_2011_p2`, `imo_2021_p5`, `imo_2022_p6`, `imo_2023_p5` |

A statement that does not compile cannot be solved by any model, and the
failure would be charged to the model, so the repository files are the default.
The rows record which copy they came from.

## Running

```bash
COMBIBENCH_LEAN_SERVER_URL=http://127.0.0.1:12332 gym env start \
    --model-type vllm_model \
    --benchmark combibench
```

```bash
gym eval run --no-serve \
    --agent combibench_agent \
    --input benchmarks/combibench/data/combibench_test.jsonl \
    --output results/combibench_rollouts.jsonl \
    --num-repeats 16 \
    --prompt-config benchmarks/combibench/prompt.yaml
```

The paper reports Pass@1/8/16 from 16 samples per problem; it does not state
the temperature or token budget used, and upstream's shipped config defaults to
`temperature: 0`, `max_tokens: 2048`, `n: 1`. Report a comparison, not a
reproduction.
