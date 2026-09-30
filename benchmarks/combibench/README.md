# CombiBench

Lean 4 combinatorics benchmark bound to the
[`combibench`](../../resources_servers/combibench/) resources server and
`simple_agent` (single-turn whole-proof generation, matching upstream's
evaluation protocol).

- **Tasks**: 100 problems — 45 fill-in-the-blank (the model supplies the answer
  in an `abbrev <name>_solution` and proves the theorem about it) and 55
  proof-only — available in the paper's two settings, "without solution" (the
  default) and "with solution"; see "Settings" below.
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

## Settings

The paper evaluates the same 100 problems two ways. Each row carries its setting in
a `split` field, and one benchmark serves both:

| `split` | Setting | What the model must produce |
| --- | --- | --- |
| `test` (default) | **without solution** | the answer *and* the proof |
| `test_with_solution` | **with solution** | the proof only; the published answer is already in the statement |

The two differ only in the 45 fill-in-the-blank problems. For example,
`hackmath_1`:

```lean
-- test: the model must replace both `sorry`s, choosing the value of the abbrev itself
abbrev hackmath_1_solution : ℕ := sorry
theorem hackmath_1 ... : sols.card = hackmath_1_solution := by sorry

-- test_with_solution: the answer 1716 is given, only the proof is missing
theorem hackmath_1 ... : sols.card = ((1716) : ℕ ) := by sorry
```

- **Without solution** measures finding the answer as well as proving it. Besides
  the compile and statement-tamper checks, the verifier compares the model's
  `abbrev` value with the ground truth by `rfl`/`norm_num`.
- **With solution** measures the proof alone. There is no abbrev, so there is no
  answer check. The statement-tamper check still applies.
- The other 55 problems are proof-only and their statements are **identical** in
  both settings (verified: 55 of 100 statements equal, exactly the 45 that declare
  an `abbrev` differ). Preparing `both` therefore asks those 55 prompts twice.

Metrics are always reported per setting; nothing pools the two, because neither of
the paper's tables does.

## Preparation

```bash
gym eval prepare --benchmark combibench
```

Downloads the pinned repository tarball, joins each `.lean` statement with its
`metadata.csv` row, strips doc comments so the prompt shape matches the
published dataset, and writes `data/combibench.jsonl`. With no arguments that is
the 100 "without solution" rows; it fails closed on any other count. Choose the
setting with `split`:

```bash
# without solution (the default)
gym eval prepare --benchmark combibench +prepare_script_args.split=test
# with solution
gym eval prepare --benchmark combibench +prepare_script_args.split=test_with_solution
# both, 200 rows: `test` first, then `test_with_solution`
gym eval prepare --benchmark combibench +prepare_script_args.split=both
```

Every choice writes the same file, `data/combibench.jsonl`, which is the path
`config.yaml` reads. Preparing again replaces it, so re-prepare to switch setting;
`use_cached_prepared_benchmarks` skips preparation and keeps whichever setting is
already on disk. `--limit` applies to each setting, so a subset of `both` still
holds both.

Rows carry `theorem_name`, `formal_statement`, `answers` (list or null),
`natural_language`, `tag`, `source`, `split`, `dataset_source`,
`dataset_revision`; prompts are applied at rollout time.

```bash
# The published dataset instead of the repository files:
python benchmarks/combibench/prepare.py --source hf
```

## Which upstream copy

Upstream's harness loads the Hugging Face dataset, last updated 2025-07-13.
The repository's Lean files were corrected afterwards (statement fixes for
`hackmath_4`, `brualdi_ch4_35` and others; answer fixes for `brualdi_ch1_5`
and `brualdi_ch2_36`) and bumped to Lean v4.24.0 on 2025-11-11. With comments
stripped from both copies the way `prepare.py` strips them, **20 of the 100**
`test` statements differ byte for byte; **12** differ once whitespace is
normalised as well, i.e. 8 of the 20 differ only in spacing and line breaks.

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
    --input benchmarks/combibench/data/combibench.jsonl \
    --output results/combibench_rollouts.jsonl \
    --num-repeats 16 \
    --prompt-config benchmarks/combibench/prompt.yaml
```

The run scores whichever setting was prepared. The server reports metrics under the
row's `split`, so a file holding one setting still gets the plain `pass@16/accuracy`
keys, while a `both` file (3,200 rollouts at 16 samples) gets
`test/pass@16/accuracy` and `test_with_solution/pass@16/accuracy` (and
`<split>/<family>/...` for the source families). The headline `mean/reward` of a
`both` run pools the two settings and is not a figure the paper reports; read the
per-setting keys.

The paper reports Pass@1/8/16 from 16 samples per problem; it does not state
the temperature or token budget used, and upstream's shipped config defaults to
`temperature: 0`, `max_tokens: 2048`, `n: 1`. Report a comparison, not a
reproduction.

## Measured

Model scores live with the run that produced them, not here — see the pull request
that added this benchmark for the open-weights baseline, and the
[server README](../../resources_servers/combibench/README.md#harness-validation)
for the model-free harness validation.
