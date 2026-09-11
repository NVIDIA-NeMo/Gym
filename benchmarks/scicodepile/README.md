# SciCodePile Benchmark

Benchmark wrapper for the executable stratum of
[SciCodePile](https://arxiv.org/abs/2607.19104) — 200 runnable scientific-code
generation tasks harvested from real computational-science repositories.

- **Tasks**: 200, each with its own `check(candidate)` test
- **Reward**: binary; the task's test either passes or it does not
- **Metrics**: `pass@1` on `accuracy`

Verification lives in
[`resources_servers/scicodepile`](../../resources_servers/scicodepile/README.md),
which also documents the statuses and the validation evidence.

Not to be confused with the [`scicode`](../scicode/README.md) benchmark already in
this repository. They are different artifacts: `scicode` is 65 scientist-written
problems decomposed into 288 sub-steps from `SciCode1/SciCode`, solved one sub-step
at a time with a dedicated multi-step agent. SciCodePile is 200 single-shot function
generation tasks from a different source and a different construction process.

## Prepare benchmark data

```bash
gym eval prepare --benchmark scicodepile
```

Downloads `SciCodePile/SciCode-Runnable-Benchmark-Reviewed` and writes
`benchmarks/scicodepile/data/scicodepile_benchmark.jsonl`. The script asserts the
expected 200 rows so an upstream change surfaces as a loud failure rather than as an
unexplained score movement.

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark scicodepile
```

## Collect rollouts

```bash
gym eval run \
    --benchmark scicodepile \
    --model-type vllm_model \
    --split benchmark \
    --output results/scicodepile_rollouts.jsonl
```

Note `--split benchmark`: in NeMo Gym `--split` selects the dataset `type` declared
in the config, not a train/test split.

## Prompting

The upstream `prompt` field — a function signature plus docstring — is passed to the
model unmodified, HumanEval-style. Upstream publishes no prompt of its own, so any
instruction wrapper would be invention that changes what is measured, and would make
results incomparable with the published figures.

**This carries a cost worth knowing.** With no instruction, a chat model sometimes
answers conversationally instead of writing code — *"It looks like you've pasted a
docstring but not the implementation. If you want, I can…"* — which the extractor
cannot parse, so the task scores zero without the model having attempted it. An
instruction wrapper removes those but changes the character of the code produced,
so it is not a free improvement.

Non-attempts are separable from genuine failures: the statuses `no_code_block`,
`entry_point_missing`, and `error` with `details.reason == "syntax_error"` isolate
them, and any run should report them alongside the headline number.

## Data notes

Verified against all 200 released rows:

- Every row is `language: python`, `runnable: true`, `test_invalid: false`, and
  `primary_score_eligible: true`.
- `setup_code` is non-empty on 105 rows and runs before the model's code.
- Upstream `audit_flags` record test stability hazards: `env_sensitive` on all 200,
  `globals_patch` on 117, `internal_state_check` on 40. These are carried through to
  `verifier_metadata` for provenance. They are the reason each task runs in a fresh
  process.
- The `prompt` field is display text, not valid Python — its docstring is not
  indented under the `def` line. It therefore cannot be prepended to the model's
  output as a BigCodeBench-style calibration prefix, so the verifier requires the
  model's own code to define `entry_point`. Nothing in the prompt asks for that
  (see [Prompting](#prompting)); a bare function body scores
  `entry_point_missing`. See the resources server README.

## Metrics

`pass@1` on `accuracy` is the headline metric. Upstream reports the strongest
evaluated model at 12.30% Pass@1; that figure has not been reproduced here.
