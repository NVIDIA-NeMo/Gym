# LeanCat

[LeanCat](https://github.com/sciencraft/LeanCat) ([arXiv:2512.24796](https://arxiv.org/abs/2512.24796)) is 100
statement-level problems in formal 1-category theory, written in Lean 4 against Mathlib v4.19.0. It is deliberately
not a search benchmark: the problems are curated from Riehl, Mac Lane, and Adámek et al., and solving them takes
navigating Mathlib's `CategoryTheory` interfaces rather than grinding out arithmetic. The published result is that
models are close to helpless at it — the best scored 12.0% pass@4, and 0.0% on the High tier.

Part I covers 1-categories; the authors flag higher categories as future work.

## Task format

Each row hands the model one self-contained Lean file — imports, an `open`/`variable` preamble, any auxiliary
definitions the problem needs, and a target theorem whose proof is `sorry` — and asks for the same file back with the
holes filled. The prompt is upstream's `prompts/static_passk.md`, unmodified — kept here as
`prompts/static-passk.md`, hyphenated only because Gym's `no-underscore-md` hook rejects the original name.

This is a **whole-file** task, which is what separates this server from `math_formal_lean`. There, the model writes a
proof body and the harness reassembles the file around it. Here the model returns the entire file, because many
LeanCat problems set up their own structures and instances before the statement and a reassembly step would have to
guess where the model's additions belong.

## Verification

Upstream calls a submission valid on five criteria (`EVALUATION.md`); this server enforces the four that are
mechanically checkable, in the order below. The first three are text-only and run **before** the sandbox call, so a
submission that has already lost does not cost a five-minute Mathlib compile.

| # | Check | Status on failure |
|---|---|---|
| 1 | A Lean code block was produced (last fenced block wins, as upstream) | `no_code` |
| 2 | No `sorry` / `admit` / `axiom` / `unsafe`, ignoring comments and strings | `banned_tokens` |
| 3 | The reference statement, assumptions and definitions are preserved | `statement_modified` |
| 4 | The file compiles clean under Lean 4.19.0 / Mathlib v4.19.0 | `compile_error`, `timeout`, `sandbox_error` |

`reward` is 1.0 only for `compiled`, and 0.0 otherwise.

The fifth upstream criterion, "maintained mathematical intent", is a human judgement and is not automated. Check 3 is
the closest mechanical proxy: the reference file is split on `sorry`, and every remaining fragment must appear in the
submission, in order, modulo whitespace and comments. That is exactly the condition "you filled the holes and changed
nothing else", and it generalises to the nine problems that carry more than one `sorry`. Preamble lines are matched
individually rather than as a block, so the model stays free to add imports and to insert auxiliary declarations —
which the prompt explicitly permits.

Check 3 exists because the whole-file format lets a model weaken the theorem it was asked to prove, and the weakened
version compiles. `require_statement_preserved: false` disables the rejection while still reporting
`statement_preserved` on every response, which is the way to measure how often the guard fires.

## Requirements

A Lean 4 sandbox on `sandbox_host:sandbox_port` exposing `POST /execute` with
`{generated_code, language, timeout, max_output_characters}` — the same
[NeMo-Skills sandbox](https://github.com/NVIDIA-NeMo/NeMo-Skills/blob/main/dockerfiles/Dockerfile.sandbox) that
`math_formal_lean` uses.

**The sandbox must be built on Mathlib v4.19.0.** LeanCat statements are written against that release's
`CategoryTheory` API. A sandbox on a different Mathlib will fail tasks for reasons that have nothing to do with the
model, and the failures look like ordinary compile errors, so this is worth confirming before trusting a number.

```yaml
sandbox_host: ${oc.env:NEMO_SKILLS_SANDBOX_HOST,127.0.0.1}
sandbox_port: ${oc.env:NEMO_SKILLS_SANDBOX_PORT,6000}
compilation_timeout: 300.0   # upstream's per-attempt budget
```

## Metrics

Pooled `pass@k` and `pass@1[avg-of-k]`, plus the same broken out by difficulty (`Easy/pass@4/accuracy`,
`Medium/…`, `High/…`). The split is the point of the paper's argument — the pooled number hides that High is a flat
zero. `statement_preserved` rides along as a second score so a run collapsing because the guard rejects everything is
visible without opening rollouts.

## Reproducing the paper

Table 1, pass@1 / pass@4, greedy-free sampling with a 50,000-token output budget and a 300s verification limit:

| Model | Easy | Medium | High | All |
|---|---|---|---|---|
| Claude Opus 4.5 | 32.50 / 50.00 | 4.17 / 4.76 | 0.00 / 0.00 | 8.25 / 12.00 |
| GPT-5.2 | 27.50 / 30.00 | 0.00 / 0.00 | 0.00 / 0.00 | 5.50 / 7.00 |
| DeepSeek Reasoner | 18.75 / 40.00 | 0.60 / 2.38 | 0.00 / 0.00 | 4.00 / 9.00 |
| Gemini 3 Pro | 11.25 / 25.00 | 2.38 / 7.14 | 0.00 / 0.00 | 3.25 / 8.00 |
| Kimi K2 | 10.00 / 20.00 | 0.00 / 0.00 | 0.00 / 0.00 | 2.00 / 4.00 |

LeanBridge — upstream's retrieve/generate/verify agent — roughly doubles the best number, to about 24%. It is not
implemented here; this server covers the static pass@k protocol only.

Two caveats before comparing against these:

1. **The difficulty split has moved.** The paper's denominators are Easy 20 / Medium 42 / High 38; the pinned dataset
   revision labels 20 / 40 / 40. The pooled `All` column is unaffected, but the per-tier columns are not directly
   comparable. Reproduce `All` first.
2. **Sampling temperature and top-p are not published.** The paper gives the token budget and the verification
   timeout but not the sampling settings, so per-tier agreement inside a point or two is the realistic target, not
   an exact match.

## Data

`prepare_leancat.py` fetches a pinned upstream revision and writes `data/train.jsonl` (100 rows) and
`data/example.jsonl` (5 rows). The pin is deliberate: another revision can change statements, difficulty labels, or
the prompt, none of which are detectable from the JSONL alone.

```bash
python prepare_leancat.py                    # fetch pinned revision, write data/
python prepare_leancat.py --records local.jsonl --prompt prompts/static-passk.md
```

## License

- Code: Apache-2.0
- LeanCat dataset: CC BY 4.0 — upstream evaluation code is MIT
