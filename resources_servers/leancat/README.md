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
| 1 | A Lean code block was produced (last fenced block wins, as upstream) | `empty_generation` |
| 2 | No `sorry` / `admit` / `axiom` / `unsafe`, ignoring comments and strings | `banned_tokens` |
| 3 | The reference statement, assumptions and definitions are preserved | `statement_modified` |
| 4 | The file compiles clean under Lean 4.19.0 / Mathlib v4.19.0 | `compile_error`, `timeout`, `sandbox_error` |

`reward` is 1.0 only for `completed`, and 0.0 otherwise.

Layout, field names and status vocabulary follow `math_formal_lean`, the repo's other Lean server, so the two read as
siblings: same module split (`app.py` / `proof_utils.py` / `sandbox_client.py` / `task_data.py`), same response fields
(`proof_status`, `predicted_proof`, `compiler_output`), same words for the outcomes that exist in both (`completed`,
`empty_generation`, `timeout`). `banned_tokens`, `statement_modified` and `compile_error` have no counterpart there —
that server reassembles the file itself, so it has nothing to catch a tampered statement.

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

Table 1, pass@1 / pass@4:

Row labels are the identifiers as Table 1 spells them, which for four of the five are API aliases rather than
pinned checkpoints.

| Model (Table 1 label) | Easy | Medium | High | All | Open weights |
|---|---|---|---|---|---|
| `claude-opus-4-5` | 32.50 / 50.00 | 4.17 / 4.76 | 0.00 / 0.00 | 8.25 / 12.00 | no |
| `gpt-5.2` | 27.50 / 30.00 | 0.00 / 0.00 | 0.00 / 0.00 | 5.50 / 7.00 | no |
| `deepseek-reasoner` | 18.75 / 40.00 | 0.60 / 2.38 | 0.00 / 0.00 | 4.00 / 9.00 | ambiguous, see below |
| `gemini-3-pro` | 11.25 / 25.00 | 2.38 / 7.14 | 0.00 / 0.00 | 3.25 / 8.00 | no |
| `kimi-k2-0905` | 10.00 / 20.00 | 0.00 / 0.00 | 0.00 / 0.00 | 2.00 / 4.00 | **yes, pinned** |

**`kimi-k2-0905` is the only row that names an exact open-weight release**, which makes it the one reproduction
target with no version ambiguity — worth more than its being second-cheapest rather than cheapest.

**`deepseek-reasoner` is not a checkpoint.** It is DeepSeek's API alias, and the paper's own prose disagrees with its
table: §3.1 names "DeepSeek-V3.2-Thinking and DeepSeek V3.2 Speciale" — two models — for what Table 1 reports as one
row. The alias resolved to DeepSeek-V3.2 in thinking mode around the paper's date, so `deepseek-ai/DeepSeek-V3.2`
with thinking enabled is the best available guess, but it is a guess. Do not treat a mismatch against this row as
evidence of a bug in this server.

LeanBridge — upstream's retrieve/generate/verify agent — roughly doubles the best number, to about 24%. It is not
implemented here; this server covers the static pass@k protocol only.

### Settings

The paper gives only the token budget and the verification timeout, but `scripts/passk.py` and
`eval_common.add_common_args` at the pinned commit fix the rest. Use these:

| Setting | Upstream default |
|---|---|
| `temperature` | **1.0** |
| `top_p` | not sent |
| `max_tokens` | 50000 |
| `k` | 4 |
| Lean timeout | 300s (`compilation_timeout`) |
| messages | one `user` message, no system prompt |

### Prompt fidelity

The prompt is byte-identical to what the reference harness sends, verified for all 100 problems against an
independent reimplementation of `passk.py`'s call chain (`load_prompt` → `load_problem` → `render_prompt`).
sha256 of the 100 concatenated rendered prompts: `e04d1313b51489083fe8153764e37a79…`.

This is why `prepare_leancat.py` reads `CAT_statement/S_<id>.lean` rather than taking `formal_statement` from
`leancat_records.jsonl`. Upstream's `load_problem` does `read_text()` with no `strip()`, and 60 of the 100 `.lean`
files end in a newline the JSONL has stripped — a newline that lands inside the prompt's code fence. Content is
identical between the two sources; the script fails loudly if that ever stops being true.

### Caveats

1. **The difficulty split has moved.** The paper's denominators are Easy 20 / Medium 42 / High 38; the pinned dataset
   revision labels 20 / 40 / 40. The pooled `All` column is unaffected, but the per-tier columns are not directly
   comparable. Reproduce `All` first.
2. **This server's statement guard is stricter than the scorer that produced the paper's numbers.** Upstream's
   `verify_lean` is `has_invalid_tokens(code)` then compile — it never compares the submission against the reference
   statement, despite `EVALUATION.md` listing preservation as a validity criterion. So a run here can score *below*
   the paper if models were weakening statements. Set `require_statement_preserved: false` to reproduce upstream's
   scoring exactly; the `statement_preserved` metric then tells you how much the difference is worth. Running both
   is the informative thing to do.
3. **pass@k must be read at k = number of rollouts.** Upstream's is plain "solved if any of 4 attempts passed"; Gym's
   `compute_pass_majority_metrics` uses the unbiased combinatorial estimator. The two agree exactly when
   n = k = 4, and diverge if you collect more rollouts and read `pass@4` off them.

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
