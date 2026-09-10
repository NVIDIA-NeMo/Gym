# LeanCat

[LeanCat](https://github.com/sciencraft/LeanCat) ([arXiv:2512.24796v2](https://arxiv.org/abs/2512.24796v2)) is 100
statement-level problems in formal 1-category theory, written in Lean 4 against Mathlib v4.19.0. It is deliberately
not a search benchmark: the problems are curated from Riehl, Mac Lane, and Adámek et al., and solving them takes
navigating Mathlib's `CategoryTheory` interfaces rather than grinding out arithmetic. The published result is that
models are close to helpless at it — the best static result is 12.0% pass@4, and 0.0% on the High tier.

Part I covers 1-categories; the authors flag higher categories as future work.

## Task format

Each row hands the model one self-contained Lean file — imports, an `open`/`variable` preamble, any auxiliary
definitions the problem needs, and a target theorem whose proof is `sorry` — and asks for the same file back with the
holes filled. The prompt is the paper's Appendix D.1 template, applied at run time from
`benchmarks/prompts/eval/leancat/paper.yaml` — see **Prompt** below.

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
individually rather than as a block, so the model stays free to add imports and to insert auxiliary declarations.
The paper's prompt neither permits nor forbids those explicitly, and upstream's own scorer allows them, so the guard
must not treat them as tampering.

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

**Read v2 ([arXiv:2512.24796v2](https://arxiv.org/abs/2512.24796v2), 25 Feb 2026), not v1.** v2 revised the Table 1
numbers, relabelled the models, added two tables, and — importantly for us — moved to the same difficulty
denominators and the same pass@1 estimator this server uses. Figures below are v2's.

### Table 1 — static pass@1 / pass@4 (the protocol this server implements)

| Model | Easy (20) | Medium (40) | High (40) | Overall (100) | Open weights |
|---|---|---|---|---|---|
| Claude-Opus-4.5 | 40.00 / 55.00 | 0.63 / 2.50 | 0.00 / 0.00 | 8.25 / 12.00 | no |
| GPT-5.2 | 27.50 / 35.00 | 0.00 / 0.00 | 0.00 / 0.00 | 5.50 / 7.00 | no |
| DeepSeek-V3.2 | 20.00 / 40.00 | 0.00 / 0.00 | 0.00 / 0.00 | 4.00 / 8.00 | **yes** |
| Gemini-3-Pro | 13.75 / 30.00 | 1.25 / 5.00 | 0.00 / 0.00 | 3.25 / 8.00 | no |
| Kimi-K2 | 10.00 / 20.00 | 0.00 / 0.00 | 0.00 / 0.00 | 2.00 / 4.00 | **yes** |

### Table 3 — specialized provers, **pass@32** (not pass@4)

Every one is open-weight and exactly pinned, which makes this the most reproducible table in the paper. Note the
8× sampling budget: comparing these against Table 1 numbers is comparing pass@32 against pass@4.

| Model | Easy | Medium | High | Avg |
|---|---|---|---|---|
| DeepSeek-Prover-V2-671B | 45.0 | 0.0 | 0.0 | 9.0 |
| Goedel-Prover-V2-32B | 20.0 | 2.5 | 0.0 | 5.0 |
| StepFun-Prover-32B | 20.0 | 2.5 | 0.0 | 5.0 |
| Kimina-72B | 10.0 | 0.0 | 0.0 | 2.0 |

Table 2 (not reproduced here) covers LeanBridge, the retrieve-generate-verify agent, at up to 4 refinement
iterations — Claude-Opus-4.5 reaches 16.0/21.0 overall. This server implements the static protocol only.

### Hyperparameters (Appendix D.3, Table 5)

| Parameter | Value |
|---|---|
| temperature | 1.0 |
| max tokens | 50,000 (generalist models; model-specific for provers) |
| top_p | not specified in the paper; `eval_common.chat_completion` sends none |
| k | 4 (generalist), 32 (specialized provers) |
| Lean version / timeout | v4.19.0 / 300s per attempt |
| messages | one message, no separate system role in the reference code |

### What lines up, and what does not

Three things that were caveats against v1 are **not** problems against v2:

- **Difficulty denominators match.** v2 uses Easy 20 / Medium 40 / High 40 — exactly the pinned dataset revision.
  (v1 used 42/38.) Every tier is directly comparable.
- **pass@1 estimator matches.** v2 states pass@1 is "estimated using the unbiased estimator from Chen (2021)",
  which is what `compute_pass_majority_metrics` computes. Collect 4 rollouts and read `pass@1/accuracy` and
  `pass@4/accuracy` directly.
- **Sampling settings are published**, in Appendix D.3 rather than the body.

One real divergence remains:

- **This server's statement guard is stricter than upstream's scorer.** `verify_lean` is `has_invalid_tokens(code)`
  then compile; it never compares against the reference statement, even though `configs/evaluation_protocol.json`
  sets `"statement_changes_allowed": false`. Set `require_statement_preserved: false` to match upstream exactly;
  `statement_preserved` reports the difference either way. Running both is the informative thing to do.

## Prompt

The repo and the paper do not ship the same prompt, so both are here, side by side:

| Template | Source | Runs by |
|---|---|---|
| `benchmarks/prompts/eval/leancat/paper.yaml` | the paper's Appendix D.1 | **default** — set in `benchmarks/leancat/config.yaml` |
| `benchmarks/prompts/eval/leancat/upstream-repo.yaml` | upstream's `prompts/static_passk.md` | `--prompt-config` |

```bash
gym eval run --benchmark leancat                                           # the paper's
gym eval run --benchmark leancat \
    --prompt-config benchmarks/prompts/eval/leancat/upstream-repo.yaml     # upstream's
```

**One dataset serves both.** Rows are flat (`formal_statement`, `level`, `problem_id`, …) with no
`responses_create_params`, the same shape `benchmarks/minif2f` uses, and the prompt is applied at run time by
`fill_prompt`. So switching templates is a flag, not a second dataset, and a per-problem diff of the two runs
isolates the prompt's contribution exactly. `{formal_statement}` is substituted from the row's top-level field,
which carries the pinned `CAT_statement/*.lean` bytes verbatim — trailing newline included — so the rendered prompt
is byte-identical to the reference harness's.

They live under `benchmarks/prompts/eval/` because they are benchmark-specific rather than reusable — contrast
`benchmarks/prompts/lean4/`, whose one template is shared by minif2f, proofnet and putnam_bench. Several variants in
one `eval/<benchmark>/` directory follows `benchmarks/prompts/eval/aai/`, which ships three.

### Why the paper's is the default

The two open identically; the fourth line differs:

| Source | Fourth line |
|---|---|
| upstream repo | "You may introduce auxiliary definitions, instances, and lemmas before the target statement if needed. The target statement and all auxiliary code must contain no `sorry`, `admit`, `axiom`, or `unsafe` declarations." |
| paper D.1 | "Please solve the statement step by step and provide your complete Lean4 code between ```` ```lean4 ```` and ```` ``` ```` after careful reasoning." |

We ran both with Goedel-Prover-V2-32B at pass@32. Upstream's prompt is not what Table 3's numbers were produced
under: 26% of submissions omitted `import Mathlib` entirely despite the instruction to include the complete header,
median output ran to 3015 tokens with visible degeneration, and Easy sat at 1/10 against a published 20.0%. The
paper's reproduces Table 3 — Easy 5/20, Medium 1/40, High 0/40, so 6.0% pass@32 against a published 5.0%. The repo
is a release mirror synced from a private development repo, so its prompt most likely post-dates the paper.

`upstream-repo.yaml` is a verbatim transcription of the pinned `prompts/static_passk.md` into Gym's prompt-config
form. `tests/test_app.py::TestPrompt::test_upstream_template_still_matches_upstream` refetches the pinned file and
compares, so the transcription is checked against the source itself rather than against a committed copy that could
drift with it. That test skips when there is no network.

### The paper's prompt cannot be transcribed byte-exactly — read this before quoting a number from it

`benchmarks/prompts/eval/leancat/paper.yaml` is a **reconstruction**, not a copy. The paper prints its prompt inside a LaTeX `lstlisting`,
and recovering a string from typeset output requires judgement. Taken from the arXiv v2 LaTeX source
(`main.tex`, Appendix D.1), the decisions were:

- **Dropped** the uniform 4-space listing indent on every line.
- **Dropped** the `"""` delimiters — Python string-literal markers showing it is a template, not prompt content.
- **Dropped** the extra 4-space indent on `{formal_statement}`. The placeholder is substituted with a whole Lean
  file beginning `import Mathlib`; indenting it would not be valid Lean, so the indent is listing cosmetics.
- **Kept** the printed line breaks verbatim, including the two that fall mid-sentence ("…so that your code can /
  pass the Lean4 compiler"). These are almost certainly page-width wrapping rather than real newlines, but
  preserving what is printed is the choice that does not silently improve on the source. Unwrapping them is a
  one-line change if you disagree, and no model will behave differently either way.
- **Stripped** trailing whitespace, and applied `.strip()` to the whole template, as `eval_common.load_prompt` does.

What is *not* in doubt is the part that matters: the paper's template asks for step-by-step reasoning inside a
`lean4` fence, and the repo's instead permits auxiliary declarations and names the banned tokens. That is a
substantive instruction difference, and it is what a comparison between the two runs actually measures.

## Data

LeanCat is 100 held-out evaluation problems with **no train split**, so this server declares only an `example`
dataset — matching `scicode`, `polymath` and `critpt`. `prepare.py` fetches a pinned upstream revision and writes
`data/example.jsonl` (5 rows, committed, what the environment gate requires); `benchmarks/leancat/prepare.py` writes
the 100-row benchmark JSONL from the same `build_rows`, gitignored and regenerated. The pin is deliberate: another
revision can change statements or difficulty labels, neither of which is detectable from the JSONL alone.

Tests read the committed 5 rows, as the repo's other server tests do. Nothing gates on the gitignored 100, which a
CI checkout never has — such a test would skip on every PR and report green without running.

```bash
python prepare.py                    # fetch pinned revision, write data/
python prepare.py --records local.jsonl
```

## License

- Code: Apache-2.0
- LeanCat dataset: CC BY 4.0 — upstream evaluation code is MIT

## Running it

### 1. Install Lean + Mathlib v4.19.0 — no container build needed

**You do not need a Lean-specific `.sqsh`.** There is no published image at Mathlib v4.19.0
(`leanprovercommunity/mathlib` ships only `latest`/`gitpod`/`debian`), the NeMo-Skills sandbox pins **v4.12.0**
(measured below), and building a correct image needs Docker, which HPC login nodes generally lack.

#### Measured: what a v4.12.0 sandbox actually costs you

Compiling all 100 **reference statements** (unmodified, `sorry` intact) inside NeMo-Skills'
`nemo-skills-sandbox-latest` — Lean/Mathlib v4.12.0:

| Tier | Compile | Hard error |
|---|---:|---:|
| Easy | 16 | 4 |
| Medium | 29 | 11 |
| High | 19 | 21 |
| **Total** | **64** | **36** |

So a v4.12.0 sandbox does not fail outright — it silently caps the score at 64/100 and skews it by difficulty, with
High hit hardest. Those 36 problems return 0 no matter what the model writes. The failures are genuine Mathlib API
drift (`invalid field 'carrier' … 'Grp.carrier'`, `invalid field 'IsRepresentable'`, `ambiguous, possible
interpretations`, `function expected at`) and are indistinguishable at a glance from a model that simply could not
prove the theorem.

That makes it usable as a plumbing smoke test and useless for a number. `check_sandbox.py` is what tells the two
apart, which is why it runs before anything else.

`elan` installs entirely in user space, so none of that is required:

```bash
./setup_lean.sh /lustre/<...>/lean4-mathlib-v4.19.0
```

`lake exe cache get` downloads prebuilt Mathlib oleans because `v4.19.0` is a tagged release, so this is a large
download rather than a multi-hour source build. Bind-mount the result into whatever base image you already have.

### 2. Verify the sandbox before spending anything on inference

```bash
python check_sandbox.py --host <node> --port 6000
```

Compiles all 100 reference statements **unmodified**. Each still contains its `sorry`, so each must come back with a
"declaration uses 'sorry'" warning and no errors. No model, no GPU. If this fails, nothing downstream is meaningful.

### 3. Prepare and run

```bash
gym eval prepare --benchmark leancat
gym eval submit --config examples/slurm_leancat_goedel_prover.yaml --dry-run
gym eval submit --config examples/slurm_leancat_goedel_prover.yaml
```

`benchmarks/leancat/` is registered, so `gym list benchmarks` shows it and `--benchmark leancat` works.
`num_repeats: 32` matches Table 3's specialized-prover budget; drop it to 4 for the generalist protocol of Table 1.

The submit config cannot start the sandbox — `services:` accepts only `type: vllm` and `type: ray`, so the sandbox
must already be reachable at `NEMO_SKILLS_SANDBOX_HOST:PORT`, launched into the same allocation with
`srun --overlap`.

## Two verification backends

| | HTTP sandbox (default) | Gym sandbox (`configs/leancat_enroot.yaml`) |
|---|---|---|
| Selected by | `sandbox_host`/`sandbox_port` | presence of `sandbox_provider` |
| Runs | NeMo-Skills `/execute` | `lake env lean <file>`, as upstream does |
| Who starts it | you, out of band | this server, lazily |
| On Slurm | a second `srun --overlap` | nothing extra |
| Parity | matches `math_formal_lean` | matches `swebench`/`deepswe`/`litmus_agent` |

The Gym backend exists because `gym eval submit` cannot start an HTTP sandbox: `ServiceConfig` is a closed union of
`vllm` and `ray`, with no generic container service. Routing through `nemo_gym.sandbox` lets the resources server own
its sandbox, which is what makes a one-command Slurm run possible. Providers available: `enroot` (HPC-native),
`apptainer`, `local`, `docker`, `e2b`, and others under `nemo_gym/sandbox/providers/`.

The Lean file is shipped into the sandbox base64-encoded rather than interpolated into a shell command — Lean sources
routinely contain quotes, backslashes and unicode, and a heredoc delimiter can appear inside a proof.

`reward` is 1.0 only when `lake env lean` **exits zero**. Matching the output for `error:` is not enough on its own:
a non-zero exit with nothing matching would otherwise score as a proof.
