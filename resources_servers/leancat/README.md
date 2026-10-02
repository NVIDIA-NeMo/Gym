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

This is a **whole-file** task: the model returns the entire file rather than a proof body the harness reassembles a
file around. Many LeanCat problems set up their own structures and instances before the statement, so a reassembly
step would have to guess where the model's additions belong.

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

Response fields (`proof_status`, `predicted_proof`, `compiler_output`) and the status vocabulary come from
`resources_servers/lean_proof`, so a rollout dump reads the same across Lean benchmarks.

Everything not specific to LeanCat is **imported** from `resources_servers/lean_proof`: the sandbox runner, the
`CompilerOutput` model, the Mathlib version probe, and the text checks.

The fifth upstream criterion, "maintained mathematical intent", is a human judgement and is not automated. Check 3 is
a mechanical proxy, not a decision procedure: the reference file is split on `sorry`, and every remaining fragment
must appear in the submission, in order, modulo whitespace and comments. It generalises to the nine problems that
carry more than one `sorry`. Preamble lines are matched individually rather than as a block, so the model stays free
to add imports and auxiliary declarations — the paper's prompt permits them and upstream's scorer allows them.

**What it catches and what it does not.** It is a *textual* check, so it catches edits to the statement, the
assumptions or the preamble — the case measured here, where a submission weakened a theorem and compiled. It does
**not** catch a submission that leaves every fragment byte-identical while changing what the statement means through
additions the check permits: a `variable (h : False)` plus `include h`, or a local `notation` rebinding a symbol the
statement uses, both keep the text intact, carry no banned tokens, compile, and would score 1.0. Closing that needs
the elaborated statement compared in the sandbox (`#check`/`#print axioms`) rather than its text; until then, treat
`statement_preserved` as evidence that the text is unchanged, not proof that the theorem is.

Check 3 exists because the whole-file format lets a model weaken the theorem it was asked to prove, and the weakened
version compiles. The benchmark ships it **reporting but not rejecting** (`require_statement_preserved: false`), so
scoring matches the upstream script the published numbers came from; see "What lines up, and what does not" below.

## Requirements

A Lean 4.19.0 / Mathlib v4.19.0 sandbox image, built by
[`resources_servers/lean_proof/lean_image`](../lean_proof/lean_image):

```bash
cd ../lean_proof/lean_image && ./build.sh v4.19.0
```

Verification runs through `nemo_gym.sandbox`, so any provider works — OpenSandbox, enroot,
docker — and nothing has to be started out of band.

**Isolation depends on the provider.** Submitted Lean can execute code while it compiles (`#eval`), so compiles run as
the unprivileged `lean` user the image creates, with a root-owned toolchain. That is enforced on OpenSandbox and docker.
Enroot maps a single uid, so there `su` keeps uid 0 and the toolchain is writable; the server logs a warning. Use enroot
for trusted-model runs such as reproducing the paper, and OpenSandbox (with `provider_options.network_policy` for
network isolation) for untrusted models or training.

**The Mathlib version must be exactly v4.19.0.** LeanCat's statements are written against that release's
`CategoryTheory` API, and a different Mathlib fails tasks for reasons that have nothing to do with the model: on
v4.12.0, 36 of the 100 reference statements fail to compile **with their `sorry` still intact**, so they score 0
whatever the model writes — a silent cap at 64/100, skewed by difficulty (see the table below).

Two checks guard it:

- **`check_sandbox.py`** compiles all 100 reference statements and fails unless every one comes back with only a
  `sorry` warning. Run it before spending anything on inference.
- **The server** probes `Lean.versionString` once on the first `verify` and logs an `ERROR` on a mismatch with the
  row's `lean_toolchain` (falling back to `expected_lean_version`). It logs rather than raises, so a run already in
  flight is not killed; set `check_lean_version: false` to skip it.

```yaml
sandbox_provider: sandbox            # any nemo_gym.sandbox provider config
sandbox_config:
  image: ${oc.env:LEANCAT_SANDBOX_IMAGE,gym-lean:v4.19.0}
compilation_timeout: 300.0           # upstream's per-attempt budget
check_lean_version: true
expected_lean_version: "4.19.0"
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
| ↳ *measured here, run 1* | *13.75 / 40.00* | *0.00 / 0.00* | *0.00 / 0.00* | *2.75 / 8.00* | |
| ↳ *measured here, run 2* | *15.00 / 30.00* | *0.00 / 0.00* | *0.00 / 0.00* | *3.00 / 6.00* | |
| Gemini-3-Pro | 13.75 / 30.00 | 1.25 / 5.00 | 0.00 / 0.00 | 3.25 / 8.00 | no |
| Kimi-K2 | 10.00 / 20.00 | 0.00 / 0.00 | 0.00 / 0.00 | 2.00 / 4.00 | **yes** |

The two measured rows are **two independent 400-rollout runs at `k=4`, reported separately**, both scored under
the full `EVALUATION.md` criteria. They are not pooled: upstream's reporting guidance is "a task is counted as
solved if at least one of the k independent attempts satisfies the valid proof criteria", which is what a single
`k=4` run measures. Pooling to `n=8` and estimating `pass@4` from it would be a different protocol.

In raw counts, which is the unit upstream asks for:

| | Easy (20) | Medium (40) | High (40) | Overall (100) |
|---|---|---|---|---|
| paper | 4 | 0 | 0 | **8** |
| run 1 | 8 | 0 | 0 | **8** |
| run 2 | 6 | 0 | 0 | **6** |

**Run 1 reproduces the published count exactly**; run 2 lands two tasks lower. Medium and High are a flat zero in
both, which is the paper's central claim. Both runs solve more Easy tasks than the paper (8 and 6 against 4) while
landing at or below it overall, so the tier profile is shifted toward Easy even where the total matches.

Two things worth knowing before quoting these:

- **Two tasks is the whole spread between the runs**, and upstream's own guidance notes one High task is 2.5
  percentage points. With 6–8 tasks solved out of 100, percentage differences between runs are one or two tasks
  moving. Report counts, and do not read a trend into a single run.
- **The statement criterion is not cosmetic on High.** The single non-Easy compile in 800 rollouts — problem
  `0064`, once — altered the statement. Under upstream's shipped script, which never checks this, it would show as
  a High solve and contradict the paper's headline claim; under the documented criteria it is correctly rejected.
  This is the concrete reason the guard exists.

Reproduced with `deepseek-ai/DeepSeek-V3.2` served on vLLM 0.19.1 (TP=16, fp8, `--reasoning-parser deepseek_v3`),
thinking enabled per request. Note the paper's model is **DeepSeek-V3.2-Thinking** (§Standard Baselines); the table
abbreviates it. A third run in which the `thinking` kwarg never reached the chat template solves **3 of 100 tasks**
against the 8 and 6 above, so the mode is not optional.

100% of rollouts in both runs carry a reasoning trace, against 0% in the non-thinking run — so the check is a clean
discriminator, and `leancat-multinode.sub` prints it per run. Look for `output` items of `type: "reasoning"`:
`response.reasoning` and `usage.output_tokens_details.reasoning_tokens` are null even when thinking is on, so a
check reading only those fields will wrongly report a thinking run as non-thinking. Output length corroborates:
mean 14.7k and 15.0k output tokens for the two runs above, against 2.9k for the non-thinking one.

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

- **Upstream's document and upstream's code disagree.** `EVALUATION.md` requires the statement, definitions and
  assumptions to be unchanged, and `configs/evaluation_protocol.json` sets `"statement_changes_allowed": false` --
  but `verify_lean` is `has_invalid_tokens(code)` then compile, and never compares against the reference statement.
  No setting matches both.

  **The benchmark ships `require_statement_preserved: false`, following the code that produced the published
  numbers**, since reproducing them is what it is for. `statement_preserved` is reported on every response, so the
  documented criterion can be applied afterwards without rerunning: measured over 16,800 rollouts, 7 submissions
  compiled clean but altered the statement, and applying it costs at most one task per run. Set the flag true to
  enforce it during the run instead.

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
`responses_create_params`, and the prompt is applied at run time by
`fill_prompt`. So switching templates is a flag, not a second dataset, and a per-problem diff of the two runs
isolates the prompt's contribution exactly. `{formal_statement}` is substituted from the row's top-level field,
which carries the pinned `CAT_statement/*.lean` bytes verbatim — trailing newline included — so the rendered prompt
is byte-identical to the reference harness's.

They live under `benchmarks/prompts/eval/` because they are benchmark-specific rather than reusable.

### Why the paper's is the default

The two open identically; the fourth line differs:

| Source | Fourth line |
|---|---|
| upstream repo | "You may introduce auxiliary definitions, instances, and lemmas before the target statement if needed. The target statement and all auxiliary code must contain no `sorry`, `admit`, `axiom`, or `unsafe` declarations." |
| paper D.1 | "Please solve the statement step by step and provide your complete Lean4 code between ```` ```lean4 ```` and ```` ``` ```` after careful reasoning." |

We ran both with Goedel-Prover-V2-32B at pass@32, full 3200-rollout runs each. The paper's prompt reproduces
Table 3 across four runs — 5, 5, 5 and 4 tasks solved against a published 5 — while the upstream-repo prompt gives
4. That is a one-task difference on a benchmark where only 3–5 tasks are ever solved, so it is weak evidence on its
own; the reason the paper's is the default is that it is the prompt the published numbers were produced under. The
repo is a release mirror synced from a private development repo, so its prompt most likely post-dates the paper.

An earlier note here claimed the upstream prompt collapsed (imports omitted, degenerate output, Easy at 1/10). The
full runs do not support it: the upstream prompt omits `import Mathlib` in 27.8% of submissions against 46.7% for
the paper's, runs shorter (median 6,959 output tokens against 11,212), and reaches Easy 4/20 — the published
count. Whatever that observation came from, it was not a full run, and it should not be cited.

`upstream-repo.yaml` is a verbatim transcription of the pinned `prompts/static_passk.md` into Gym's prompt-config
form. It is **not** checked automatically: a test that refetched the pinned file would need network egress and would
skip on every CI run, so it was dropped rather than kept as a check that never executes. To verify the
transcription by hand, diff it against the source:

```bash
curl -s https://raw.githubusercontent.com/sciencraft/LeanCat/4e136a13e5/prompts/static_passk.md
```

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

### 1. Build the Lean image

```bash
cd ../lean_proof/lean_image
./build.sh v4.19.0                      # local
./build.sh v4.19.0 <registry>/gym-lean  # and push, prints the digest to pin
```

One image per Mathlib version; the pins live in `versions.json`. See that directory's README for what is pinned
and why the build fails loudly rather than producing a subtly wrong image.

### 2. Gate it before spending anything on inference

```bash
python check_sandbox.py --image <ref>            # 100/100 required
python check_sandbox.py --provider enroot --image /path/to/gym-lean-v4.19.0.sqsh
```

Compiles all 100 reference statements **unmodified**. Each still contains its `sorry`, so each must come back with a
"declaration uses 'sorry'" warning and no errors. No model, no GPU. If this fails, nothing downstream is meaningful.

### 3. Prepare and run

```bash
gym eval prepare --benchmark leancat
gym eval run --benchmark leancat
```

`num_repeats: 4` is the generalist budget of Table 1, and one of upstream's `recommended_k_values`. For Table 3's
specialized provers, raise it to 32 with `--num-repeats`.

## Verification backend

Verification goes through `nemo_gym.sandbox`: one sandbox per server process, created on the first `verify` and
reused, with each attempt compiled by `lake env lean` — which is what upstream's `verify_lean` does. A sandbox per
rollout is not viable, since pod allocation costs minutes and a run is thousands of rollouts. There is no server
shutdown hook, so `sandbox_config.ttl_s` is what reclaims the sandbox if the process dies.

The Lean file is written in through a heredoc rather than interpolated into the command, so quotes, backslashes and
unicode in a proof need no escaping.

`reward` is 1.0 only when Lean exits zero **and** the output carries neither `error:` nor a `sorry` warning. The
output scan is needed because a warning-only build that declared a `sorry` still exits zero.
