# CombiBench benchmark environment

[Dataset](https://huggingface.co/datasets/AI-MO/CombiBench),
[repository](https://github.com/MoonshotAI/CombiBench) (MIT) and
[paper](https://arxiv.org/abs/2505.03171), "CombiBench: Benchmarking LLM
Capability for Combinatorial Mathematics" (Liu et al., 2025).

One hundred combinatorics problems formalized in Lean 4. Fifty-five are proof-only:
the model replaces the `sorry` proof of a theorem. Forty-five are fill-in-the-blank:
the statement declares `abbrev <name>_solution : <type> := sorry` and a theorem about
it, and the model must supply both the answer and the proof. This server reproduces
upstream's **one-stage Fine-Eval** verdict: a Lean 4 compile through a Kimina Lean
Server plus the syntactic checks upstream applies around it.

Pinned upstream revision: GitHub `c67e4213597b1477351d9ef5ca37fb622084cc78` (Lean and
Mathlib `v4.24.0`). The Hugging Face dataset at `882ba08befd0856f5364db1e53d58c7e2cf704f9`
is the alternative source; the benchmark README explains why it is not the default.

## Scope

One benchmark, [`benchmarks/combibench`](../../benchmarks/combibench/), uses this server.
It prepares one of the paper's two settings per file (default `test`), told apart by each
row's `split`. They differ only in the 45 fill-in-the-blank problems: without solution the
model supplies the answers and the proof and the answer is checked; with solution the
published answer is already in the statement and only the proof is checked. The other 55
statements are identical in both.

| `split` | Setting | Rows |
| --- | --- | --- |
| `test` | "without solution": answer and proof both withheld | 100 |
| `test_with_solution` | "with solution": published answers substituted, proof withheld | 100 |

Not implemented: upstream's two-stage Fine-Eval, which asks the model for an `rfl`/`norm_num`
equality proof in a second turn when the filled-in answer is not proved equal to the ground
truth. The paper reports that in the without-solution setting "both evaluation methods ...
produced identical results". A row that would need the second stage scores 0 here.

## Prompting and data

The prompt is upstream's own (`evaluation/config/template.json5`): a system message and a
user message showing only the formal statement in a ```` ```lean4 ```` fence. The informal
statement is carried as `natural_language` but not shown. The strings are byte-identical; the
one difference is that `prepare.py` stores each statement with a trailing newline where
upstream strips it at render time. It cannot affect a verdict (the statement check strips
every chunk it compares). Decoding parameters and the token budget are not published, so no
run is a reproduction.

Rows are flat task fields; `prompt.yaml` builds the messages at rollout time:

```json
{
  "theorem_name": "hackmath_1",
  "formal_statement": "import Mathlib\n\nabbrev hackmath_1_solution : ℕ := sorry\n\ntheorem hackmath_1 ... := by sorry",
  "answers": ["1716"],
  "natural_language": "How many ways can a teacher select ...",
  "tag": "hackmath",
  "source": "https://www.hackmath.net/en/word-math-problems/combinatorics",
  "split": "test",
  "dataset_source": "github",
  "dataset_revision": "c67e4213597b1477351d9ef5ca37fb622084cc78"  # pragma: allowlist secret
}
```

`verify()` reads `formal_statement` and `answers`; `tag` groups the per-family metrics; the
rest is provenance.

## Scoring

`reward` is 1.0 iff every step passes, else 0.0. Steps 1–5 are upstream's rules
(`evaluation/util.py`, `evaluation/verifier/one_stage_verify.py`); `status` names the first
one that failed. Status names come from [`lean_proof/status.py`](../lean_proof/status.py)
where the concept is shared with `leancat`; CombiBench adds `format_error`, `code_too_long`,
`lean_error`, `header_timeout`, `model_header_timeout`, `model_header_error` and `bad_task`.

1. **Extract** the last ```` ```lean4 ```` block (falling back to ```` ```lean ````). None →
   `format_error`; empty output → `empty_generation`; longer than `max_code_characters`
   (200,000) → `code_too_long`.
2. **Remove comments**, so a model cannot hide the real theorem in one (paper, Appendix A.2).
3. **Prepend** upstream's default header only when the code does not start with an import.
4. **Forbid** the substrings `axiom` and `local_instance` → `banned_tokens`.
5. **Statement check.** Every non-header paragraph of the reference statement, with `sorry`
   removed, must appear verbatim in the code → `statement_modified`.
6. **Answer check.** For each `abbrev <name>_solution`, append
   `example : <name>_solution = (<gold> : <type>) := by try rfl; try norm_num`. Tags and
   answers are zipped positionally. If the statement declares `_solution` abbrevs and the
   counts disagree, the row is `bad_task` rather than a silently skipped check.
   `prepare.py::validate_rows` applies the same rule, so this only catches a hand-made row.
7. **Compile** through the Lean server with a 60 s timeout. Any error message →
   `compile_error`; a `sorry` warning or REPL `sorries` entry → `has_sorry`; the REPL timing
   out on the submission → `timeout`; any other REPL error, a 500 from executing the snippet,
   or a `{"message": ...}` payload → `lean_error`.

### Who a failure is charged to

Masking removes a rollout from `mean/reward` and pass@k, so the rule is conservative in one
direction: **a failure the model could have caused is scored 0, never masked.** Each masked
case below is one where this harness reached no verdict at all.

Charged to the model (reward 0.0, in the denominator):

| Status | When |
| --- | --- |
| `timeout` | the REPL hit the budget on the submission body; a non-terminating proof is the model's output |
| `lean_error` | the REPL reported an error string, `/verify` answered 500 from executing this snippet, or the reply carried a `{"message": ...}` Error object |
| `model_header_timeout` | a header the model wrote itself did not load inside the budget |
| `model_header_error` | a header the model wrote made the REPL die or write non-JSON while loading (an unresolvable module is not this case: it lands as `compile_error`) |

Charged to the harness (`harness_failure: 1.0`, `mask_sample: true`, out of the denominator,
reported under `coverage/`):

| Status | `failure_kind` | When |
| --- | --- | --- |
| `sandbox_error` | `provider_unavailable` | connection refused or timed out client-side; a non-5xx HTTP error (401, 404, 422); 500 "Failed to start REPL" or a harness-supplied header that failed to run; a 5xx Kimina never emits (502, 504: a proxy); 429/503 that survived three retries; a reply that is not JSON, has no `results`, has a result with neither `error` nor `response`, or carries an `error`/`stderr` object instead of a verdict |
| `header_timeout` | `provider_unavailable` | a cold REPL could not finish `import Mathlib` inside the timeout, and the header was the reference statement's or the default one |
| `bad_task` | `combibench:bad_task` | the row cannot be scored: no `formal_statement`, malformed `answers`, or an answer count that disagrees with the statement's `_solution` abbrevs |

Which header a failed header command belongs to is decided by re-deriving Kimina's own split
(`server/split.py`) for the submission and for the reference statement and comparing them
(`fine_eval.header_is_harness_supplied`). Kimina answers 500 for one snippet in several
shapes, told apart by the fixed `detail` strings `Failed to start REPL` and `Failed to run
header on REPL` (`lean_client.is_model_attributable_server_error`, `is_header_run_failure`).
This keys on server wording deliberately: if upstream reworded a marker, those cases would
become charged rather than silently masked, which is the direction that cannot inflate a
score. A 500 with no matching `detail` falls through to `lean_error`, for the same reason.

Every response carries `lean_version`, the version the server reports, because a server built
for another toolchain otherwise scores every row `compile_error` with nothing saying why. The
probe runs as a background task on the first compile and is never awaited, so a server that
accepts connections without answering produces the masked `sandbox_error` rather than a
run-wide stall; a probe that returns nothing is retried, up to three times per process.

`compute_metrics` emits pooled `pass@k` / `pass@1[avg-of-k]` and per-family pass rates keyed
on `tag` (`hackmath/`, `brualdi/`, `imo/`, `math_competitions/`); the pooled keys and
`mean/reward` are the headline and the per-family keys are supplementary. `get_key_metrics`
also promotes the `coverage/` block (`measured_rollouts`, `masked_rollouts`, `measured_tasks`,
`fully_masked_tasks`): masking keeps a harness fault out of `mean/reward`, so a Lean-server
outage shrinks the measured corpus instead of lowering the score. It is empty unless
something was masked.

### Five deliberate departures from upstream

1. **The gold answer is elaborated at the abbreviation's declared type.** Upstream's
   `example : imo_2014_p2_solution = fun n => ⌈√n⌉₊ - 1` reads the right-hand side with
   `n : ℝ` and fails against an `ℕ → ℕ` abbreviation, so `brualdi_ch8_6`, `imo_2014_p2`,
   `imo_2019_p5`, `imo_2022_p1` and `imo_2023_p5` can never score there. Ascribing the type
   makes the check elaborate for 45 of 45 problems instead of 40 of 45. Set
   `answer_check_ascription: false` for upstream's behaviour.
2. **Trailing whitespace is ignored in the statement check.** Some statements carry
   space-only lines left behind when comments were deleted (1 of 100 GitHub rows, 12 of 100
   Hugging Face rows). Lean ignores them, so a model that omits them has not changed what it
   proves. Set `normalize_trailing_whitespace: false` for upstream's behaviour. These counts
   were measured ad hoc, not from a committed report.
3. **A structurally reported `sorry` is rejected.** A non-empty REPL `sorries` list gives
   `has_sorry`, which upstream never reads. This is the one departure stricter than upstream,
   and it is not configurable.
4. **A `/verify` reply whose payload is an error object is never a success.** Kimina can
   return `{"message": ...}` in `response` with no top-level `error`, which looks exactly like
   a clean compile. `message` is charged (`lean_error`); `error` and `stderr` are masked
   (`sandbox_error`), since the pinned Kimina cannot produce them and they only guard against
   another server. A result with neither `error` nor `response` is likewise masked.
5. **Code longer than `max_code_characters` is rejected unsent** (`code_too_long`). It is a
   wire-safety bound, not a scoring rule, and it never fired in the 3,200 rollouts measured.

### Known blind spots, kept for fidelity

- `native_decide`, `decide` and `set_option maxHeartbeats` are allowed, as upstream allows them.
- The `axiom` ban is a substring test: an identifier containing `axiom` fails a valid proof,
  and `opaque`, `implemented_by` and `unsafe` are not banned.
- Extra declarations are allowed anywhere; only the reference paragraphs are required.
- The answer tag is the text between a paragraph's first `abbrev` and first `_solution`, which
  is upstream's rule verbatim. A paragraph with a helper `abbrev` beside the `_solution` one
  yields an invalid tag under upstream as much as here. No pinned paragraph is shaped that way.
- `answer_tags` returns a list where upstream keys a dict, so two paragraphs declaring the
  same `_solution` would collapse upstream and not here. The pinned corpus has 47 distinct
  abbreviations and no repeat.
- A paragraph opening with `open ... in` directly above its `theorem` is skipped by the
  statement check, as upstream skips it. No pinned paragraph is shaped that way.

## Harness validation

Model-free checks against Mathlib v4.24.0 through this server's code path. The reports are
not committed, so the numbers are stated here and re-derived with
`scripts/harness_validation.py` against a live Lean server (see "Re-running the validation").
Each report records the `dataset_source`, `dataset_revision`, `split` and
`answer_check_ascription` it was produced with.

| Check | GitHub source (default) | Hugging Face source |
| --- | --- | --- |
| Statement with its `sorry`s compiles (only `sorry` warnings) | **100 / 100** | 94 / 100 |
| "With solution" statement compiles | **100 / 100** | 94 / 100 |
| Published answer substituted + ascribed answer check elaborates | **45 / 45** | 41 / 45 |
| Same with upstream's unascribed check | 40 / 45 | 37 / 45 |

The six Hugging Face failures (`hackmath_6`, `imo_2008_p5`, `imo_2011_p2`, `imo_2021_p5`,
`imo_2022_p6`, `imo_2023_p5`) are statements upstream rewrote in the repository for the Lean
bump and never pushed to the dataset.

Negative controls through `verify()`, all 100 rows, every one scoring 0:

| Control | Denominator | Status observed |
| --- | --- | --- |
| Empty output | 100 | `empty_generation` |
| Statement echoed back with `sorry` | 100 | `has_sorry` (55 proof-only), `compile_error` (45 fill-in) |
| `axiom cheat : False` prepended | 100 | `banned_tokens` |
| Main theorem's goal replaced by `True` | 99 (one theorem is not the last declaration) | `statement_modified` |

Upstream publishes no reference proofs, so there is no gold-as-prediction check over the real
corpus. The five synthetic example problems have complete proofs in
`tests/fixtures/synthetic_solutions.json` and all five score 1.0 through the live server; one
answers `3 / 12` for gold `1 / 4`, exercising the `norm_num` path.

Model scores are not kept here, because they go stale; the open-weights baseline is in the
pull request that added this benchmark.

## Lean server

Scoring needs a [Kimina Lean Server](https://github.com/project-numina/kimina-lean-server)
(MIT), the REPL-pooling service upstream's own harness verifies through, at Lean and Mathlib
`v4.24.0`. The published image defaults to another Lean version, so
[`kimina_image/README.md`](kimina_image/README.md) has the Dockerfile, build, `docker run`
line, `LEAN_SERVER_*` settings and isolation caveats. Point this server at a running instance
with `COMBIBENCH_LEAN_SERVER_URL` (and `COMBIBENCH_LEAN_SERVER_API_KEY` if it has one).

- Keep `max_concurrent_lean_requests` equal to the server's `LEAN_SERVER_MAX_REPLS` (both 8
  in `kimina_image`; Kimina's own default depends on the host's CPU count). It bounds the
  whole resources server, so under `num_workers: N` each worker gets `1/N` of it.
- A compile that exhausts the client timeout is not retried. A 429 or 503 is, up to three
  attempts with a 1 s then 2 s backoff.
- The client's HTTP budget is `lean_server_max_wait + 2 * lean_timeout_seconds + 30 s`
  (`lean_client.http_budget_seconds`), so the server always answers first. A client-side
  timeout is a masked `sandbox_error`, so a shorter budget would delete a non-terminating
  proof from the denominator instead of charging it to the model. Set `lean_server_max_wait`
  if the server uses something other than 60.

### Shared with the other Lean benchmarks

Reused from [`lean_proof/`](../lean_proof/): `status.py` and `toolchain.py`'s
`TOOLCHAIN_PROBE` / `parse_lean_version`. Not reused, deliberately:

- **`proof_utils.py`.** Its extraction, banned tokens and whole-file statement check differ
  from upstream CombiBench's, so sharing them would change scores relative to published ones.
- **`lean_sandbox.py`.** It shells `lake env lean` once per request. CombiBench needs Kimina's
  header-keyed REPL pool, both because upstream's harness uses it and because a fresh
  `import Mathlib` per proof is unaffordable at 100 problems × 16 repeats.
  [`kimina_image/README.md`](kimina_image/README.md#why-not-the-existing-lean-sandbox) has
  the comparison.

## Quickstart

All commands run from the repository root with the Lean server up.

```bash
# 1. start servers (leave running)
COMBIBENCH_LEAN_SERVER_URL=http://127.0.0.1:12332 gym env start \
    --resources-server combibench \
    --model-type openai_model \
    --model "<model id>" \
    --model-url "<openai-compatible base url>" \
    --model-api-key "$API_KEY"

# 2. collect rollouts against them
gym eval run --no-serve \
    --agent combibench_simple_agent \
    --input resources_servers/combibench/data/example.jsonl \
    --output resources_servers/combibench/data/example_rollouts.jsonl
```

The committed `data/example.jsonl` is five hand-written synthetic problems (a natural-number,
a function and a rational fill-in answer, one proof-only statement, and one with a space-only
line), not benchmark rows. Regenerate it in three stages; the tracked file is the third's output:

```bash
DATA=resources_servers/combibench/data
python benchmarks/combibench/prepare.py --output $DATA/example_prepare.jsonl \
    --source-file resources_servers/combibench/tests/fixtures/synthetic_problems.json
gym dataset render --input $DATA/example_prepare.jsonl --output $DATA/example.jsonl \
    --prompt-config benchmarks/combibench/prompt.yaml
gym dataset collate --output-dir $DATA --mode example_validation \
    --config resources_servers/combibench/configs/combibench.yaml
```

### Re-running the validation

After any change to the scoring path, regenerate the reports and update the numbers above if
they moved. Prepared rows go to a scratch path, not `benchmarks/*/data/`, so the two sources
cannot overwrite each other.

```bash
URL=http://127.0.0.1:12332
V=resources_servers/combibench/scripts/harness_validation.py
D=/tmp/combibench_validation && mkdir -p $D

# one report per (source, split), plus a --no-ascription rerun of each `test` split
for SOURCE in github hf; do
  for SPLIT in test test_with_solution; do
    python benchmarks/combibench/prepare.py --source $SOURCE --split $SPLIT \
        --output /tmp/combibench_${SOURCE}_${SPLIT}.jsonl
    python $V --lean-server-url $URL --input /tmp/combibench_${SOURCE}_${SPLIT}.jsonl \
        --output $D/harness_validation_${SOURCE}_${SPLIT}.json
    [ "$SPLIT" = test ] && python $V --lean-server-url $URL --no-ascription \
        --input /tmp/combibench_${SOURCE}_${SPLIT}.jsonl \
        --output $D/harness_validation_${SOURCE}_${SPLIT}_upstream_check.json
  done
done

# the five synthetic problems, with their reference proofs
python $V --input /tmp/combibench_example_prepare.jsonl --lean-server-url $URL \
    --solutions resources_servers/combibench/tests/fixtures/synthetic_solutions.json \
    --output $D/harness_validation_example.json
```

### Agreement with upstream's harness

`scripts/upstream_agreement.py` downloads CombiBench at the pinned revision, imports
`evaluation/verifier/one_stage_verify.py` unmodified, and re-scores collected rollouts through
the same Lean server, reporting per-item agreement. It measures whether two verifiers agree,
not how good a model is. Over the 3,200 rollouts of the open-weights baseline in the pull
request (reports not committed):

| Setting | Rollouts | Scored by both | This verifier | Upstream | Agreement |
| --- | --- | --- | --- | --- | --- |
| `test` | 1600 | 1598 | 34 | 34 | **1598 / 1598** |
| `test_with_solution` | 1600 | 1600 | 31 | 31 | **1600 / 1600** |

The same rollouts pass under both. Two `test` rollouts were `sandbox_error` (no verdict from
this verifier) and are excluded from the denominator. That run predates the current
charging rules, so a rerun may score those two rows instead.

This is weak evidence for departures 1 and 3: 1566 of the 1600 `test` rollouts are rejections
both harnesses make for the same reason, and none of the five problems departure 1 affects was
solved by the profiled model. The evidence for departure 1 is the model-free 45/45 vs 40/45
measurement above.

**Upstream's harness needs a transport-level fix to run at all**, applied in the script and
nowhere else. It reads `res["error"]` and `res["response"]` by subscript, but Kimina omits
`error` when there was none and omits `response` when there was an `error`. The first read
raises `KeyError`, upstream's blanket `except Exception` reports "proof invalid", and every
compiling proof fails (0/1600 unpatched). The script now fills both keys. The figures above
were measured with only the first key filled, which does not change them: timeout rows have
`error` set, so upstream reaches "not a success" either way.

```bash
uv pip install loguru strenum tenacity tqdm   # upstream's imports, which Gym does not ship
python resources_servers/combibench/scripts/upstream_agreement.py \
    --rollouts results/combibench/rollouts.jsonl \
    --output /tmp/combibench_validation/upstream_agreement.json \
    --lean-server-url http://127.0.0.1:12332
```

The report keeps only disagreeing rows; pass `--full-rows` for the complete map. To measure
agreement with the two configurable departures removed, rescore the same rollouts with
`answer_check_ascription: false` and `normalize_trailing_whitespace: false` and pass that file
as `--rescore-with`; it must carry a verdict for every rollout compared.

## Tests

```bash
gym env test --resources-server combibench
```

## Licensing

Code: Apache 2.0. `fine_eval.py` re-derives the extraction, statement and answer rules of
upstream's MIT-licensed `evaluation/util.py` and `one_stage_verify.py` (Copyright (c) 2025
Moonshot AI and Project Numina); `benchmarks/combibench/prompt.yaml` reproduces the prompt
strings from the same repository's `evaluation/config/template.json5`.

CombiBench data: MIT, per the repository `LICENSE` and the dataset card. Upstream sources:
hackmath.net exercises, Brualdi's *Introductory Combinatorics*, IMO and other olympiad
problems; the IMO 2024 P3 and P5 statements are taken from Mathlib's `Archive`. No benchmark
rows are committed; `prepare.py` downloads them at run time.
