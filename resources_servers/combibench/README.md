# CombiBench benchmark environment

[Dataset](https://huggingface.co/datasets/AI-MO/CombiBench) and
[repository](https://github.com/MoonshotAI/CombiBench) (MIT) and
[paper](https://arxiv.org/abs/2505.03171), "CombiBench: Benchmarking LLM
Capability for Combinatorial Mathematics" (Liu et al., 2025).

One hundred combinatorics problems formalized in Lean 4, from middle-school
exercises to IMO. Fifty-five are proof-only: the model replaces the `sorry`
proof of a theorem. Forty-five are fill-in-the-blank: the statement declares
`abbrev <name>_solution : <type> := sorry` and a theorem about it, and the model
must supply both the answer and the proof. This server reproduces upstream's
**one-stage Fine-Eval** verdict: a Lean 4 compile through a Kimina Lean Server
plus the syntactic checks upstream applies around it.

Pinned upstream revision: GitHub `c67e4213597b1477351d9ef5ca37fb622084cc78`
(Lean and Mathlib `v4.24.0`). The Hugging Face dataset revision
`882ba08befd0856f5364db1e53d58c7e2cf704f9` is the alternative source; the
benchmark README explains why it is not the default.

## Scope

One benchmark, [`benchmarks/combibench`](../../benchmarks/combibench/), uses this server. It
prepares one of the paper's settings per file (default `test`), told apart by each row's
`split`. They differ only in the 45 fill-in-the-blank problems: without solution the model
supplies the `abbrev <name>_solution` answers and the proof, and the answer is checked; with
solution the published answer is already in the statement and only the proof is checked. The
other 55 (proof-only) statements are identical in both.

| `split` | Setting | Rows |
| --- | --- | --- |
| `test` | "without solution": answer and proof both withheld | 100 |
| `test_with_solution` | "with solution": published answers substituted, proof withheld | 100 |

Upstream also ships a two-stage Fine-Eval that, when the filled-in answer is not
proved equal to the ground truth by `rfl`/`norm_num`, asks the model for that
equality proof in a second turn. It is not implemented; the paper reports that
in the without-solution setting "both evaluation methods ... produced identical
results". A row that would need the second stage scores 0 here.

## Prompting

Upstream's own prompt from `evaluation/config/template.json5`: a system message
("You are an expert in mathematics and proving theorems in Lean 4.") and a user
message that shows only the formal statement inside a ```` ```lean4 ```` fence.
The informal statement is carried in the data as `natural_language` but is not
shown, matching upstream. The prompt strings are byte-identical; the rendered
statement is not quite, and deliberately so — `prepare.py` stores each statement
with a trailing newline where upstream strips it at render time, so the fence
closes one blank line later here. It cannot affect a verdict (the statement check
strips every chunk it compares) and the stored rows are what all the committed
evidence was measured against, so the difference is recorded rather than removed;
see `prepare.py`'s module docstring. Runs are comparable to the paper's protocol
in prompt and endpoint style (chat); decoding parameters and the token budget are
not published, so no run is a reproduction.

## Dataset format

Rows are flat task fields; `prompt.yaml` builds the messages at rollout time.

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

`formal_statement` and `answers` are read by `verify()`. `tag` groups the
per-family metrics. `natural_language`, `source`, `split`, `dataset_source`
and `dataset_revision` are provenance and are never read. The tracked
`data/example.jsonl` carries `responses_create_params.input` already rendered
and a `task_source` key, because `gym dataset render` and `gym dataset collate`
rewrite the rows; the task fields are identical.

## Scoring

`reward` is 1.0 iff every step below passes, else 0.0. Steps 1–5 are upstream's
rules (`evaluation/util.py`, `evaluation/verifier/one_stage_verify.py`); the
`status` field names the first one that failed.

Status names come from [`lean_proof/status.py`](../lean_proof/status.py)
wherever the concept is shared, so `completed`, `empty_generation`,
`banned_tokens`, `statement_modified`, `compile_error`, `has_sorry`, `timeout`
and `sandbox_error` mean the same thing here as in `leancat`. CombiBench adds
`format_error`, `code_too_long`, `lean_error`, `header_timeout`,
`model_header_timeout`, `model_header_error` and `bad_task` next to them, which
is the extension that module describes: upstream's Fine-Eval distinguishes
outcomes a whole-file benchmark has no equivalent for.

1. **Extract** the last ```` ```lean4 ```` block (falling back to ```` ```lean ````);
   none → `format_error`; empty output → `empty_generation`; a block longer than
   `max_code_characters` (200,000; the longest pinned statement is 3,054) →
   `code_too_long`, so nothing pathological is put on the wire.
2. **Remove comments.** The paper's Appendix A.2 shows a model passing the Lean
   check by hiding the real theorem in a comment; stripping comments first is
   what defeats it.
3. **Prepend** upstream's default header (`import Mathlib`, `import Aesop`,
   `set_option maxHeartbeats 0`, `open BigOperators Real Nat Topology Rat`) only
   when the code does not start with an import.
4. **Forbid** the substrings `axiom` and `local_instance` → `banned_tokens`.
5. **Statement check.** Every non-header paragraph of the reference statement,
   with `sorry` removed, must appear verbatim in the code → `statement_modified`.
   This is the guard against proving a weaker theorem.
6. **Answer check.** For each `abbrev <name>_solution`, append
   `example : <name>_solution = (<gold> : <type>) := by try rfl; try norm_num`.
   Tags and answers are zipped positionally and a surplus on either side is
   dropped, as upstream does. Whenever the statement declares at least one
   `_solution` abbrev, a disagreement between the two counts is a `bad_task`
   instead of a silently skipped check: the answer is what the model has to
   supply, so a truncated row could otherwise score 1.0 with the wrong answer
   filled in. The guard is keyed on the statement's own tags rather than on
   `split`, which is an untyped, defaulted field a malformed row can simply omit
   — the case the guard exists for. `test_with_solution` rows declare no
   `_solution` abbrev (the answer is already substituted), so they never reach
   it. `prepare.py::validate_rows` applies the same rule and refuses to write a
   disagreeing row, so this only catches a hand-made one.
7. **Compile** through the Lean server with a 60 s timeout. Any error message →
   `compile_error`; a `sorry` warning or REPL `sorries` entry → `has_sorry`;
   the REPL timing out on the submission → `timeout`; any other REPL error
   string, a 500 raised from executing this snippet, and a `{"message": ...}`
   payload → `lean_error`, charged to the model because every one of those is a
   failure the REPL reported while running the model's own snippet, and a
   failure the model could have caused is scored 0 rather than masked.

### Who a failure is charged to

Masking removes a rollout from `mean/reward` and pass@k entirely, so the rule has
to be conservative in one specific direction: **a failure the model could have
caused is scored 0, never masked.** Upstream charges every compile failure to the
model; each masked case below is a case where this harness reached no verdict at
all, so counting it as a failed proof would be the opposite error.

Charged to the model (reward 0.0, `mask_sample: false`, in the denominator):

| Status | When |
| --- | --- |
| `timeout` | the REPL hit the budget on the submission body — a proof that does not terminate is the model's output, and excusing it would make hanging reward-neutral |
| `lean_error` | the REPL reported an error string; `/verify` answered 500 from executing this snippet; or the per-item `response` was a `{"message": ...}` Error object — which Kimina's own `ReplResponse.analyze()` would call `repl_error`, but which the REPL produced while running the model's snippet, so it is charged rather than masked (see "A `/verify` reply whose payload is an error object") |
| `model_header_timeout` | the header that would not load inside the budget is one the model wrote itself |
| `model_header_error` | the header Kimina could not run at all (500, "Failed to run header on REPL") is one the model wrote itself. Narrower than "a bad import": an unresolvable module is *not* this case — the REPL reports it in parseable JSON and it lands in the body diagnostics as `compile_error`. What reaches here is a REPL that died or wrote non-JSON while loading the header, which model code can cause through the `RLIMIT_AS` cap and `native_decide` |

One shape reaches the model without any 5xx at all, and is charged: a header
that fails at the *Lean* level without raising and without timing out — a
partially built Mathlib, a bad olean, a module the REPL reports on rather than
dying on. Kimina's `check.py` guards this with `if prep and prep.error`, but
`send_timeout` never sets `error`, so the guard is dead code at this pin and the
body is run against the broken environment anyway. The cascading errors arrive
as ordinary diagnostics and are scored `compile_error`. Upstream behaves
identically, and the direction is the conservative one; the systematic version
of it is what the image's build-time REPL probe and the `lean_version` echo are
there to catch.

The 500 case is the one worth spelling out, because not every 500 is the
model's. Kimina answers 5xx for one snippet in four shapes — three
`HTTPException(500, str(e))` raised in `run_one` (`server/routers/check.py`;
nothing else in the mounted routers answers 5xx — `server_old/healthcheck.py` raises 503, but `create_app` does not mount it, and 503 is retried as saturation anyway), plus one the endpoint never catches:

| Shape | Raised when | Charged to |
| --- | --- | --- |
| `check.py:159` | executing the **body** raised. `server/repl.py` raises `LeanError("Lean process broken pipe")` / `LeanError("Failed to write to REPL stdin")` when the REPL is gone, and `ReplError("JSON decode error")` when its stdout is not JSON — all reachable by model code that kills the REPL, which runs under an `RLIMIT_AS` cap and may use `native_decide` by design | the model → `lean_error` |
| `check.py:118`, detail `"Failed to start REPL"` | the Lean process would not start (`repl.start()`, which shells out to `lake env`) | the harness → `sandbox_error` |
| `check.py:118`, detail `"Failed to run header on REPL"` | the **import header** raised. Kimina's header is the submission's own leading `import` run (`server/split.py`), so this can be a header the model wrote as easily as a broken toolchain. Not an unresolvable module, though — the REPL answers that in JSON without raising; what reaches here is the REPL dying or writing non-JSON while loading the header. `manager.prep` normalises every non-timeout header failure to this one string and the reply says nothing more | whoever wrote the header: `sandbox_error` if harness-supplied, `model_header_error` if not |
| an exception `run_one` never caught | Starlette's default handler answers with a 500 whose body is **not** JSON. Reachable through the prisma writes at `check.py:96-106`/`129-140`/`169-186`, which only run with `LEAN_SERVER_DATABASE_URL` set — the shipped `kimina_image` sets it empty | no `detail` to match, so the model → `lean_error` |
| `check.py:84` | `manager.get_repl` raised something other than `NoAvailableReplError`. **Not** a failed spawn, despite where it sits: spawning is `repl.start()`, which runs inside `manager.prep`, so a real spawn failure arrives at `:118` as "Failed to start REPL". What `get_repl` can raise is `Repl.create`, which writes through prisma when a database is configured | no `detail` to match, so the model → `lean_error`; near-unreachable, see below |

FastAPI serialises the three `HTTPException`s as `{"detail": str(e)}`, so the
two `manager.prep` cases arrive as those fixed strings and are told apart by
them (`lean_client.is_model_attributable_server_error`,
`lean_client.is_header_run_failure`). This does key on the wording of a server
message, deliberately — it is the only thing distinguishing sites that share one
status code, and `fine_eval.HEADER_TIMEOUT_MARKER` already keys `header_timeout`
on Kimina's literal "header command timed out". If upstream reworded either
marker these cases would become charged rather than silently masked, which is
the direction that cannot inflate a score.

The last two rows have no `detail` this client can match and so fall through to
`lean_error`. For the Starlette shape that is a deliberate choice of direction:
an operator who enables the proof log takes a downward bias on the score rather
than a shrinking denominator, and a systematic database failure is loud in the
server's own logs. For `check.py:84` it is a residual rather than a live
tradeoff — against the shipped image, which configures no database, essentially
nothing in `get_repl` can raise at all.

A 5xx Kimina does not emit at all — 502, 504 and friends — is a proxy or load
balancer in front of the server answering for a server that did not, which no
model can cause, so it is masked. (503 never reaches that decision: it is
retried as saturation.)

Charged to the harness (reward 0.0, `harness_failure: 1.0`, `mask_sample: true`,
`failure_reason` set, out of the denominator and reported under `coverage/`):

| Status | `failure_kind` | When |
| --- | --- | --- |
| `sandbox_error` | `provider_unavailable` | the connection was refused or timed out client-side; `/verify` answered a non-5xx HTTP error (401, 404, 422 — this client or its credentials, not the model); it answered 500 with "Failed to start REPL", or with "Failed to run header on REPL" **and** the header was the harness's own; it answered a 5xx Kimina never emits (502, 504 — a proxy, not the server); saturation (429/503) survived all three retries; the reply was not JSON, had no `results`, carried a result with neither an `error` nor a `response`, or carried an `error`/`stderr` object instead of a verdict |
| `header_timeout` | `provider_unavailable` | a cold REPL could not finish `import Mathlib` inside the timeout, **and** the header was the reference statement's own or the default one `extract_lean_code` prepends — Kimina reports this as `Lean REPL header command timed out`, distinct from the submission timing out |
| `bad_task` | `combibench:bad_task` | the row cannot be scored: no `formal_statement`, malformed `answers`, or an answer count that disagrees with the number of `_solution` abbrevs the statement declares |

None of these is a verdict on the proof: the server either never ran it or never
reported what it found. `bad_task` is namespaced because
`nemo_gym/failure_kinds.py` has no shared name for a malformed task row; the two
Lean-server faults are the registered `provider_unavailable`.

Which header a failed header command belongs to — the timeout and the 500 alike
— is decided by re-deriving Kimina's own split (`server/split.py`: the leading
run of `import` lines, Mathlib hoisted, duplicates dropped) for the submission
and for the reference statement and comparing them; see
`fine_eval.header_is_harness_supplied`. Harness-supplied, it is masked
(`header_timeout` / `sandbox_error`); model-authored, it is charged
(`model_header_timeout` / `model_header_error`).

Every response carries `lean_version`, the Lean version the server reports for
`#eval Lean.versionString`; a server built for another toolchain otherwise scores
every row `compile_error` with nothing in the rollouts saying why. The probe runs
as a **background task** on the first compile and is never awaited: it goes
through the same code path a submission does, so waiting for it would put its
whole Lean timeout in front of the first rollouts whenever the server accepts
connections without answering — the run would die of agent and eval timeouts
rather than produce the masked `sandbox_error` that case is meant to produce.
`lean_version` is null until the probe comes back, which is that field's
documented meaning of "not known". A probe that returns no version is not cached
as an answer — one dropped connection would otherwise switch the guard off for
the whole run — and is started again by the next compile, up to three probes per
process, after which the failure is logged at ERROR and `lean_version` stays null
for the run. Its Lean timeout is 30 s rather than the submission's 60 s: nothing
waits on it, so it should not hold a `max_concurrent_lean_requests` slot longer
than it must.

`compute_metrics` emits the pooled `pass@k` / `pass@1[avg-of-k]` keys — the
figures the tables below report — and adds `hackmath/`, `brualdi/`, `imo/` and
`math_competitions/` pass rates keyed on `tag`. Upstream reports one pooled
figure, so the pooled keys and the inherited `mean/reward` are the headline and
the per-family keys are supplementary; they are not promoted to `key_metrics`.
`get_key_metrics` does promote the `coverage/` block alongside `mean/*`: because
masking keeps a harness fault out of `mean/reward`, a Lean-server outage shrinks
the corpus the score was computed from instead of lowering it, and a run that
measured 3 rollouts out of 1600 would otherwise read like a healthy one with a
slightly different number. `coverage/measured_rollouts`,
`coverage/masked_rollouts`, `coverage/measured_tasks` and
`coverage/fully_masked_tasks` are therefore in the headline set. They are empty
unless something was masked, so a clean run publishes exactly the keys it did
before.

### Five deliberate departures from upstream

Two of them can only make this server accept where upstream rejects; the other
three run the other way. All five are named below; the fifth is a bound on what
goes on the wire rather than a scoring rule.

**The gold answer is elaborated at the abbreviation's declared type.** Lean's
`=` elaborates both sides before unifying them, so in upstream's form
`example : imo_2014_p2_solution = fun n => ⌈√n⌉₊ - 1` the right-hand side is
read with `n : ℝ` and fails against an `ℕ → ℕ` abbreviation, and
`fun n => n * (n + 1) / 4` fails to synthesize `HDiv ℕ ℕ ℚ`. Measured with the
published answers substituted into their own statements: upstream's form
elaborates for **40 of 45** fill-in-the-blank problems; `brualdi_ch8_6`,
`imo_2014_p2`, `imo_2019_p5`, `imo_2022_p1` and `imo_2023_p5` can never score
under it, even with the exact gold answer and a complete proof. Ascribing the
declared type, `(<gold> : <type>)`, elaborates for **45 of 45**. Set
`answer_check_ascription: false` for upstream's behaviour; both measurements
come from `scripts/harness_validation.py` on the GitHub `test` split, with and
without `--no-ascription` (commands under "Re-running the validation").

**Trailing whitespace is ignored in the statement check.** Statements carry
lines consisting only of spaces, left behind when comments were deleted from
the published copy. The count depends on which copy is measured: **12 of the
100** Hugging Face `test` statements at `882ba08b` (13 counting any trailing
whitespace), and **1 of the 100** rows `prepare.py` writes from the GitHub
files, the default source. Trailing whitespace is never significant to Lean, so
a model that reproduces the statement without those invisible characters has
not changed what it proves; upstream's byte-exact substring test would reject
it. Indentation and every visible character are still compared exactly. Set
`normalize_trailing_whitespace: false` for upstream's behaviour. Unlike every
other figure in this README, those two counts come from no committed report:
they were measured ad hoc by matching `^[ \t]+$` against the `formal_statement`
of the rows `prepare.py` writes for each source, and prepared rows are not
committed. Re-derive them from a prepared split rather than treating them as
traceable evidence.

**A structurally reported `sorry` is rejected.** `classify_lean_result` returns
`has_sorry` when the REPL's `sorries` list is non-empty as well as when a
warning says the declaration uses `sorry`. Upstream's `is_error` never reads
that field — `sorries` appears nowhere in its `evaluation/` tree — so this is
the one departure that is *stricter* than upstream: a submission whose leftover
`sorry` is reported only structurally fails here and passes there. It is not
configurable, and it is kept deliberately: rewarding a proof that still contains
`sorry` would be a worse error than disagreeing with upstream about it. It was
not exercised on the run measured below: all 65 `has_sorry` rollouts there
(41 + 24) were rejected by upstream too, so Kimina had emitted the warning that
upstream's `is_error` reads in every one of them.

**A `/verify` reply whose payload is an error object is never a success.** The
per-item `response` is not always the `{messages, sorries, env, time}` command
response: Kimina's `server/repl.py` hands back whatever `json.loads` produced
from the REPL's stdout with nothing validating it, and its client's `extend()`
maps a `{"message": ...}` body to `ExtendedError`, so `/verify` can answer with
an Error object in `response` and no top-level `error` at all. Such a reply has
no messages and no sorries, which is exactly what a clean compile looks like, so
`parse_verify_response` treats a payload carrying `message`, `error` or `stderr`
as a failure and fails closed. The test is key *presence*, matching upstream's
`is_error` (`if "error" in feedback` / `if "stderr" in feedback`), so a payload
carrying `{"error": null}` or `{"stderr": ""}` fails here exactly as it does
there; guarding `message` is an *addition* — upstream has no case for it and
would score that reply 1.0.

Which of the two fail-closed directions each key takes is a choice this harness
makes, not one Kimina makes for it. Kimina's own per-snippet classifier,
`ReplResponse.analyze()` in `client/kimina_client/models.py`, groups all three
with infrastructure rather than with a verdict: it tests `"message" in
self.response` **before** `is_error(...)` and returns
`SnippetStatus.repl_error` ("Error while running snippet, at REPL level"), so
Kimina would call a `message` payload a REPL-level failure, not a Lean error on
the snippet.

`error` and `stderr` are masked (`sandbox_error`): a REPL that returned one of
those instead of a command response never evaluated the proof, so there is no
verdict to charge to the model, and upstream reaching the same 0.0 by failing
the submission differs only in whether the rollout stays in the denominator. On
top of that, neither key can arrive from the pinned Kimina at all — its own
`client/kimina_client/proof_utils.py` says so in a comment ("there is never
error or stderr in the feedback") — so they are guarded as defence against a
non-Kimina server and there is no model-caused failure behind them to mask.

`message` is **charged** (`lean_error`) despite Kimina's `repl_error` reading,
because it is the one of the three the pinned server actually produces, and it
is produced while the REPL is running the model's own snippet: it reaches
`response` through `server/repl.py` handing back unvalidated `json.loads`
output, and the client's `extend()` admits the shape explicitly. That makes it a
failure the model could have caused, and the rule this harness keeps is that
such a failure is never masked — masking is the direction that deletes a
model-caused failure from the denominator, while charging it costs at most a 0
on a rollout the REPL did not finish, which is the same 0 upstream reaches for
the shapes it reads at all. What a `message` payload means to Lean is still not
readable at the pin (`leanprover-community/repl` is not vendored in the Kimina
tree, only referenced by URL from its `Dockerfile`/`setup.sh`), and the choice
of direction does not depend on knowing.

**A result carrying neither an `error` nor a `response` is not a verdict
either.** `{"results": [{"custom_id": "x"}]}` and the same with
`"response": null` have no error, no messages and no sorries — indistinguishable
from a clean compile — and `parse_verify_response` fails closed on both, to the
masked `sandbox_error`, for the same reason as the error payload above: this is
the same under-specified-reply-read-as-success bug one level up. Unlike that
one, this guard is **defence against a malformed or non-Kimina server rather
than a shape the pinned server produces**: `ReplResponse`
(`client/kimina_client/models.py`) carries a `@model_validator`
`require_error_or_response` that raises unless exactly one of the two fields is
set, so a well-behaved Kimina cannot build such a result in the first place.
(`/verify` is declared `response_model_exclude_none=True` in
`server/routers/backward.py`, which is what would serialise such an object to
neither key if one ever existed.) It is kept because failing closed costs
nothing and a reply nothing on the wire guarantees must not be able to score
1.0.

**Extracted code longer than `max_code_characters` is rejected unsent.**
Upstream's extraction has no length bound at all, so this is a departure, but it
is a wire-safety bound rather than a scoring rule: the status is `code_too_long`
and nothing is put on the Lean server's queue. The cap is 200,000 characters
against a longest pinned statement of 3,054, and it did not fire once in the
3,200 rollouts measured below — it exists so a pathological generation cannot
occupy a REPL, not to decide any proof.

### Known blind spots, kept for fidelity

- `native_decide`, `decide` and `set_option maxHeartbeats` are allowed, as
  upstream allows them. `native_decide` trusts the compiler rather than the
  kernel.
- The `axiom` ban is a substring test: an identifier containing `axiom` fails
  a valid proof, and constructs upstream does not name (`opaque`,
  `implemented_by`, `unsafe`) are not banned.
- Extra declarations are allowed anywhere in the code; only the reference
  paragraphs are required.
- The answer tag is the text between the paragraph's **first** `abbrev` and its
  **first** `_solution`, which is upstream's rule verbatim. A paragraph holding a
  helper `abbrev` beside the `_solution` one therefore yields a tag that is not
  an identifier (`"helper : ℕ := 5\nabbrev a_solution"`), and the
  appended `example:` is invalid Lean — under upstream as much as here. The
  `abbrev_types` regex is anchored on the `_solution` declaration, but that only
  keeps the type ascription from adding a *Gym-only* error on top: upstream has
  no ascription, so a mis-read type would be damage this server alone does. The
  tag is left as upstream computes it, because changing it would diverge from
  the harness this server exists to reproduce. No paragraph in the pinned corpus
  is shaped that way.
- `answer_tags` returns a **list**, where upstream collects the tags into a dict
  keyed by tag name (`evaluation/util.py`). Two paragraphs that declare the same
  `<name>_solution` therefore collapse to one answer check upstream and stay two
  here, which shifts the positional pairing with the published answers from that
  point on. The pinned corpus has 47 distinct abbreviations and no repeat, so
  this is unreachable on it; a list is kept because it is also what preserves
  statement order, which the positional zip depends on.
- A statement paragraph that opens with `open ... in` directly above its
  `theorem` starts with a header prefix, so the whole paragraph — theorem
  included — is skipped by the statement check. Upstream has the identical
  blind spot and no paragraph in the pinned corpus is shaped that way, so the
  behaviour is kept rather than diverging.
- `REVERIFY_MODE` is `STATELESS`: the server carries nothing between calls.
  It is not a claim that Lean compilation is a pure function of the text —
  timeouts depend on the machine.

## Harness validation

Model-free checks, all run against Mathlib v4.24.0 through this server's code
path, in one job rather than assembled from separate runs. The reports are not
committed (see "Re-running the validation" for the commands and why), so the
numbers below are stated here and re-derived by running
`scripts/harness_validation.py` against a live Lean server at Mathlib v4.24.0;
without one they cannot be checked. Every report records the `dataset_source`,
`dataset_revision`, `split` and `answer_check_ascription` it was produced with,
so a number in this table traces to the corpus and the setting that produced it.

| Check | GitHub source (default) | Hugging Face source |
| --- | --- | --- |
| Statement with its `sorry`s compiles (only `sorry` warnings) | **100 / 100** | 94 / 100 |
| "With solution" statement compiles | **100 / 100** | 94 / 100 |
| Published answer substituted + ascribed answer check elaborates | **45 / 45** | 41 / 45 |
| Same with upstream's unascribed check | 40 / 45 | 37 / 45 |

The six Hugging Face statement failures (`hackmath_6`, `imo_2008_p5`,
`imo_2011_p2`, `imo_2021_p5`, `imo_2022_p6`, `imo_2023_p5`) are statements
upstream rewrote in the repository for the Lean bump and never pushed to the
dataset. The Hugging Face gold-answer column is bounded by that: its four
ascribed failures (`hackmath_6`, `imo_2008_p5`, `imo_2022_p6`, `imo_2023_p5`)
are all statements that do not compile in the first place, so no answer can be
checked against them. The unascribed column adds four of the five problems named
under "Five deliberate departures" — `brualdi_ch8_6`, `imo_2014_p2`,
`imo_2019_p5` and `imo_2022_p1` — the fifth, `imo_2023_p5`, being already in the
ascribed column because its Hugging Face statement does not compile at all.
41 → 37 is that same ascription effect, measured on a corpus where one of the
five problems it affects was already lost for another reason.

Negative controls through `verify()`, all 100 rows, every one scoring 0:

| Control | Denominator | Status observed |
| --- | --- | --- |
| Empty output | 100 | `empty_generation` |
| Statement echoed back with `sorry` | 100 | `has_sorry` (55 proof-only), `compile_error` (45 fill-in: the answer check cannot reduce a `sorry` abbrev) |
| `axiom cheat : False` prepended | 100 | `banned_tokens` |
| Main theorem's goal replaced by `True` | 99 (one statement's theorem is not the last declaration) | `statement_modified` |

The controls cover these named failure classes and no others. Upstream publishes
no reference proofs, so there is no gold-as-prediction check over the real
corpus; the five synthetic example problems have complete proofs in
`tests/fixtures/synthetic_solutions.json`, and all five score 1.0 through the
live server — `scripts/harness_validation.py` over the prepared example rows
with `--solutions tests/fixtures/synthetic_solutions.json`, the last command
under "Re-running the validation". One of them answers
`3 / 12` where the gold is `1 / 4`, exercising the `norm_num` path of the
answer check.

## Reward profiling

A score is a property of a model, its prompt, its sampling configuration and the
date, so it goes stale in a README without anyone noticing; this repository's
README guide asks for model scores to live with the run that produced them
instead. The open-weights baseline for this benchmark — which model, the pass@16
and pass@1 figures for both settings, the per-family breakdown and the token-budget
caveat — is in the pull request that added it.

What stays here is the harness self-validation below, which involves no model: it
is a test result, not a measurement of anyone's model.

## Lean server

Scoring needs a [Kimina Lean Server](https://github.com/project-numina/kimina-lean-server)
(MIT) — the REPL-pooling service upstream's own harness verifies through — at
Lean and Mathlib `v4.24.0`, upstream CombiBench's toolchain. The published image
defaults to a different Lean version, so this server ships its own Dockerfile:
[`kimina_image/README.md`](kimina_image/README.md) has the build, the `docker
run` line with the hardening it needs, the `LEAN_SERVER_*` settings and the
isolation caveats. Point this server at a running instance with
`COMBIBENCH_LEAN_SERVER_URL` (and `COMBIBENCH_LEAN_SERVER_API_KEY` if it has
one).

Keep `max_concurrent_lean_requests` equal to the server's
`LEAN_SERVER_MAX_REPLS`. Both default to 8 here, but that is the value
`kimina_image` sets, not a Kimina default — Kimina's own is
`max(cpu_count() - 1, 1)`, a property of whichever host it runs on — so against
any other server both numbers have to be set deliberately.
It is the bound for the whole resources server, not per
process: the `asyncio.Semaphore` that enforces it lives on one client instance,
so under `num_workers: N` the configured value is divided by `N` and each worker
holds its share (floored at 1). Rollout fan-out is otherwise unbounded, and a request
beyond that number is a connection waiting for a REPL that does not exist yet;
past the server's own queue it becomes a 429 charged to nobody. A compile that
exhausts the client timeout is not retried (`_max_connection_retries=1`):
Gym's shared client would otherwise spend three REPL jobs and three times the
wall clock to reach the same verdict. A 429 or 503 *is* retried, up to three
attempts with a 1 s then 2 s backoff — saturation costs the server no REPL time,
and giving up on it would turn a busy moment into a masked `sandbox_error` that
quietly shrinks the measured denominator.

The client's HTTP budget for one `/verify` is derived from the server's own
worst case rather than set as a flat margin, so the server always answers first:
`lean_server_max_wait + 2 * lean_timeout_seconds + 30 s`
(`lean_client.http_budget_seconds`). Kimina spends, for one snippet, up to
`max_wait` waiting for a free REPL (`manager.get_repl`; `LEAN_SERVER_MAX_WAIT`,
60 both in Kimina's defaults and in `kimina_image`), then up to the request
timeout running the import header on a cold REPL, then up to it again running
the body — 180 s at the defaults. A shorter budget would matter for attribution
and not only for latency: a client-side timeout is a masked `sandbox_error`, so
a non-terminating proof the server would have returned as its own timeout, and
charged to the model, would instead be deleted from the denominator whenever the
client blinked first. `lean_server_max_wait` is configurable for a server set to
something other than 60.

### What is shared with the other Lean benchmarks, and what is not

Reused from [`lean_proof/`](../lean_proof/): **`status.py`**, the status
vocabulary as above, and **`toolchain.py`**'s `TOOLCHAIN_PROBE` and
`parse_lean_version` — the REPL answers in structured messages rather than on
stdout, so the client joins them into the shape the parser expects rather than
writing the version regex twice. Not reused, and deliberately:

- **`proof_utils.py`.** Its extraction strips thinking, accepts any fenced
  block, and falls back to an unfenced Lean file, where upstream CombiBench
  takes the last ```` ```lean4 ```` block (falling back to ```` ```lean ````),
  prepends a default header and calls anything else a format error; its
  banned-token set is `sorry`/`admit`/`axiom`/`unsafe` against CombiBench's
  `axiom`/`local_instance` substrings; its statement check is whole-file against
  upstream's paragraph-substring test. Sharing any of these would change scores
  relative to the published numbers, which is the one thing this server exists
  not to do.
- **`lean_sandbox.py`.** It shells `lake env lean` through `nemo_gym.sandbox`,
  one process per request. CombiBench needs Kimina's header-keyed REPL pool —
  both because it is what upstream's harness talks to, and because a fresh
  `import Mathlib` per proof is unaffordable at 100 problems × 16 repeats.
  [`kimina_image/README.md`](kimina_image/README.md#why-not-the-existing-lean-sandbox)
  has the point-by-point comparison and why a new image had to be built.

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

Regenerating the committed example is three stages — the tracked file is the
output of the third, not the first:

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

After any change to the scoring path, regenerate every report — one per
(source, split, ascription) combination the README table cites — and update the
numbers above if they moved. The reports are written to a scratch directory and
not committed: a resources server's `data/` holds only `example.jsonl`,
`example_rollouts.jsonl` and `example_metrics.json`.

```bash
URL=http://127.0.0.1:12332
V=resources_servers/combibench/scripts/harness_validation.py
D=/tmp/combibench_validation && mkdir -p $D

# six reports: one per (source, split), plus a --no-ascription rerun of each
# plain `test` split for the upstream-check row. See `--help` for the flags.
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

# the five synthetic example problems, with their reference proofs; prepare them
# as in the first stage above, with --output /tmp/combibench_example_prepare.jsonl
python $V --input /tmp/combibench_example_prepare.jsonl --lean-server-url $URL \
    --solutions resources_servers/combibench/tests/fixtures/synthetic_solutions.json \
    --output $D/harness_validation_example.json
```

Prepared rows go to a scratch path, not into `benchmarks/*/data/`: benchmark
rows are not committed, and the two sources would otherwise overwrite each
other on `prepare.py`'s default output path — which is how an earlier set of
these reports came to name a GitHub input while measuring the Hugging Face
corpus.

### Agreement with upstream's harness

The re-derivation in `fine_eval.py` is checked against upstream's actual code,
not only against fixtures. `scripts/upstream_agreement.py` downloads CombiBench
at the pinned revision, imports `evaluation/verifier/one_stage_verify.py`
unmodified, and re-scores collected rollouts through the same Lean server,
reporting per-item agreement rather than a matching headline.

This is harness self-validation, not a model result: it measures whether two
verifiers agree on the same outputs, and the outputs' own score is irrelevant to
it. Measured over the 3,200 rollouts of the open-weights baseline recorded in the
pull request. The reports are not committed, so the numbers are stated here and
reproduced by the `upstream_agreement.py` command at the end of this section,
which needs both those rollouts and a Lean server:

| Setting | Rollouts | Scored by both | This verifier | Upstream | Agreement |
| --- | --- | --- | --- | --- | --- |
| `test` | 1600 | 1598 | 34 | 34 | **1598 / 1598** |
| `test_with_solution` | 1600 | 1600 | 31 | 31 | **1600 / 1600** |

The same rollouts pass under both, not merely the same number of them.

Two rollouts of the `combibench` run are `sandbox_error`: this verifier reached
no scoring decision on them at all, so counting them as "agreements" — which the
script's `summary.agreements` of 1600 does, because upstream also called them
not-a-success — would be counting a non-verdict as a match. They are excluded
from the denominator in the table above and the summary counters in the JSON are
left as the script wrote them. That run predates "Who a failure is charged to"
above, which narrowed what `sandbox_error` covers: a 500 raised from executing
the snippet is now a `lean_error` charged to the model rather than a masked
non-verdict, and so is a 500 from a header the model wrote (`model_header_error`)
and a `{"message": ...}` payload, so a rerun may place those two rows in the
scored denominator instead — unless they were a failure to start a REPL, a
harness-supplied header, or a gateway 5xx, which stay masked.

What the run does and does not establish about the departures. The two
accepts-here-rejects-there departures produced no disagreement, but that is
weak evidence: 1566 of the 1600 rollouts are rejections both harnesses make for
the same reason, and none of the five problems the ascription departure affects
(`brualdi_ch8_6`, `imo_2014_p2`, `imo_2019_p5`, `imo_2022_p1`, `imo_2023_p5`)
was ever solved by the profiled model, so the ascription path was never
exercised end to end. The evidence for that departure is the model-free
measurement instead: `harness_validation.py` on the GitHub `test` split with
the ascription against the same run with `--no-ascription`, 45/45 answers
elaborating with the ascription against 40/45 without it. The stricter
`sorries` departure was not exercised either (see above).

**Upstream's harness needs two transport-level fixes to run at all**, applied
in that script and nowhere else. It reads `res["error"]` *and* `res["response"]`
by subscript in one expression. Kimina omits `error` when there was no error, so
the first read raises `KeyError`, upstream's blanket `except Exception` turns it
into "proof invalid", and every compiling proof is reported as failed — 0/1600
unpatched. It also omits `response` whenever a result carries an `error`
(`BackwardResponse.response` is `NotRequired` and `/verify` is
`response_model_exclude_none=True`), so the second subscript raises on every
server-side timeout.

> The figures in the table above were measured with only the first key filled.
> That does not change them: a timeout row has `error` set, so upstream reaches
> "not a success" either way, and this verifier scored those rows 0 as well. But
> those rows agreed through upstream's exception path rather than through its
> `is_error`, and the script now fills both keys so a rerun exercises upstream's
> own logic on every row. Nor is it a dependency
that could have been pinned better: upstream's harness does not use the `kimina`
client package at all — it hand-rolls the HTTP calls with `aiohttp` — and its
`pyproject.toml` carries only a floor, `kimina>=0.1.1` (0.1.1 is 2025-07-24,
three months before Lean v4.24.0 was released on 2025-10-14), while its own
statements are now on v4.24.0. It is also why this server talks to Kimina
through its own client, which reads that field with `.get`.

```bash
uv pip install loguru strenum tenacity tqdm   # upstream's imports, which Gym does not ship
python resources_servers/combibench/scripts/upstream_agreement.py \
    --rollouts results/combibench/rollouts.jsonl \
    --output /tmp/combibench_validation/upstream_agreement.json \
    --lean-server-url http://127.0.0.1:12332
```

The report keeps only the disagreeing rows under `rows`; pass `--full-rows` for
the complete per-row map. A disagreement can come from any of the five departures
above, each in the direction named there — except the error-payload guard, which
takes the row out of the comparison entirely (it is `sandbox_error`, a
non-verdict), and the `code_too_long` bound, which is unreachable on this corpus.
To measure agreement with the two configurable ones removed, rescore the same
rollouts with
`answer_check_ascription: false` and `normalize_trailing_whitespace: false` and
pass that file as `--rescore-with`. It must carry a verdict for every rollout
being compared: the script fails closed on a missing key exactly as it does on a
colliding one, because a rollout with no verdict is not an agreement.

### The committed example is synthetic

`data/example.jsonl` holds five hand-written problems in the upstream shape
(binomial coefficient, permutation count, pigeonhole, Gauss sum, a rational
probability), not benchmark rows. They cover a fill-in answer of each type the
verifier handles differently — natural number, function, rational — one
proof-only statement, and a statement carrying a spaces-only line to mirror the
published data. Real runs use the prepared split, which is never committed.

## Tests

```bash
gym env test --resources-server combibench
```

## Licensing

Code: Apache 2.0. `fine_eval.py` re-derives the extraction, statement and
answer rules of upstream's MIT-licensed `evaluation/util.py` and
`one_stage_verify.py` (Copyright (c) 2025 Moonshot AI and Project Numina);
`benchmarks/combibench/prompt.yaml` reproduces the prompt strings from the same
repository's `evaluation/config/template.json5`.

CombiBench data: MIT, per the repository `LICENSE` and the dataset card.
Upstream sources: hackmath.net exercises, Brualdi's *Introductory
Combinatorics*, IMO and other olympiad problems (APMO, Baltic Way, EGMO, IMO
Shortlist, IZhO, BxMO, USAMO); the IMO 2024 P3 and P5 statements are taken from
Mathlib's `Archive`. No benchmark rows are committed; `prepare.py` downloads
them at run time.
