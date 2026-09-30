# Formal Conjectures

[Formal Conjectures](https://github.com/google-deepmind/formal-conjectures) is a Google DeepMind library of
formalized mathematics in Lean 4. Most of it is **open** (the conjecture carries a `sorry`), so it has no
reference proof and cannot be scored.

What is scored here is the theorems that ship with Lean proofs: `@[category test]` sanity checks,
`@[category API]` supporting lemmas, textbook exercises, and formalized `research solved` results. `extract.py`
strips such a proof off and hands back the file with a hole in it.

The result is **1,526 tasks drawn from 323 upstream files**, against Lean/Mathlib **v4.33.1** —
1,526 of 1,788 extractable candidates, the rest dropped by the validation sweep below:

| `category` | Tasks | What it is |
|---|---:|---|
| `test` | 1,285 | sanity checks pinning a definition to a known value |
| `API` | 196 | supporting lemmas built around a new definition |
| `textbook` | 25 | exercises |
| `research solved` | 20 | formalized results from the literature |

The gradient is the point: `test` is near-solved and `research solved` is not, so the pooled number is the least
interesting one this server reports.

## Task format

Each row hands the model one Lean file — `import Mathlib`, whatever definitions and lemmas the target depends on,
and the target theorem with its proof replaced by `sorry` — and asks for the same file back with that one hole
filled. The prompt is `benchmarks/prompts/eval/formal-conjectures/default.yaml`.

Two properties follow from how upstream files are built, and they drive every design decision below:

1. **The file legitimately contains other `sorry`s.** A file typically pairs a proved lemma with the open
   conjecture it sanity-checks, and the task keeps every declaration *before* the target because proofs lean on
   them — dropping them broke 14 of a 32-task sample. So only the target's hole is the model's job.
2. **The file is not upstream's file.** `@[category ...]`/`@[AMS ...]` attributes are stripped and
   `import FormalConjecturesUtil` is rewritten to `import Mathlib`, so tasks compile against a stock Mathlib
   sandbox and FC never has to be installed. Statements that genuinely need FC-only definitions cannot survive that
   rewrite; they are dropped, and anything the filter misses is caught by the validation sweep.

## Verification

The first four checks are text-only and run **before** the sandbox call, so a submission that has already lost does
not cost a Mathlib compile.

| # | Check | Status on failure |
|---|---|---|
| 1 | A Lean code block was produced (last fenced block wins) | `empty_generation` |
| 2 | No `axiom`/`unsafe` **declaration** was added | `banned_tokens` |
| 3 | No unterminated `/-` block comment | `compile_error` |
| 4 | The target's statement is unchanged | `statement_modified` |
| 5 | The file compiles under `lake env lean` | `compile_error` / `timeout` / `sandbox_error` |
| 6 | `#print axioms <target>` reports no `sorryAx` | `unproved` |

Reward is 1.0 only if every check passes, else 0.0.

Three of these are specific to this server, because the file is allowed to keep holes the model was not asked to
fill:

- **Check 2 bans declarations, not tokens.** `sorry` and `admit` are *not* on the list, and `axiom`/`unsafe` only
  count when they open a declaration — `Classical.axiom_of_choice` is an ordinary reference.
- **Lean's "declaration uses `sorry`" warning is not an error** here (`determine_proof_status(…,
  sorry_is_error=False)`). Treating it as one would score every real FC task 0.
- **Check 6 stands in for it.** An incomplete Lean proof depends on `sorryAx`, and `#print axioms` reports that per
  declaration. The probe is appended by the server, never trusted to the model. It also catches indirection a
  textual scan cannot: a proof that leans on a sorry'd lemma earlier in the file. During dataset validation this
  rejected upstream "proofs" that were not actually proofs.

`#print axioms` has **two** output forms, and both matter:

```
'thm' depends on axioms: [propext, Classical.choice, Quot.sound]   -- ordinary classical proof
'thm' does not depend on any axioms                                -- fully constructive proof
'thm' depends on axioms: [sorryAx]                                 -- not a proof
```

Reading only the first form makes the *strongest* possible result — a constructive proof, which cites no axioms at
all — look like "no axiom line found", i.e. a missing declaration, and score 0. The parser handles both, and is
anchored to the target's name: the submission is a whole file and can print the axioms of any proved Mathlib lemma
it likes, so an unanchored scan lets a decoy line stand in for the server's probe.

**Check 4 is textual, and its limits are worth stating.** It asks whether the target's recorded signature is still
present, normalising away comments, indentation and `lemma`/`theorem` (which are interchangeable in Lean 4 —
comparing them verbatim rejected 53 correct submissions in a 12k-rollout run). Because it is a containment check,
a submission that *appends* to the conclusion — `True` becoming `True ∨ False` — passes it. Tightening it risks
the false rejections the rule was tuned to avoid, so it is left as is and recorded here;
`#check`/`#print axioms` on a stored rollout are the stricter tools. `statement_preserved` is reported on every
response, so a stricter criterion can be applied to stored rollouts without rerunning.

## Requirements

A Lean 4.33.1 / Mathlib v4.33.1 sandbox image, built by
[`resources_servers/lean_proof/lean_image`](../lean_proof/lean_image):

```bash
cd ../lean_proof/lean_image && ./build.sh v4.33.1
```

Verification runs through `nemo_gym.sandbox`, so any provider works — OpenSandbox, enroot, docker — and nothing has
to be started out of band.

**The Mathlib version must be exactly v4.33.1.** A different Mathlib does not fail loudly; it fails individual
tasks with ordinary-looking "unknown identifier" errors, indistinguishable from a model that could not do the
problem, and the run reports a plausible, meaningless number. Two checks guard it:

- **`check_sandbox.py`** compiles reference files and fails unless each target comes back proved. Run it before
  spending anything on inference.
- **The server** probes `Lean.versionString` once on the first `verify` and logs an `ERROR` on a mismatch with the
  row's `lean_toolchain` (falling back to `expected_lean_version`). It logs rather than raises, so a run already in
  flight is not killed; set `check_lean_version: false` to skip it.

```yaml
sandbox_provider: sandbox            # any nemo_gym.sandbox provider config
sandbox_config:
  image: ${oc.env:FORMAL_CONJECTURES_SANDBOX_IMAGE,gym-lean:v4.33.1}
compilation_timeout: 300.0
check_lean_version: true
expected_lean_version: "4.33.1"
```

## Metrics

Pooled `pass@k` and `pass@1[avg-of-k]`, plus the same broken out by `category` — FC's own difficulty proxy. The
split is the point: `test` sanity checks are near-solved while `research solved` is not, and a run that only moves
the easy tier should not read as progress. `statement_preserved` rides along as a second score so a run collapsing
because the guard rejects everything is visible without opening rollouts.

## Data

`verified_tasks.json` **is** the benchmark definition: the 1,526 task ids whose reference version was *observed to
compile clean with the target free of `sorryAx`* inside a Mathlib v4.33.1 sandbox. Candidacy is not answerability —
a statement can reference a definition that does not survive the rewrite to plain Mathlib, and some upstream
"proofs" transitively depend on a `sorry` elsewhere in the file. Neither is detectable by reading the text.

The 262 candidates the sweep rejected are not noise, and the two largest groups are worth knowing about: files
whose syntax needs `FormalConjecturesUtil` after all (`unexpected token '('`), and proofs that are correct but
exceed the kernel's budget (`(kernel) deterministic timeout`). Both would otherwise be unanswerable tasks scoring
a flat 0 for every model.

Task ids are `<path>::<namespace-qualified name>`. The qualified name is load-bearing: a file may declare the same
short name in several namespaces, and an id built from the short name collides — `sIncreasingrTuples.lean` alone
has five `not_lt₂*` lemmas. `prepare.py` and `check_sandbox.py` both raise on a duplicate id rather than letting
one task silently displace another.

`prepare.py` re-derives the rows for exactly those ids from the pinned upstream revision (`FC_COMMIT`), so the
dataset is reproducible from the pin without a Lean install. It fails loudly if the pin and the verified list have
drifted apart.

Regenerating the list means re-running the sweep, which needs a v4.33.1 sandbox:

```bash
python check_sandbox.py --image gym-lean:v4.33.1 --all --write-verified
```

Bump `FC_COMMIT` deliberately: another revision changes statements, proofs and category labels, none of which is
detectable from the JSONL alone, and it invalidates the verified list.

## License

Upstream Formal Conjectures is **Apache 2.0**, and the task files are derived from it. Task text carries upstream's
own copyright header. No data is vendored in this repository beyond the five-row `data/example.jsonl` smoke sample
and the task-id list; `prepare.py` fetches the rest at the pin.

## Running it

### 1. Build the Lean image

```bash
cd ../lean_proof/lean_image && ./build.sh v4.33.1
```

### 2. Gate it before spending anything on inference

```bash
python check_sandbox.py --image gym-lean:v4.33.1 --limit 25
```

### 3. Prepare and run

```bash
python benchmarks/formal-conjectures/prepare.py
gym run --benchmark formal-conjectures ...
```

## Verification backend

Everything not specific to FC — the sandbox, the toolchain probe, the text checks and the status vocabulary — comes
from [`resources_servers/lean_proof`](../lean_proof), the library the Lean benchmarks share. `unproved` is the one
status this server adds, and it sits next to the shared ones rather than redefining them.
