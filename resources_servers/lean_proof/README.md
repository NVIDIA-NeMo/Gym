# lean_proof

Shared plumbing for Lean benchmarks. Not a resources server — it has no `app.py` and no config;
benchmarks import from it.

| Module | What it owns |
|---|---|
| `proof_utils.py` | `<think>` stripping, comment/string stripping, fenced-block extraction, banned declarations, unterminated-comment detection, whole-file statement preservation |
| `status.py` | the `proof_status` vocabulary and the sandbox-result mapping |
| `lean_sandbox.py` | `LeanSandbox` — one sandbox per process, `lake env lean` compile, toolchain probe, `CompilerOutput` |
| `toolchain.py` | parsing and normalising Lean versions |
| `lean_image/` | one container image per Mathlib version |

Shared rather than copied because this is where subtle bugs accumulate: a fenced block taken
from the wrong place, a `sorry` matched inside a comment, a compile error reported as a sandbox
failure. Each was wrong at least once, and a fix should land once.

## Adding a Lean benchmark

Reuse the modules above; write only what is yours:

- `prepare.py` — fetch a pinned upstream revision, write flat rows
- `task_data.py` — the row schema
- `configs/` — `sandbox_provider`, the image for your Mathlib version, your prompt
- `app.py` — your metrics, and any verification rule the shared status vocabulary does not
  cover. A benchmark with its own rule adds a status next to the shared ones rather than
  redefining them.

`resources_servers/leancat` is the worked example. `resources_servers/formal_conjectures` is the
second one, and shows where a benchmark legitimately parts company with the defaults.

## One hole, or several?

The two shapes a whole-file Lean task comes in, and what each needs from this library:

| | reference has exactly one hole | file keeps holes the model must not fill |
|---|---|---|
| example | `leancat` | `formal_conjectures` |
| statement check | `check_statement_preserved` (splits the reference on `sorry`) | `check_target_statement_preserved` (target signature only) |
| banned tokens | `find_banned_declarations(code)` | `find_banned_declarations(code, DECLARED_SHORTCUT_TOKENS, declarations_only=True)` |
| Lean's `sorry` warning | a failure | expected — `determine_proof_status(…, sorry_is_error=False)` |
| what rules on the proof | the compile | `#print axioms <target>`, in the server |

Getting the second column wrong is not a subtle miss: splitting on `sorry` when the hole is not
textually unique threw out 23 of 100 honest answers, and treating the warning as an error scores
every task 0. A server that opts out of the `sorry` rule owes a check of its own that the
*target* is proved — dropping it without that is how an unfilled proof scores 1.0.

## Mathlib versions are not interchangeable

Compiled `.olean` files do not carry across versions, and a benchmark pointed at the wrong one
does not error at startup — it fails individual tasks with ordinary-looking "unknown
identifier" errors. Measured on LeanCat against v4.12.0: 36 of 100 reference statements fail to
compile *with their `sorry` still intact*, so the run reports a plausible number capped at
64/100 and skewed by difficulty.

Hence one image per version (`lean_image/`), and a gate that compiles every reference statement
before a run is trusted.
