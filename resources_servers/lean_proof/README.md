# lean_proof

Shared plumbing for the Lean benchmarks. Not a resources server — it has no `app.py` and no
config; benchmarks import from it.

| Module | What it owns |
|---|---|
| `proof_utils.py` | `<think>` stripping, comment/string stripping, fenced-block extraction, banned declarations, whole-file statement preservation |
| `status.py` | the `proof_status` vocabulary and the sandbox-result mapping |
| `lean_sandbox.py` | `LeanSandbox` — one sandbox per process, `lake env lean` compile, toolchain probe, `CompilerOutput` |
| `toolchain.py` | parsing and normalising Lean versions |
| `lean_image/` | one container image per Mathlib version |

These are shared rather than copied because they are where subtle bugs accumulate: a fenced
block chosen from the wrong place, a `sorry` found inside a comment, a compile error reported
as a sandbox failure. Each was wrong at least once, and a fix should land once.

## Writing a new Lean benchmark

Reuse everything above; write only what is yours:

- `prepare.py` — fetch a pinned upstream revision, write flat rows
- `task_data.py` — the row schema
- `configs/` — `sandbox_provider`, the image for your Mathlib version, and your prompt
- `app.py` — your metrics, and any verification rule the shared status vocabulary does not
  cover. `formal_conjectures` adds `unproved` (compiled, but the target still rests on
  `sorryAx`); that belongs in its server, not here.

## Mathlib versions are not interchangeable

Compiled `.olean` files do not carry across versions, and each benchmark pins its own:
miniF2F/ProofNet/PutnamBench v4.12.0, LeanCat v4.19.0, Formal Conjectures v4.33.x. Pointing a
benchmark at the wrong one does not error at startup — it fails individual tasks with
ordinary-looking "unknown identifier" errors. Measured on LeanCat against v4.12.0: 36 of 100
reference statements fail to compile *with their `sorry` still intact*, so the run scores a
plausible-looking number that is capped at 64/100 and skewed by difficulty.

Hence `lean_image/versions.py`, one image per version, and `check_sandbox.py`-style gates
that compile every reference statement before a run is trusted.
