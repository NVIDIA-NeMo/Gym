# lean_proof

Shared plumbing for Lean benchmarks. Not a resources server — it has no `app.py` and no config;
benchmarks import from it.

| Module | What it owns |
|---|---|
| `proof_utils.py` | `<think>` stripping, comment/string stripping, fenced-block extraction, banned declarations, whole-file statement preservation |
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

`resources_servers/leancat` is the worked example.

## Mathlib versions are not interchangeable

Compiled `.olean` files do not carry across versions, and a benchmark pointed at the wrong one
does not error at startup — it fails individual tasks with ordinary-looking "unknown
identifier" errors. Measured on LeanCat against v4.12.0: 36 of 100 reference statements fail to
compile *with their `sorry` still intact*, so the run reports a plausible number capped at
64/100 and skewed by difficulty.

Hence one image per version (`lean_image/`), and a gate that compiles every reference statement
before a run is trusted.
