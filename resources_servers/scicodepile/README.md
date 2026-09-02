# SciCodePile Resources Server

### Overview

Verifies Python solutions against the executable stratum of
[SciCodePile](https://arxiv.org/abs/2607.19104), a corpus and benchmark for
scientific code generation built from 37,737 public computational-science
repositories. This server covers the **runnable benchmark** — 200 tasks, each
shipping its own test — not the corpus's completion/infilling tasks.

- Task type: single-turn function generation + local code execution
- Domain: `coding`
- Tasks: 200 (`SciCodePile/SciCode-Runnable-Benchmark-Reviewed`)
- Reward: binary — `1.0` iff the task's own `check(candidate)` returns without raising

### Verification

Each task ships a `test` that defines `check(candidate)`. The server executes
`setup_code + model_code + test` in a single namespace, looks up the function named
by `entry_point`, and calls `check` with it. This mirrors the upstream harness.

Execution happens in a subprocess (`scp_runner.py`) with a wall-clock timeout and an
`RLIMIT_AS` cap. A fresh process per task is not just for isolation from the server:
all 200 tasks carry the upstream `env_sensitive` audit flag and 117 carry
`globals_patch`, so tests mutate global state and would otherwise contaminate
one another.

### Input schema

- `responses_create_params`: OpenAI Responses create params with the user prompt.
- `verifier_metadata` (required):
  - `test` (required): source defining `check(candidate)`.
  - `entry_point` (required): name of the function under test.
  - `setup_code`: preamble executed first; non-empty on 105 of 200 tasks.
  - `task_id`, `audit_flags`, `primary_score_eligible`: provenance only.

### Statuses

`pass`, `fail` (check raised), `entry_point_missing`, `error` (syntax or import
failure), `timeout`, `empty_output`, `no_code_block`. Only `pass` earns reward.

### The model must return a complete function

Unlike BigCodeBench, there is **no calibration prefix**. BigCodeBench can prepend
`code_prompt + "pass"` so the entry point exists even if the model returns only a
body; that is impossible here because SciCodePile's `prompt` field is display text
whose docstring is not indented under the `def` line, so it is not valid Python.

The prompt built in `benchmarks/scicodepile/prepare.py` therefore demands a complete
definition under the exact `entry_point` name, and a bare function body scores
`entry_point_missing`. All 200 upstream `canonical_solution` values are likewise
complete definitions, so this matches the benchmark's own expectation.

### Code extraction

`code_extraction.py` is a byte-identical copy of the BigCodeBench server's module,
kept in sync deliberately so a score difference between the two servers can never be
an extractor artifact. One inherited quirk is worth knowing: with an **untagged**
` ``` ` fence followed by trailing prose, extraction returns empty and the task
scores `no_code_block`. The prompt asks for a ` ```python ` tag to avoid this.

### Validation

The 200 upstream `canonical_solution` values were run through `scp_runner.py`:
**200/200 pass**. Negative controls over 30 tasks behave as required — a stub
returning `None` and a raising implementation both score `fail`, a renamed function
scores `entry_point_missing`, and unparseable code scores `error`.

Re-run this whenever the runner changes; it validates the harness with no model
involved, which is the only check here that cannot be confounded by model quality.

### Reported upstream result

The paper reports the strongest evaluated model at **12.30% Pass@1** on this
stratum. That number was produced by the upstream harness, not this server, and
this repository has not reproduced it.

### Example

```bash
gym env start \
    --model-type openai_model \
    --resources-server scicodepile

gym eval run --no-serve \
    --agent scicodepile_simple_agent \
    --input resources_servers/scicodepile/data/example.jsonl \
    --output resources_servers/scicodepile/data/example_rollouts.jsonl \
    --limit null
```

## Licensing information

Code: Apache 2.0
Data: see the upstream dataset card for `SciCodePile/SciCode-Runnable-Benchmark-Reviewed`

Dependencies
- nemo_gym: Apache 2.0
