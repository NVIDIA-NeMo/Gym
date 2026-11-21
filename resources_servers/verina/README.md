# Verina: verified code generation in Lean 4

The model gets a programming task together with its formal specification, a precondition and a postcondition
written in Lean 4. It writes the function, and a proof that the function meets the specification. The verifier
compiles the answer with Lean and Mathlib, runs the task's hidden test cases against the function, and checks which
axioms the proof depends on.

The other Lean environments in Gym ask for proofs of mathematical statements or of existing verification
obligations. Here the model writes the program as well as its proof, which trains correct-by-construction
programming.

## Task format

Each Verina task is a Lean file with tagged regions. The model fills four of them and sees the rest:

| Region | Written by | Content |
| --- | --- | --- |
| `precond`, `postcond`, their helpers, `solution_aux` | task | the specification and the helper definitions it uses |
| `code_aux` | model | optional helper definitions for the function |
| `code` | model | the function body, the expression after `:=` |
| `proof_aux` | model | optional helper lemmas |
| `proof` | model | the tactics that prove `<name>_spec_satisfied` |

The prompt shows the task file with `{{code}}`-style placeholders and asks for the answer in `<code_aux>`, `<code>`,
`<proof_aux>` and `<proof>` tags. Each row of `data/*.jsonl` carries the task file with the solution regions
removed (`lean_template`), the function and theorem names, and the test cases rendered as typed Lean terms
(`tests`). The reference solution is not in the rows.

## Reward

| Outcome | Reward |
| --- | --- |
| Every test passes and the proof checks | `1.0` |
| Every test passes, but the proof is missing, incomplete or rejected | `tests_only_reward` (default `0.5`) |
| Anything else | `0.0` |

A proof checks when the whole file compiles without errors and `#print axioms` lists only `propext`,
`Classical.choice` and `Quot.sound`. That rejects `sorry`, `admit`, `native_decide` (which adds
`Lean.ofReduceBool`), and axioms the answer declares. A full proof also needs the tests to pass, so a proof cannot
earn full reward through a specification that admits a wrong answer.

The partial reward matters because Verina ships 143 of its 189 reference proofs as `sorry`: for most tasks, a
complete proof is beyond what the benchmark's authors wrote. Set `tests_only_reward: 0.0` to reward only verified
solutions.

### How the checks resist gaming

- Tests run as `#eval` commands appended after the answer. Each prints a pass or fail marker, and the verifier only
  accepts a marker reported on that test's own line, so output the answer prints cannot pass a test, and an answer
  that stops the file early (`#exit`) passes nothing.
- Tests supply the function's precondition argument through an axiom declared after the theorem, so the theorem
  cannot depend on it.
- Declaring the theorem early with an easier statement makes the task's theorem a duplicate, which is an error, so
  the file no longer compiles cleanly.
- Answers may not contain new syntax or notation (which could change what the specification means), `instance`
  declarations (which could redefine the `==` the tests use), commands that run code (`#eval`, `run_cmd`, ...),
  `implemented_by`, `extern`, `unsafe`, or `debug.skipKernelTC`. None of these appear in Verina's reference
  solutions.

The answer's code runs inside Lean while the file is checked. The checks above block the direct ways to run
arbitrary code, but they are not a sandbox: run the server in an isolated container, as for other environments that
execute model code.

## Setup

The server installs Lean on first start (see `setup_lean.py`): elan, the Lean toolchain `v4.18.0` that Verina pins,
and a Lake project with Mathlib `v4.18.0`, whose prebuilt `.olean` files come from Mathlib's cache. The workspace
takes about 7 GB and lives in `resources_servers/verina/.lean_workspace/` (set `VERINA_LEAN_WORKSPACE` to move it).
To install it ahead of time:

```bash
python resources_servers/verina/setup_lean.py
```

Prepare the data (downloads Verina at a pinned commit and writes `data/train.jsonl`, `data/validation.jsonl` and
`data/reference_answers.jsonl`):

```bash
python -m resources_servers.verina.prepare_verina
```

The split is fixed by a hash of each task id: 155 training tasks and 34 validation tasks. Verina is also used as a
benchmark, so do not report Verina results for a model trained on its training split without saying so.

## Running

```bash
gym env start --resources-server verina --model-type vllm_model

gym eval run --no-serve \
    --agent verina_simple_agent \
    --input resources_servers/verina/data/example.jsonl \
    --output resources_servers/verina/data/example_rollouts.jsonl
```

Each check starts a Lean process that imports Mathlib, which takes several seconds and maps about 5 GB of `.olean`
files (shared between processes through the page cache). `max_concurrency` bounds the number of Lean processes and
defaults to half the CPU count. `timeout_s` (default 120 s) stops an answer that loops forever, and `memory_limit_gb`
(default 8) caps each Lean process's heap, so an answer that allocates without bound aborts instead of exhausting the
machine's memory; both score 0.

## Validation

`validate_references.py` scores Verina's reference solutions, and broken variants of them, through the real
verifier:

```bash
python -m resources_servers.verina.validate_references
```

On all 189 tasks, 180 reference solutions score as expected: 41 have complete proofs and score `1.0`, the rest ship
their proof as `sorry` and score `0.5`, and the `sorry`, missing-proof and wrong-code variants score as described in
the script. The other 9 were written against a newer Lean than the `v4.18.0` Verina pins: `verina_advanced_28`,
`verina_advanced_55`, `verina_advanced_74` and `verina_basic_22` use `Std.HashSet.emptyWithCapacity` (score `0.0`),
and the proofs of `verina_basic_16`, `verina_basic_17`, `verina_basic_31`, `verina_basic_36` and `verina_basic_92` fail
(score `0.5`).

`data/example_rollouts.jsonl` holds 20 rollouts (the 5 examples, 4 each) of `nvidia/Nemotron-Mini-4B-Instruct`
served by vLLM. All score `0.0`: the model answers in Python-like pseudocode that does not compile as Lean.

## Licensing

- Code: Apache 2.0
- Data: Verina, Apache 2.0 ([sunblaze-ucb/verina](https://github.com/sunblaze-ucb/verina), commit
  `66b3ce5014edafb20cb26b87c681e7f31847249f`; paper: [arXiv:2505.23135](https://arxiv.org/abs/2505.23135))
- Lean 4 and Mathlib: Apache 2.0
