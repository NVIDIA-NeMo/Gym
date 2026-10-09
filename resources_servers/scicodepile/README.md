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
`setup_code`, the model's code and `test` as three separate compile units sharing one
namespace, looks up the function named by `entry_point`, and calls `check` with it.
This mirrors the upstream harness. (The separation is load-bearing: concatenated, a
trailing decorator in the model's code binds to the test's own `def check`.)

That namespace is a real module registered in `sys.modules` as `__scicodepile__`,
with `__file__` pointing into the task's throwaway working directory — not a bare
dict. `@dataclass` under `from __future__ import annotations`, `pickle`, and
`multiprocessing` all resolve a function's module by name, and all of them fail
against a dict in ways that would be charged to the model.

Execution happens in a subprocess (`scp_runner.py`). A fresh process per task is not
just for isolation from the server: all 200 tasks carry the upstream `env_sensitive`
audit flag and 117 carry `globals_patch`, so tests mutate global state and would
otherwise contaminate one another. Several also write files relative to the working
directory, so each task gets a throwaway one.

### This is not a security sandbox

`code` is unreviewed model output, and the runner does **not** sandbox it. Task code
runs with the privileges and environment of the resources server, and can shell out,
open sockets, or write outside its working directory. Containment is limited to four
things: process isolation, an `RLIMIT_AS` address-space cap, a throwaway working
directory, and the parent's wall-clock timeout (enforced by killing the runner's
entire process group, so spawned children do not outlive it).

The cap is 30 GiB, matching bigcodebench. It exists to stop a runaway allocation
taking down the node, not to measure the model, so it sits well above anything a
legitimate scientific function needs — an under-sized cap surfaces as the model's
`exec_failed`, which is exactly the confound this benchmark cannot afford. For the
same reason the runner is spawned with `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1`:
every task imports numpy, and OpenBLAS reserves a per-core buffer at library load
sized from the machine's core count, which counts against `RLIMIT_AS`. Measured on a
28-core host, `import numpy` plus one SVD reserves 1.18 GiB of address space unpinned
versus 0.12 GiB pinned; the reservation grows with core count, so the margin matters
most on exactly the many-core nodes where the cap would otherwise bite. `RLIMIT_AS`
is applied on Linux only, so the tests that exercise it are Linux-only too.

Do not run untrusted rollouts on shared nodes without a real sandbox — see
`nemo_gym/sandbox/`.

The result channel is kept off file descriptor 1: the runner reports its verdict on
a private duplicate and points fd 1 at `/dev/null`. Task code owns fd 1 too, and
`redirect_stdout` rebinds only `sys.stdout`, not the descriptor, so without this an
honest task's incidental output would corrupt its own verdict.

### Leaving the runner

Once the verdict is on the channel the runner calls `os._exit(0)` rather than
returning. A normal interpreter shutdown joins non-daemon threads and runs `atexit`
hooks, both of which task code can leave behind; fd 2 is also pointed at `/dev/null`
so that anything the task spawns cannot hold the parent's stderr pipe open. Without
these, a solution whose `check` passed — six upstream tasks already use
`subprocess`/`multiprocessing` — was scored `timeout` and held a concurrency slot for
the full 120 s.

The runner is spawned with `start_new_session=True` and its whole process group is
SIGKILLed in a `finally`, so a task's children die before the parent removes the
working directory they are writing into, and a cancellation on server shutdown takes
the same path.

**That is a robustness property, not a security one — the verdict is not
tamper-proof.** Task code can still reach the result channel through another
descriptor, or replace `json.dumps` before the runner serialises. Closing that off
would need a parent-generated nonce, and even then frame introspection defeats it.
Since the runner is explicitly not a sandbox and task code may shell out or open
sockets regardless, a determined completion has cheaper options than forging a
verdict. Trust a verdict only to the extent you trust the code that produced it.

### Input schema

- `responses_create_params`: OpenAI Responses create params with the user prompt.
- `verifier_metadata` (required):
  - `test` (required): source defining `check(candidate)`.
  - `entry_point` (required): name of the function under test.
  - `setup_code`: preamble executed first; non-empty on 105 of 200 tasks.
  - `task_id`, `audit_flags`, `primary_score_eligible`: provenance only.

### Statuses

`pass`, `fail` (check raised), `entry_point_missing`, `error`, `timeout`,
`empty_output`, `no_code_block`, `malformed_task`. Only `pass` earns reward.

`malformed_task` is a dataset fault, not a model one: a row missing `test` or
`entry_point` cannot be scored. It is reported as a harness failure rather than
raised, because an exception here is an HTTP 500 and a 500 aborts the whole run.

The statuses do **not** cleanly isolate non-attempts. With no fence at all the
extractor returns the whole text, so a prose answer or a refusal is compiled and lands
as `error`/`syntax_error` — indistinguishable from a genuine attempt with a syntax
error. `no_code_block` catches only the untagged-fence case. Counting non-attempts
means looking at the text, not the status.

`error` carries a `details.reason` distinguishing `syntax_error`, `exec_failed`
(the module body raised, typically a missing import), `test_defines_no_check`,
`runner_crashed`, and `unparseable_runner_output`. `details.phase` says which compile
unit raised: `setup`, `model` or `test`.

Every string in the response is forced back into encodable UTF-8 before it is sent.
A lone surrogate — from `raise ValueError('\udcff')`, a `surrogateescape`-decoded
filename, or the `\udcff` JSON escape in the incoming request — is representable in a
Python `str` and survives the runner's JSON round trip, but raises when the response
is encoded for the wire. That would be an HTTP 500, and with the default
`route_failures_to_sidecar=False` a 500 aborts the whole rollout run. One task's
exception message must not be able to end the job.

### Harness faults

`failure_reason` is set only when `reward=0.0` does not reflect policy quality —
dataset-owned `setup_code` raising, the task's own test failing to execute or defining
no `check`, or the runner crashing before any model code ran. The rate is published as
a `harness_failure` score, so it appears as its own metric line
(`pass@1[avg-of-{k}]/harness_failure`) instead of needing a manual filter over the
rollouts. Those rollouts still score `accuracy` 0: nothing is dropped silently.

Outcomes the model can cause are **not** flagged, even though they never reach an
assertion — a `timeout` (an infinite loop is the model's), an
`unparseable_runner_output` (reachable by `os._exit` in the candidate), and a
`runner_crashed` raised after the model's module body executed (model code can rebind
a builtin the runner calls). Flagging any of these would inflate pass@1 and make
hanging or exiting reward-neutral under RL.

One hole remains open by construction: the model's code runs before the test's module
body, so a model that deliberately breaks the test earns `test_code_failed`. Watch the
`harness_failure` rate rather than assuming it is zero.

### The model must return a complete function

Unlike BigCodeBench, there is **no calibration prefix**. BigCodeBench can prepend
`code_prompt + "pass"` so the entry point exists even if the model returns only a
body; that is impossible here because SciCodePile's `prompt` field is display text
whose docstring is not indented under the `def` line, so it is not valid Python.

The server therefore looks `entry_point` up in the namespace produced by executing
the model's code, and a bare function body scores `entry_point_missing`. All 200
upstream `canonical_solution` values are complete definitions, so this matches the
benchmark's own expectation.

The benchmark deliberately does **not** instruct the model to do this — it passes
the upstream prompt through unmodified, because upstream publishes no prompt to
match and a wrapper would change what is measured. See
[`benchmarks/scicodepile/README.md`](../../benchmarks/scicodepile/README.md#prompting)
for that reasoning and the non-attempt rate it costs.

### Code extraction

`code_extraction.py` is duplicated from the BigCodeBench server's module and
currently differs from it only in the docstring. Keeping the two identical is what
stops a score difference between the servers being an extractor artifact — but
nothing enforces it: there is no shared import and no test comparing them, so an
edit to either file silently breaks the invariant. One inherited quirk is worth
knowing: with an
**untagged** ` ``` ` fence followed by trailing prose, extraction returns empty and
the task scores `no_code_block`. Nothing asks the model for a ` ```python ` tag —
the upstream prompt is passed through unmodified (see above), so untagged fences are
part of the non-attempt rate that choice costs.

### Validation

The 200 upstream `canonical_solution` values were run through `scp_runner.py`:
**200/200 pass**. Negative controls were run over all 200 tasks (not a 30-task
sample) and behave as required:

| Control | Result |
| --- | --- |
| raising implementation | `fail` 200/200 |
| unparseable code | `error` 200/200 |
| stub returning `None` | `fail` 197/200, **`pass` 3/200** |
| renamed function | `entry_point_missing` 199/200, `fail` 1/200 |
| empty answer | `entry_point_missing` 199/200, `fail` 1/200 |

Two caveats fall out of running the controls over the whole set:

- **`alignment/python/144`, `178` and `273` pass with a stub that returns `None`.**
  Their tests only assert that the return value is `None` or that some object was
  produced, so they cannot distinguish a real implementation from an empty one.
  Three tasks out of 200 are worth 1.5 percentage points of any score reported here.
- **`alignment/python/76` defines its own entry point (`gsea`) in `setup_code`.** The
  function therefore exists even when the model returns nothing, which is why the
  renamed-function and empty-answer controls score `fail` rather than
  `entry_point_missing` on that one task. It does not pass, so nothing is scored
  wrongly — but `entry_point_missing` is not a reliable non-attempt signal there.

Re-run this whenever the runner changes; it validates the harness with no model
involved, which is the only check here that cannot be confounded by model quality.

### Reported upstream result

[The paper](https://arxiv.org/abs/2607.19104) evaluates 15 models on this stratum
and reports Pass@1 and Pass@5 using the HumanEval estimator (Table 3). The strongest
is **GPT-5.4-mini at 12.30% Pass@1 / 15.50% Pass@5**, followed by o3-mini
(10.50% / 12.00%) and DeepSeek-R1 (8.50% / 12.50%); the weakest reported are
StarCoder2-7B/15B at 0.30% Pass@1.

Models are evaluated zero-shot on the task `prompt` alone — "HumanEval-style task
prompts, each consisting of a function signature and its natural-language
specification". The paper publishes no system prompt, no decoding parameters, and no
sample count per task.

These figures were produced by the upstream harness, not by this server.

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
