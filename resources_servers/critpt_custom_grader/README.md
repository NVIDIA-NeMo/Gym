# CritPt Custom Grader Resources Server

A sandboxed grader for CritPt-like tasks with operator-supplied task data: Python problems whose answer is a function,
graded against test cases and stored expectations, with numeric tolerances, exact integers, composite values, symbolic
expressions, categorical strings and expected exceptions. Every reference, candidate and comparator step runs in a
sandbox created through Gym's provider-neutral sandbox API, so the same server runs on OpenSandbox, docker, apptainer,
enroot or the `local` provider by swapping one provider config; every provider but `local` also needs a build of
`runtime/Dockerfile` named in `CRITPT_CUSTOM_GRADER_IMAGE` (see [The grading image](#the-grading-image)). It does not
call an external grading service and
claims no parity with any hidden grader. It is a separate server from `resources_servers/critpt` and imports nothing
from it.

## What it grades and how

1. `/verify` receives an already generated response. The grader never generates candidates.
2. Extraction: first ```` ```python ```` fence, else first bare fence, else the raw text. A completed response with no
   usable assistant text, or a source that is empty, not valid UTF-8, holds a NUL byte or exceeds
   `max_candidate_source_bytes`, scores 0.0 (`candidate_source_limit`) before any sandbox is created. A response that
   did not complete is marked for regeneration (`response_incomplete`).
3. A reference preflight runs the task's reference solution and compares every outcome with the stored expectations.
   Candidate code runs only after the preflight passes. The preflight outcome is cached per task, so a task's
   reference runs once per server process (`reference_cache_dir` also keeps entries on disk for later processes).
4. The candidate runs in its own sandbox with only the source, the normalized inputs, the public runner package and
   the operator's `sandbox_spec.files`. It never sees expectations or tolerances.
5. The comparator, in a third sandbox, judges the candidate's outcomes per leaf. Tolerance precedence: an authored
   per-leaf entry or task setting (`rtol`, `atol`) wins; otherwise the accuracy promise the problem statement makes
   for the leaf, floored at the strict default (`rtol 5e-12`, `atol 0`); otherwise exactness for an integer
   expectation; otherwise the server default (`comparison.default_rtol` and `default_atol`, shipped as `5e-12` and
   `0`). A setting that gives only one of `rtol` and `atol` takes the strict default for the other. The statement is
   read by a narrow extractor, described below.
6. Symbolic answers use a bounded grammar built directly into SymPy. Two expressions are equal only when shown equal:
   structural identity, simplification, SymPy's assumptions, an exact polynomial test, bounded deterministic rewrites
   or, for a difference with no free symbols, a 50-digit evaluation (relative tolerance 1e-9); fixed positive real
   sample points can only disprove. A float never equals a symbolic constant, even its rounding to the nearest
   double (a symbolic expectation requires a symbolic answer, and numeric expectations carry tolerances); the
   50-digit evaluation decides only differences that are exact constants on both sides. A pair that none of these
   decides is never guessed: that case is uncertain (see 7), except that an expression which cannot be shown
   equal to a numeric expectation is a mismatch.
7. Candidate faults after a successful preflight (source error, suite timeout, exception, encoding error, worker
   abort) score 0.0, and so does a value the comparator finds wrong or invalid (`candidate_mismatch`,
   `candidate_invalid_output`). Reference faults, task-data defects and comparator uncertainty are unscorable and
   masked, with one exception inside a row: a decided candidate-side failure on any case (a mismatch or an invalid
   output) outranks candidate-side uncertainty on another case. The first failing case then names the category, and an
   uncertain case is passed over, never counted equal. Only when no case fails outright does one uncertain case make
   the row unscorable (`comparator_uncertain`). A reference fault or defect ends the row at preflight, before the
   candidate runs; in the candidate batch an expectation defect outranks a decided failure and uncertainty and leaves
   the row unscorable. A comparison that overruns its per-case budget (`execution.compare_per_case_s`) leaves that
   case uncertain; in the candidate batch it therefore makes the row unscorable only when no other case fails
   outright. A candidate batch that overruns its whole compare deadline is `candidate_invalid_output` at the case it
   timed out on, which currently outranks a decided failure on an earlier case: a known limitation that changes only
   the failure category and the first failed case, never whether the row passes.

### Accuracy promises in the problem statement

The comparator reads the statement (`problem`) for accuracy promises with a narrow, fixed set of patterns; anything
else stays unrecognised rather than guessed at.

- A number counts only when an accuracy keyword (error, accuracy, tolerance, precision, decimal places, significant
  figures) claims it and its kind is stated. A scientific or power-of-ten number needs the word "absolute" or
  "relative" near its keyword, in the same sentence; a percentage is relative; a digit count carries its own kind. The
  value must lie above 1e-15 and at most 0.1.
- "within", "correct to" and "agree to" claim nothing themselves. They borrow the claim and the kind of an accuracy
  keyword in their own sentence that has no number of its own, so "relative error within 1e-6" and "tolerance: within
  1%" count, while a bare "within 1%" does not. A sentence break stops a clause, but a colon right after an accuracy
  keyword on the same line does not. A soft line wrap inside a sentence is not a sentence break: a newline reads as a
  space when the line before it does not end in `.`, `;`, `:`, `?` or `!` and is no heading, table row or fence, and
  the next line goes on with a lowercase ASCII letter, a digit, `$`, a backslash or `(` without starting a list item
  or another Markdown block.
- An error or failure rate is a measured quantity, not a tolerance: "physical error rate 0.1%" counts for nothing.
- The promise that governs a leaf is the clause that names the leaf's key, else the loosest honoured clause.
- A refused clause is visible only through `statement_promises`, an inspection helper; the `/verify` response carries
  no clause data. `statement_promises` in `comparator.py` lists every clause it recognised, with the reason for each
  refusal (`no_accuracy_keyword`, `no_kind`, `below_float64_resolution`, `too_loose_to_be_an_accuracy_claim`,
  `unsupported_number`).

## Task data format

Rows carry `responses_create_params` (the prompt) and the task fields, either flat, under `verifier_metadata`, or
under `task_data`. Task fields: `problem_id`, `reference_source` (alias `reference_code`), `entrypoint` (alias
`entry`), optional `reference_entrypoint`, `test_cases` (alias `testcases`), optional `problem` (the statement, read
for accuracy promises), `code_template`, `uuid`, `rtol`/`atol`, per-leaf `tolerances`, `input_conversions`. Test
cases are `{"args": [...], "kwargs": {...}, "expected": {"kind": "value", "value": <wire>}}` or
`{"expected": {"kind": "exception"}}`; legacy `{"input": [...], "output": ...}` cases are converted. Values travel in
a tagged wire format (`critpt-value-v1`) that keeps integers and decimals exact; `data/example.jsonl` holds synthetic
rows. Framework row keys (`agent_ref`, `task_source`, every `_ng_*` key) are stripped before validation; any other
unknown key is a task-data defect.

## Running it

Pick a provider config and a model; the provider block binds the name `sandbox` the server references:

```bash
export CRITPT_CUSTOM_GRADER_IMAGE=<registry>/<path>:<tag>  # your build of runtime/Dockerfile (see The grading image)
gym env start \
    --config resources_servers/critpt_custom_grader/configs/critpt_custom_grader.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --model-type openai_model
# gym env start stays in the foreground: run the next command from a second terminal while the servers are up
gym eval run --no-serve +agent_name=critpt_custom_grader_simple_agent \
    +input_jsonl_fpath=resources_servers/critpt_custom_grader/data/example.jsonl \
    +output_jsonl_fpath=results/example_rollouts.jsonl '+limit=null'
```

Provider credentials come from the provider's own config (OpenSandbox: `OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY`).
`CRITPT_CUSTOM_GRADER_IMAGE` names the grading image; none is shipped (see [The grading image](#the-grading-image)).

### Startup check

Before it reports ready, the server starts one sandbox from the operator's `sandbox_spec` and checks that commands run
as a non-root identity under `execution.exec_user`, that the provider accepts that identity, that the interpreter
imports the pinned packages (`pydantic`, `sympy`, `numpy`, `scipy`), and that an upload to a nested path creates its
parent directories, is found, with a plain-digit size, by the grader's own size command (`test -f`, then
`stat -c %s`), and comes back byte for byte through the grader's own bounded download. The sandbox is stopped
whatever the outcome. A failed check refuses to serve and names one of seven categories:

- `image_missing`: `sandbox_spec.image` is unset or empty on a provider other than `local` (a provider-side `snapshot_id` or
  `template` does not count: the reference cache names the runtime by its image); checked before any sandbox starts.
- `exec_user_unsupported`: the provider refuses `execution.exec_user`, or `id -u` reports an error or no uid.
- `exec_user_is_root`: commands run as uid 0, which the runner refuses.
- `interpreter_missing_dependency`: the import check reports an error: the interpreter cannot import the packages.
- `provider_create`: the sandbox could not be created and set up within `execution.create_timeout_s`.
- `upload_unsupported`: the nested upload failed; the size command raised, timed out, or did not find the file and
  print plain digits; or the bounded download failed or returned other bytes.
- `provider_exec`: only the `id -u` or the import-check command raised, or the host's bounded wait on it expired.

`startup_probe: false` skips the startup sandbox and its checks, but not the `image_missing` check. Gym gives every
server `server_spinup_timeout_seconds` (default 600 s) to become ready and the probe's sandbox create counts against
it, so on a provider with a slow first image pull raise it above `execution.create_timeout_s` plus a margin, for
example `++server_spinup_timeout_seconds=1200`.

### Providers

| Provider | Status | `exec_user` | Memory and ttl enforced |
|---|---|---|---|
| opensandbox | validated end to end with an image built from `runtime/Dockerfile` | `su` wrapper | yes |
| local | CI end-to-end test | must be `null` | no |
| docker | supported | `--user` | yes |
| apptainer, enroot | supported | `su` (needs root inside the container) | no |
| daytona | selectable, not validated | only `create.os_user` | memory yes; `ttl_s` not read |

- opensandbox: self-signed endpoints work with the shipped `tls_verify: false`.
- local: the host process must be non-root; no isolation: trusted inputs only.
- docker: set `sandbox.docker.create.network: none` for isolation.
- enroot: with `exec_user: null` commands run as the invoking user (root when the container is started with
  `create.remap_root`, which the runner refuses); a named `exec_user` goes through `su` and needs root inside the
  container.
- daytona: `ttl_s` is not read; the stop, archive and delete intervals (`create.auto_stop_interval`,
  `create.auto_archive_interval`, `create.auto_delete_interval`) are the backstop. A blocking exec may not survive very
  long runs through the Daytona proxy.

### The grading image

No image is shipped: build `runtime/Dockerfile` and publish it where your provider pulls from. `runtime/Dockerfile`:
`python:3.13-slim`, `runtime/requirements.txt` pinned in the interpreter's own site-packages (the runner runs with
`-I`), a `grader` user (uid 1000), `/work` owned by it.

1. Build it multi-arch and push it:

   ```bash
   docker buildx build --platform linux/amd64,linux/arm64 \
       -t <registry>/<path>:<tag> --push resources_servers/critpt_custom_grader/runtime
   ```

   With the `docker` provider on the host that builds it, a plain
   `docker build -t <registry>/<path>:<tag> resources_servers/critpt_custom_grader/runtime` is enough; apptainer and
   enroot pull a bare reference from a registry (`docker://`), so push the image for them too.
2. Point the server at it: `export CRITPT_CUSTOM_GRADER_IMAGE=<registry>/<path>:<tag>`, which the shipped config reads
   into `sandbox_spec.image`, or set `sandbox_spec.image` directly. Pinning by digest
   (`<registry>/<path>@sha256:<digest>`) is recommended, since a tag can be pushed again with other contents.

The `local` provider runs the host interpreter and needs no image. On every other provider the server refuses to start
without `sandbox_spec.image` (`image_missing`), even where the provider could start without one, so snapshot- or
template-only setups are not supported: a `provider_options.snapshot_id` (daytona, opensandbox), an E2B
`create.template` or an OpenShell gateway's default image. The reference cache names the runtime by its image and
could not tell such runtimes apart. Daytona also refuses a `snapshot_id` beside an image, so leave it unset there. On
E2B, map the image to its template with `create.template_map`.

Network isolation is a provider setting. The sandbox runtime's hard limits must allow an `RLIMIT_FSIZE` of at least
156,303,360 bytes and an `RLIMIT_NPROC` of at least `execution.max_processes`, which the runner sets on every task
process; otherwise every run fails as `runner_error: limits` (non-terminal).

### Configuration

`sandbox_provider` (named block or inline `{provider: {...}}`), `sandbox_spec` (`image`, `workdir`, `ready_timeout_s`,
`ttl_s`, `resources.cpu`, `resources.memory_mib`, `env`, `files`, `metadata`, `provider_options`), `execution`
(`interpreter`, `workdir`, `exec_user`, `memory_limit_mib`, `cpu_time_limit_s`, `max_processes`, `suite_timeout_s`,
`compare_per_case_s`, `compare_timeout_s`, `create_timeout_s`, `transfer_timeout_s`, `exec_timeout_margin_s`,
`job_timeout_s`, `cleanup_timeout_s`, `max_candidate_source_bytes`, `max_concurrent_jobs`), `comparison`
(`default_rtol`, `default_atol`), `reference_cache_dir` (memory only when null) and `startup_probe`.

`sandbox_spec.files` maps an absolute sandbox path to text that the sandbox API uploads when a sandbox starts; every
grading sandbox gets the same files, the candidate's included. `sandbox_spec.workdir` must exist in the image
(OpenSandbox does not create it), each grading domain works in a directory below it, and `execution.workdir` must equal
it.

Startup refuses a configuration whose policy disagrees with the spec or whose bounds cannot hold one job: an
`execution.create_timeout_s` below `sandbox_spec.ready_timeout_s + 2 * execution.transfer_timeout_s` (a null
`ready_timeout_s` counts as 0), an `execution.workdir` that differs from `sandbox_spec.workdir`, and, when
`sandbox_spec.ttl_s` is set, a `ttl_s` below the job deadline plus `execution.cleanup_timeout_s`. The job deadline is
`execution.job_timeout_s`, or the sum of the bounds one job awaits when it is null; an explicit `job_timeout_s` may not
be below that sum. The server block forbids unknown keys, nested blocks included, so a misspelt `sandbox_provider`
fails startup instead of silently selecting the default `sandbox` block; it still accepts the keys Gym itself reads
from a resources-server block, such as `allowed_agents` and `allowed_model_types`.

The shipped values: 1800 s per suite (`suite_timeout_s`, with the same `cpu_time_limit_s`), 1536 MiB of address space
per process, 64 processes, 20 s per comparison case in batches capped at 600 s, a 600 s readiness wait and four jobs at
once. BLAS and OpenMP are always limited to one thread. `max_concurrent_jobs` bounds jobs inside this process; Gym's
`num_samples_in_parallel` bounds the client side.

## Verify response contract

`reward` 0.0 or 1.0; `scored`; `mask_sample` (true on every unscored row); `category`; `cleanup_status`;
`sandbox_observations` (per-domain provider, sandbox id, outcome, exit code, wall time); `failure_kind` and
`failure_reason` on unscored rows. An unscored row also carries `_ng_failure_class` and `_ng_failure_subcategory`. Only
a terminal unscorable row carries `_ng_failure_terminal` (true): a non-terminal row never has it, and a scored row
carries no `_ng_failure_*` key.

`cleanup_status` is `resolved` when no sandbox the row created is left running and `unresolved` when a stop failed: the
sandbox then lives until the provider-side backstop reaps it, which is `sandbox_spec.ttl_s` where the provider enforces
it (on Daytona, which does not read `ttl_s`, the stop, archive and delete intervals are the equivalent). `not_started`
means the row was answered before any sandbox was created; `unknown` means the grader's result could not be read or the
grader call itself raised.

`/verify` and `/seed_session` answer a request they cannot process with a category-only body,
`{"error": {"category": ...}}`:

| Status | Category | When |
|---|---|---|
| 413 | `request_limit` | the request body is larger than 10 MiB |
| 422 | `request_invalid` | not a JSON object, or a bad request shape (see below) |
| 422 | `task_invalid` | `/seed_session` only: the task fields fail validation |
| 503 | `server_not_ready` | `/verify` with the grader not running, once the request, task and source checks pass |

A bad request shape is invalid prompt parameters, tools or `tool_choice`, or, on `/verify`, a `response` that is not a
valid response. `responses_create_params` or `response` above 1 MiB, or beyond the JSON node, depth or type limits (an
integer wider than 63 bits, a non-finite number), is `request_invalid` too; only a body above 10 MiB is
`request_limit`. On `/verify` an invalid task is not an error status: it is a 200 row with category `task_invalid`,
terminal. `server_not_ready` is answered only after the request, task and source checks have passed, so while the
grader is not running (before startup finishes or after shutdown) an invalid task, an incomplete response or an
unusable source still gets its 200 row.

Categories. Scored: `passed`, `candidate_mismatch`, `candidate_invalid_output`, `candidate_source_limit`,
`candidate_source_error`, `candidate_timeout`, `candidate_exception`, `candidate_encoding_error`,
`candidate_worker_abort`. Unscorable, terminal (`verifier_error`): `task_invalid` (a row that fails task
validation, or a stored expectation or policy the comparator cannot represent) and `reference_mismatch` (the
reference does not reproduce its own expectation). Unscorable, non-terminal (`provider_unavailable`, re-graded by
`gym eval reverify`): `exec_user_unsupported` (the provider refused `execution.exec_user`), `reference_error`,
`reference_timeout`, `comparator_uncertain`, `provider_create`, `provider_exec`, `transfer_limit`, `result_invalid`,
`*_no_result`, `job_deadline`, `server_draining`, `execution_unknown`. No usable source: category
`response_incomplete`, failure class `critpt_custom_grader:response_incomplete` (regenerate the candidate).

The reference cache holds a passed preflight and exactly two refusals, `task_invalid` and `reference_mismatch`: facts
about the task itself. Every other unscorable category describes the infrastructure or the configuration, so it is
never cached. The key covers the task, the policy limits that reach the runner, the runtime files, the operator's
`sandbox_spec.files` and `sandbox_spec.env`, the default tolerance and the runtime the reference ran in (the
provider's class, the image, `execution.interpreter` and `execution.exec_user`), so an entry persisted under one
provider or image is never read under another.

## Recovery

The server is stateless (`/reverify_mode` is `stateless`). Non-terminal rows sit in `<rollouts>_failures.jsonl`, with
one row per attempt. Take the latest attempt of each rollout that is not already in `<rollouts>.jsonl`, keep the
`provider_unavailable` ones and re-grade them without a model call:

```bash
jq -cn --slurpfile ok <(jq -c '[._ng_task_index, ._ng_rollout_index]' <rollouts>.jsonl) '
  ($ok | map({key: tostring, value: true}) | from_entries) as $done
  | reduce inputs as $r ({}; .[([$r._ng_task_index, $r._ng_rollout_index] | tostring)] = $r)
  | to_entries[] | select($done[.key] | not) | .value
  | select(._ng_failure_class == "provider_unavailable")' \
  <rollouts>_failures.jsonl > <retry>.jsonl
# Stop the gym env start servers first: reverify launches its own server stack, with a head server on the same port.
gym eval reverify --config resources_servers/critpt_custom_grader/configs/critpt_custom_grader.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml --model-type openai_model \
    '+model_endpoint_readiness_timeout_seconds=0' \
    --inputs <rollouts>_materialized_inputs.jsonl --rollouts <retry>.jsonl --output <recovered>.jsonl
# Run gym env start again with the same configs: aggregate reads the config from the head server and posts to each
# agent server's /aggregate_metrics endpoint.
gym eval aggregate -i "'<rollouts>.jsonl,<recovered>.jsonl'" -o <merged>.jsonl
```

Sandboxes are single-use and stopped in a `finally`; `sandbox_spec.ttl_s`, where the provider enforces it, and the
metadata labels (`ng-grader-job`, `ng-grader-domain`) are the provider-side backstop. Daytona does not read `ttl_s`;
its stop, archive and delete intervals (`create.auto_stop_interval`, `create.auto_archive_interval`,
`create.auto_delete_interval`) are its equivalent.
OpenSandbox operators can reap leftovers with `nemo_gym/sandbox/providers/opensandbox/cleanup_sandboxes.py`.

## Integrity model

A result file is read only behind a process exit the provider reports, is shape-checked without decoding values, and
must carry the paired status and the nonce the host wrote. None of this authenticates a result: same-UID code in the
candidate sandbox can forge its own failure and its own unscorability, never a pass, because expectations never enter
that sandbox and the comparator alone decides. Killing the runner's supervisor is `candidate_no_result`, a corrupt or
oversized result file is `result_invalid` or `transfer_limit`, and a `runner_error` envelope (`result_limit`, for
example) is `provider_exec`; each of these rows is masked out of `accuracy_scorable`. Where candidates may be
adversarial, read `accuracy_all`, which counts every unscorable row as not passed. The runner refuses to run as uid 0.

## Tests

`gym env test --resources-server critpt_custom_grader +should_validate_data=true` runs the host-safe suites (codec,
comparator, symbolic, runner, execution over an in-memory provider, the HTTP adapter, the config) and a real grade of
the example rows through the `local` provider when the host is Linux, non-root and has procfs. The hostile runner suite
runs deliberately hostile code: set `NG_CRITPT_RUNNER_REMOTE=1` only inside a disposable non-root sandbox with procfs,
never on a shared host.

## Licensing

Apache-2.0. `runtime/requirements.txt` pins the runtime packages; nothing is vendored, and a distributed image must
retain the upstream license notices of those packages.
