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
   stopped at `max_output_tokens` with status `incomplete` and no response error scores 0.0
   (`response_truncated`), preserving its truncation metadata and preventing regeneration on resume.
   Other incomplete responses are marked for regeneration (`response_incomplete`).
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

### Agentic code execution

`configs/critpt_custom_grader_agentic.yaml` composes this grader with Gym's Stirrup agent and a persistent Python
execution sandbox. It includes an original synthetic polynomial task. The default one-response configuration and
`answer_source: output_text` extraction remain available; the agentic recipe explicitly selects `finish_tool`.
The synthetic dataset uses Gym's `benchmark` dataset schema so its prompt template and five-repeat setting survive
dataset registration; it is an integration example, not a CritPt benchmark release.

The recipe uses three kinds of sandbox, all created through Gym's sandbox API:

- Stirrup orchestration can reach the model and resources servers. Its built-in `code_exec` is disabled.
- The resources server's `/code_exec` runs commands in a separate Linux sandbox with denied network egress. Files
  persist during one episode; each command starts a fresh shell. No grading task data is uploaded to this sandbox.
- Verification retains separate reference, candidate and comparison sandboxes. It grades only the submitted
  `finish.answer`; working files and `finish.derivation` are not inputs to grading. The saved response can be
  verified again after all generation-session cleanup has completed.

The shipped OpenSandbox policy denies all execution and grading egress. The provider must enforce that policy; a
prompt instruction alone is insufficient. The execution manager rejects providers and configurations for which it
cannot establish an offline policy. Orchestration needs network access to Gym and the model, so applying an egress
ban to the orchestration sandbox would break inference and tool routing.

#### Runtime and model configuration

Build `runtime/agentic/Dockerfile` and publish it where OpenSandbox can pull it. It provides Python 3.13, NumPy,
SciPy, SymPy and Matplotlib, plus Bash and `timeout`. Packages are installed before the sandbox starts; task commands
cannot install them from the network. The image records the Python version and complete installed package set in
`/opt/critpt-agentic-runtime/`. The pinned package versions and resource limits are choices made by this recipe.
They are not published Artificial Analysis image pins.
The image leaves the provider's exec daemon running as root so OpenSandbox can switch users. The recipe sets
`agentic_execution.sandbox_spec.user: agent`, making every model command run as the non-root `agent` account.

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
    -t <registry>/<path>/critpt-agentic-execution:<tag> --push \
    resources_servers/critpt_custom_grader/runtime/agentic
```

Resolve the image's immutable registry digest and set these three image variables:

```bash
export CRITPT_CUSTOM_GRADER_IMAGE='<registry>/<path>/grader@sha256:<digest>'
export CRITPT_AGENTIC_EXECUTION_IMAGE='<registry>/<path>/execution@sha256:<digest>'
export CRITPT_AGENTIC_ORCHESTRATION_IMAGE='<registry>/<path>/orchestration@sha256:<digest>'
```

The grading image is the existing `runtime/Dockerfile`. The orchestration image must support the existing
[Stirrup sandbox setup](../../responses_api_agents/stirrup_agent/README.md); it may be the same scientific image,
but runs as a separate sandbox with its own network policy. Stirrup installs the version used by the agent server
during setup. Record that resolved version and the Gym commit with the image digests.

The recipe deliberately requires explicit model settings: `context_window_tokens`, `max_completion_tokens_cap`,
`enable_thinking`, `temperature` and `top_p`. Supply them under
`critpt_custom_grader_stirrup_agent.responses_api_agents.stirrup_agent` in a model overlay. Use the served model's
actual context and output limits, not the defaults of another model. For a nonreasoning model the published AA
baseline is 16,384 output tokens, subject to model capacity. For a reasoning model use its supported output limit
and the intended reasoning mode. AA documents temperature 0.6 for reasoning models and 0 for nonreasoning models,
with model-specific exceptions. The endpoint may require its own reasoning-effort or thinking settings in addition
to Stirrup's `enable_thinking`; configure those in the model server and retain the resolved request settings.
With Gym's `vllm_model`, set `forward_request_chat_template_kwargs: true` in the model-server block to honor
Stirrup's per-request `enable_thinking`. The `openai_model` adapter forwards these template fields to its upstream;
use an endpoint that accepts them, or a provider adapter that translates them into its supported reasoning controls.
Validate the resolved request on each model endpoint before collecting results.

Gym sizes each call's output budget from estimated prompt tokens and context capacity, with the configured
completion buffer, minimum and cap. This is a per-call budget; it does not impose a cumulative episode token limit.

For vLLM endpoints that expose `/tokenize`, set `fit_completion_to_context: true` in the `vllm_model` block to
measure the fully rendered prompt on the selected backend before generation. To use the full available output
budget, set Stirrup's `min_completion_tokens` equal to `max_completion_tokens_cap`, using the model's supported
output capacity. The adapter then sends the smaller of that requested cap and the exact remaining context; a
smaller per-request output limit still applies. This mode preserves the conversation, reasoning settings and turn
limit. Tokenization failures, exhausted context and rejected model requests remain errors, rather than empty
completions that consume turns and become misleading no-answer scores. It requires a vLLM chat-completions
endpoint; see the [model adapter's configuration](../../responses_api_models/vllm_model/README.md).

Start the synthetic example after supplying a model overlay and the provider connection variables:

```bash
gym env start \
    --config resources_servers/critpt_custom_grader/configs/critpt_custom_grader_agentic.yaml \
    --config /path/to/model-overlay.yaml --model-type openai_model ++use_absolute_ip=true
# Run from a second terminal while the servers are ready.
gym eval run --no-serve --agent critpt_custom_grader_stirrup_agent \
    --input resources_servers/critpt_custom_grader/data/agentic_example.jsonl \
    --prompt-config resources_servers/critpt_custom_grader/prompts/agentic.yaml \
    --num-repeats 5 --output results/agentic_example_rollouts.jsonl
```

The orchestration sandbox must reach the advertised Gym addresses. The recipe starts at eight simultaneous
episodes and grades each item independently, so no complete-dataset batch barrier is required.
The synthetic dataset belongs to the resources-server block, where its task schema is validated. Its
`critpt_custom_grader_synthetic` taskset routes prepared rows to the configured environment server. Change that
dataset's `prompt_config` when supplying a different prompt through registered dataset preparation.

#### Submission and accounting

The model can call `code_exec(cmd)` and `finish(reason, derivation, answer)`. `finish` is terminal only when the
resources server accepts it. Verification requires the matching successful tool acknowledgement and its answer
digest in the saved trajectory. A rejected call, unrelated assistant text or code written during exploration is
not an accepted submission. Saved trajectories remain verifiable after execution-session cleanup.

There are at most 50 model-response turns. A completed episode that never submits an answer, an empty accepted
answer, or normal turn-budget exhaustion scores zero. Provider failures and unknown incomplete outcomes remain
unscorable, as do inconclusive verification failures. Preserve those categories and report incomplete coverage;
do not silently count infrastructure failures as incorrect answers or regenerate legitimate no-answer outcomes.
An explicit content-filter outcome with no answer also scores zero; ambiguous API rejections remain unscorable.

Five repeats mean the mean of five independently scored attempts per task (mean pass@1), not credit when any one
of five attempts succeeds. Keep the repeat count and denominator in the result report. Retry only infrastructure
failures according to the configured policy, without replacing valid failed attempts.

#### Relationship to the published CritPt v2 method

[Artificial Analysis's methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking#critpt-v2)
describes Stirrup, an offline Linux Python 3.13 environment with NumPy/SciPy/SymPy/Matplotlib, up to 50 turns, a
structured `finish` tool and five repeats. This recipe provides those execution and submission mechanisms with
local grading. The default system prompt is an original prompt for the synthetic example.

For the authors' one-step code-tools system prompt, supply an external Gym prompt YAML via `--prompt-config` (and
the dataset's `prompt_config` when running through registered datasets). Use the rendered prompt from the
[CritPt source at revision bc37a22bedb8744dff8996ccce7f9eedec548915](https://github.com/CritPt-Benchmark/CritPt/tree/bc37a22bedb8744dff8996ccce7f9eedec548915):
`templates/prompt_template_default.yaml` with the one-step configuration, and the code-tools wording applied by
`src/critpt/generation/prompts.py`. Keep that system text unchanged. The external user template should describe the
problem, code template, installed packages, actual execution limits and `finish` fields; the included user template
provides those operational instructions. Preserve the selected prompt's contents and hash with the run.

Gym's adapter retains partial responses with `finish_reason=length` or `max_tokens` and may continue within the
50-turn limit. `truncation_recovery: false` disables the thinking-off recovery instruction; it does not terminate
the episode on those responses. Each such solver response consumes a turn, and a terminal no-answer still scores zero.
Upstream Stirrup 0.1.12 instead [raises `ContextOverflowError`](https://github.com/ArtificialAnalysis/Stirrup/blob/c7edc996f13e953dff477fd1082e61ca989e871c/src/stirrup/clients/chat_completions_client.py#L174),
which can [rewind context and retry or end with an error](https://github.com/ArtificialAnalysis/Stirrup/blob/c7edc996f13e953dff477fd1082e61ca989e871c/src/stirrup/core/agent.py#L1152).
AA specifies zero for terminal no-answer outcomes, including output consumed by reasoning, but does not disclose
its precise per-call overflow or partial-tool handling. Equivalence of this adapter behavior to AA is unverified.

AA does not publish its exact image digest, package versions, Stirrup revision, full user prompt or all execution
limits. AA names E2B as its primary provider for agentic benchmarks; this recipe uses Gym's OpenSandbox provider.
The recipe records its own choices explicitly: a 300-second command deadline, four CPUs, 8 GiB memory,
50 turns and a bounded session lifetime. Matching the published generation conditions does not establish equality
with hidden grading behavior. This server uses the supplied task schema and its documented comparison policy;
it does not call an external grading API.

### Semantic answer validators

Some tasks admit several valid outputs: a protocol can be split into more stages without changing its result, and
different bases can span the same vector space. An expected-value comparison cannot establish those properties.
For these tasks, an operator can register a semantic validator in the resources-server configuration:

```yaml
critpt_custom_grader:
  resources_servers:
    critpt_custom_grader:
      semantic_validators:
        plane_basis:
          entrypoint: validate
          source: |
            def validate(*, observed, expected, args, kwargs, config):
                # Synthetic example: any integer basis for the xy-plane in R^3.
                if type(observed) is not list or len(observed) != 2:
                    return False
                if any(type(row) is not list or len(row) != 3 for row in observed):
                    return False
                if any(type(value) is not int for row in observed for value in row):
                    return False
                a, b = observed
                return a[2] == b[2] == 0 and a[0] * b[1] - a[1] * b[0] != 0
```

The task selects the registered identifier with `semantic_validator: {id: plane_basis, config: {}}`. It still supplies
a reference implementation and value expectations for its cases. Tasks cannot supply validator source, paths,
entry points or execution limits. Omitting `semantic_validator` preserves the expected-value comparison policy.

The validator receives the decoded observed value, stored expected value, original positional and keyword inputs,
and its bounded JSON configuration. Input-conversion code is not run in the comparator: symbolic text remains an
inert codec value, and callables or candidate-defined objects are not deserialized. The validator must return an
exact Python `bool`. It owns its numerical tolerances and acceptance rules; ordinary comparison tolerances are not
automatically applied to it. Define any objective target, feasibility boundary and rounding rule explicitly in the
validator or configuration.

Only the operator's selected source is delivered to the protected comparator sandbox. It runs in a fresh namespace
for each case, within the existing resource limits and comparison deadlines; module initialization is included in
that deadline. The same validator first checks the reference outputs. A rejected reference makes the task
unscorable and prevents candidate execution. After a successful preflight, a candidate rejection scores zero.
Validator exceptions, non-Boolean returns and timeouts remain unscorable, even if another case rejects the answer.
Changing validator source, entry point or task configuration invalidates the reference-preflight cache.

These are task-authoring mechanisms, not a supplied implementation of any benchmark's hidden acceptance policy.

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
161,546,240 bytes and an `RLIMIT_NPROC` of at least `execution.max_processes`, which the runner sets on every task
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

Categories. Scored: `passed`, `response_truncated`, `candidate_mismatch`, `candidate_invalid_output`, `candidate_source_limit`,
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
