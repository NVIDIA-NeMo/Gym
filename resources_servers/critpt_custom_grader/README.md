# CritPt Custom Grader Resources Server

Grades a model's Python solution against **stored expectations** by executing code in
isolated Daytona sandboxes. The stored expectations are the oracle. The candidate never
receives them.

- Task type: single-turn. The agent emits one solution. The server executes and scores it.
- Domain: `coding`.
- Value kinds: exact, tolerance, composite, and symbolic.
- Reward: binary. `1.0` when every scorable case matches, `0.0` on a demonstrated mismatch.

## What it grades and how

Each job runs three **separately owned** Daytona sandboxes in a fixed order:

1. **Host bounds check.** The server validates the task and the candidate source on the host.
   It parses nothing and executes nothing here.
2. **Reference sandbox.** The stored reference source runs and produces one outcome per case.
3. **Comparator sandbox (preflight).** The comparator checks every reference outcome against
   the stored expectations. A reference outcome that contradicts an expectation, or that the
   comparator cannot represent, stops the job. The candidate never runs.
4. **Candidate sandbox.** Only after a clean preflight, the candidate source runs. It receives
   its own source, the normalized inputs, and the public runner package. It receives no
   expectations, no tolerances, no reference source, and no other case's material.
5. **Comparator sandbox (candidate).** The same comparator sandbox judges the candidate output.
6. **Cleanup.** Every owned sandbox is deleted, and the deletion is confirmed.

The comparator is the sole verdict authority. It runs no task code and holds the stored
expectations, which come from the task only and are never derived from a run. Reported values
from any sandbox are unauthenticated observations. See "Integrity model" below.

## Task data format

A dataset row has two parts: `responses_create_params` (the model query) and the task grading
spec. The spec may be flat on the row, nested under `verifier_metadata`, or nested under
`task_data`. All three validate to the same schema (`task_data.py`).

### Task fields

| Field | Meaning |
|---|---|
| `schema_version` | Must be `1`. |
| `problem_id` | Task identifier (1 to 256 characters). |
| `reference_source` | Trusted reference solution source. Alias: `reference_code`. |
| `entrypoint` | The function the candidate must define. A Python identifier. Alias: `entry`. |
| `reference_entrypoint` | Optional. The reference's function. Defaults to `entrypoint`. |
| `test_cases` | 1 to 64 cases. Alias: `testcases`. |
| `problem` | Optional problem statement text. The comparator reads precision clauses from it. |
| `code_template` | Optional. Provenance only. Never used to discover an entrypoint. |
| `uuid` | Optional. |
| `rtol`, `atol` | Optional task-level tolerance defaults. |
| `tolerances` | Optional task-level per-leaf overrides, each with a `path` and `rtol`/`atol`. |
| `input_conversions` | Optional. One entry per positional argument, at most 128: `null`, `"symbol"`, or `"function"`. Before each call, `"symbol"` turns a string argument into `sympy.Symbol(name)` and `"function"` into `sympy.Function(name)`. Other entries leave the argument alone. Keyword arguments are never converted. The reference and the candidate get the same conversions. |

### Test case shapes

A canonical case carries `args` (positional), `kwargs` (keyword), `expected`, and optional
per-leaf `tolerances`. `expected` is either a value or an exception.

Three delivered aliases are also accepted, exactly:

- `{input, output}` — positional inputs and an expected value.
- `{input, expected_error}` — positional inputs and an expected exception.
- `{inputs, expected_output}` — keyword inputs and an expected value. An optional
  `comparison_type` key is accepted on this shape only. It is text of at most 64 bytes or null,
  validated and then dropped. No other unknown key is accepted on any shape.

Inside an argument or an expected value, a map of exactly `{"__complex__": [re, im]}` whose two
parts are finite reals is read as a complex number. Any other `__complex__` map stays a plain map.

### Value kinds and tolerance policy

Values are tagged on the wire (`codec.py`): null, bool, int, float, fraction, decimal,
nonfinite, complex, list, tuple, map, set, and text tagged `str`, `symbolic`, or `legacy`.
A SymPy expression that contains floating-point atoms is carried as `symbolic` text, the form the
reference stores, and its decimal literals are read back exactly. A bare finite SymPy `Float` is
carried as a float.

- **Exact.** A categorical string expectation always compares exactly. An integer expectation
  compares exactly unless a task-level tolerance or a statement promise applies to that leaf.
- **Tolerance.** A numeric leaf passes when `|a - b| <= atol + rtol * |b|`, resolved separately
  for every expected leaf.
- **Composite.** Lists, tuples, and maps compare element by element. List and tuple transport
  differences are ignored. A complex value is one leaf, each component checked against its own
  reference part. With `atol` 0 a near-zero expected component is held to near-exact equality.
- **Unordered.** A `set` matches a `set` or a plain list one-to-one without order, each member
  matching exactly one member of the other side. A `set` against a scalar, complex, or map
  expectation is a type mismatch. An undecided or budget-exhausted match is not scored
  (`unordered_undecided`, `unordered_budget`). It is never a guess.
- **Symbolic.** `symbolic` text is compared by algebraic equivalence (`symbolic.py`). `legacy`
  text parses as a decimal or rational literal, then as bounded arithmetic notation, otherwise as
  exact categorical text.

Tolerance resolution for a leaf, in order: a non-emptied precision promise the statement makes
for that leaf governs it, tighter or looser than an authored entry. Where the statement is
silent, the leaf entry governs, else the task setting, else the server default, or exact for an
integer expectation. A leaf entry whose path names no numeric or complex leaf is unused and
dropped, not a defect. A malformed entry, one with no tolerance value or an unknown field, is an
expected defect. A nonfinite expectation makes a value case unscorable.

The server default for a silent numeric leaf is `rtol=5e-12`, `atol=0`. An operator sets it once
under the grader config as decimal strings, and may loosen it there (for example to the
`rtol=1e-5`, `atol=1e-8` of the public CritPt harness):

```yaml
comparison:
  default_rtol: "5e-12"
  default_atol: "0"
```

Both must parse as finite, nonnegative decimals or startup is refused. The server default never
overrides a statement promise, a leaf entry, or a task setting. A leaf entry replaces the whole
pair, and an explicit `0` stays `0`. The strict `rtol=5e-12`, `atol=0` also fills a missing
component of a partial authored entry and floors a harvested statement promise where no author
spoke, so a loosened server default cannot loosen a tight promise.

A stated absolute promise larger than the leaf's own magnitude is emptied, not applied. Such a
promise would accept any answer near zero, so it makes the leaf a free pass. The leaf then falls
to the authored entry, else the strict `5e-12`/`0` floor where no author spoke, which can be
tighter than the emptied promise.
A promise equal to the leaf magnitude is kept. An authored window is never emptied by this guard.

## Operator setup

### Mandatory config values

`configs/critpt_custom_grader.yaml` ships with four required operator inputs under `execution`,
each marked `???`:

| Value | What it must be |
|---|---|
| `snapshot` | A pinned, immutable Daytona snapshot identity. |
| `os_user` | A non-root account name that exists in the snapshot. |
| `owner_id` | A short label stamped on every owned sandbox for ownership attribution. |
| `journal_dir` | A durable, operator-owned directory that survives restarts. Not a temp dir. |

The server also needs an operator-composed `policy_model` config. This contribution supplies no
model, no credentials, and no snapshot identity.

This grader enforces its own request privacy at its HTTP boundary, independent of the model
server. The stock model server is a separate component, and the grader does not control what it
logs. A model server can record an upstream error body or a validation traceback, so a prompt or
a value can reach that log. Route sensitive content away from the model server at the operator
layer. The grader's own privacy guarantees, below, do not extend to it.

The stock client may resend a generation, and the collector may retry a `/verify`. The grader is
safe under a retry. It grades one job at a time and re-grades an idempotent request without any
double-scoring. Each dispatch is scored on its own returned answer, so a duplicate generation
costs at most one extra graded run and never affects scoring.

### Execution limits

The contributed config sets every execution bound to the values the acceptance replay ran with.
An operator with a different snapshot size or task population changes them together, because the
outer bounds derive from the suite timeout.

| Value | Config | What it bounds |
|---|---|---|
| `memory_limit_mib` | 1536 | Address space (`RLIMIT_AS`) of one runner process. 1536 MiB leaves headroom for the runner and the operating system in a 2 GiB snapshot. A suite that needs more raises `MemoryError` inside the case and is scored on that outcome. |
| `cpu_time_limit_s` | 1800 | CPU seconds of one runner process. It must stay at or below `suite_timeout_s + exec_timeout_margin_s`. |
| `suite_timeout_s` | 1800 | Wall clock for one full test suite (reference or candidate). |
| `compare_timeout_s` | 30 | Wall clock for one protected comparison run. |
| `create_timeout_s`, `transfer_timeout_s` | 120, 30 | Sandbox create, and each upload or download. |
| `exec_timeout_margin_s` | 20 | Extra time past the suite deadline so the runner can write a clean timeout result. It must stay at or above 15 s. |
| `job_timeout_s`, `queue_timeout_s`, `deadline_seconds` | 5900, 6020, 12310 | The whole job, the wait in the queue, and the server request deadline. Each one strictly exceeds the sum of the bounds inside it. |
| `max_concurrent_jobs`, `max_queued_jobs` | 1, 4 | See Known limitations. |

A suite that runs past `suite_timeout_s` is reported as `candidate_timeout` (scored 0) or
`reference_timeout` (unscorable). A reference that needs more memory or time than these bounds is
a task-side limit, not a grader fault. The operator raises the bound and the snapshot size together.

### The runtime snapshot

The grader uploads its own runner and comparator package into each sandbox at job time, so the
snapshot supplies the environment, not the grader code. The snapshot must contain:

- The interpreter at `execution.interpreter` (default `/usr/bin/python3`).
- The non-root `os_user` account, owning the workdir (`execution.workdir`, default `/tmp/ng-grader`).
- The libraries pinned in `runtime/requirements.txt`: `pydantic==2.13.4`, `sympy==1.14.0` with
  `mpmath==1.3.0`, `numpy==2.5.1`, and `scipy==1.18.0`. The runner and comparator need only
  pydantic and sympy. Task reference and candidate code import numpy and scipy, so the snapshot
  carries them at the versions the stored expectations were produced with.

### The journal

`journal_dir` holds one atomic record per job (`OwnershipJournal`), created mode `0700`. Each
record carries the sandbox stable names, labels, and lifecycle states needed to reconcile
ownership after a restart. It holds no task content, no expectations, and no source.

### Daytona credentials

The Daytona SDK reads credentials from its own environment variables. Set `DAYTONA_API_KEY`, or
`DAYTONA_JWT_TOKEN` together with `DAYTONA_ORGANIZATION_ID`. Set `DAYTONA_API_URL` (or
`DAYTONA_SERVER_URL`) and `DAYTONA_TARGET` when the target is not the SDK default. Never place a
credential value in the config, on a command line, or in a log. `execution.api_url` and
`execution.target` are optional non-credential connection settings.

### Provider seam

`execution.provider` selects the backend. Daytona is the only implemented backend
(`Literal["daytona"]`). Other Gym sandbox providers can be added as adapters behind the same seam.

The grader does not use the shared `AsyncSandbox` session wrapper. It uses its own small backend
interface so that it can journal a sandbox before any upload, keep the SDK client alive after a
delete to confirm the sandbox is gone, bound the result download from untrusted candidate code,
and look a sandbox up by its journaled stable name after a crash. Four points of the Daytona
adapter are provider-specific:

1. **Stable-name lookup for reconciliation.** Reconciliation looks a sandbox up by its exact id,
   or by its exact stable name when no id was learned, never by a prefix or list scan. After the
   settle deadline (the create timeout plus a margin of max(2 × create timeout, 60 s)), an absent
   lookup by stable name closes an ambiguous create as never made.
2. **Snapshot plus pre-created user.** Create passes the pinned `snapshot_id` and the `os_user`,
   whose account exists in the snapshot and owns the workdir.
3. **SDK error classification.** Provider exceptions map to a fixed category vocabulary with
   category-only diagnostics. Provider text never leaves the backend. Both the pinned SDK
   generation (Daytona 0.183.0) and the newer typed-error model are supported.
4. **Root exec with a privilege drop.** The Daytona toolbox runs every command as root. The
   runner refuses to run as root, so the run command drops privileges to the operator-configured
   `os_user` with `runuser -u <os_user> --`.

Long runs use the session API, not a blocking exec, because a blocking `process.exec` through the
Daytona proxy did not return once a command ran past about ten minutes. The grader starts each run
as an asynchronous session command (`run_async`) under a fresh unguessable session id and polls
the exit code every 5 s, each request under its own 30 s bound. A failed poll is retried while the
deadline allows, so one transport blip cannot discard a finished run. A command that reports no
exit before the deadline is provider uncertainty, never a scored failure. File uploads and
downloads use the SDK file API under `transfer_timeout_s`.

## Running it

Start the server with the grader config, a model server selected with `--model-type`, and the
four mandatory execution overrides:

```bash
gym env start \
    --config resources_servers/critpt_custom_grader/configs/critpt_custom_grader.yaml \
    --model-type openai_model \
    --model <model-name> \
    --model-url <model-base-url> \
    '+policy_api_key=${oc.env:CRITPT_GRADER_MODEL_API_KEY}' \
    '+critpt_custom_grader.resources_servers.critpt_custom_grader.execution.snapshot=<snapshot>' \
    '+critpt_custom_grader.resources_servers.critpt_custom_grader.execution.os_user=<os-user>' \
    '+critpt_custom_grader.resources_servers.critpt_custom_grader.execution.owner_id=<owner-id>' \
    '+critpt_custom_grader.resources_servers.critpt_custom_grader.execution.journal_dir=<journal-dir>'
```

`CRITPT_GRADER_MODEL_API_KEY` holds the model server API key. OmegaConf's `oc.env` resolver reads
it at startup, so the value never enters the config or a command line. Use `++policy_api_key=...`
when the config already holds a value at that key.

Validate the committed example data. Collate instantiates the config, so it needs the same four
execution overrides:

```bash
gym dataset collate \
    '+config_paths=[resources_servers/critpt_custom_grader/configs/critpt_custom_grader.yaml]' \
    --mode example_validation \
    --output-dir <out-dir> \
    '+critpt_custom_grader.resources_servers.critpt_custom_grader.execution.snapshot=<snapshot>' \
    '+critpt_custom_grader.resources_servers.critpt_custom_grader.execution.os_user=<os-user>' \
    '+critpt_custom_grader.resources_servers.critpt_custom_grader.execution.owner_id=<owner-id>' \
    '+critpt_custom_grader.resources_servers.critpt_custom_grader.execution.journal_dir=<journal-dir>'
```

Collect rollouts over the five example rows, mirroring `data/example.jsonl`,
`data/example_rollouts.jsonl`, and `data/example_metrics.json`:

```bash
gym eval run --no-serve \
    --agent critpt_custom_grader_simple_agent \
    --input resources_servers/critpt_custom_grader/data/example.jsonl \
    --output results/example_rollouts.jsonl \
    --concurrency 1 \
    '+limit=null'
```

Pass `--concurrency 1` unless the operator raised the job limit. The collector reads the
`--concurrency` flag, not the config's `num_samples_in_parallel`, and the grader cannot see it.
The shipped default queues a second concurrent verify instead of refusing it: `max_queued_jobs`
is `4` and `queue_timeout_s` covers about one full job. So a run without `--concurrency 1` queues
rather than dropping rows, but a waiter completes only while its own cumulative wait stays within
`queue_timeout_s`, not for as many jobs as the queue holds. A deep waiter behind several full jobs
can time out although the queue had room. Concurrency past `1 + max_queued_jobs` still answers `busy`.

## Verify response contract

`/verify` returns a `CritPtVerifyResponse`. Its fields:

- `reward` — `0.0` or `1.0`.
- `scored` — whether the reward is a real verdict about the candidate.
- `category` — the outcome name.
- `cleanup_status` — one of `not_started`, `resolved`, `unresolved`, `unknown`.
- `problem_id` — the task id, or null when the task could not be parsed.
- `responses_create_params`, `response` — the query and the saved candidate response.
- `_ng_failure_class`, `_ng_failure_subcategory`, `_ng_failure_terminal` — set only when the
  row is not a clean score. `failure_kind` carries the same value as `_ng_failure_class`. Two
  class names come from core's `FAILURE_KINDS` registry (`nemo_gym/failure_kinds.py`). The third
  uses the `critpt_custom_grader:` prefix that the registry asks for a server-specific kind.

### Categories

**Scored.** These carry a real reward.

- `passed` — every case matched. Reward `1.0`, cleanup `resolved`.
- `candidate_mismatch` — the comparator found a mismatch. Reward `0.0`, cleanup `resolved`.
- `candidate_invalid_output` — the comparator judged an observed value invalid. Reward `0.0`.
- `candidate_source_limit` — the candidate source failed the host size or encoding contract, or a
  completed response carried no usable source at all (an empty submission). Reward `0.0`.
- `candidate_source_error`, `candidate_timeout`, `candidate_exception`, `candidate_encoding_error`,
  `candidate_worker_abort` — after a completed preflight the candidate ran, but its process
  reported a source error, exceeded the suite timeout, raised, returned an unencodable object, or
  aborted. Reward `0.0`, cleanup `resolved`. A candidate cannot forge such a fault into a pass,
  only into its own failure, so it is scored 0 rather than left unscored. A containment failure,
  an external kill of the whole sandbox, an absent result file, or no reported process exit stay
  provider-ambiguous and unscorable.

**Terminal, not scored.** A defect in the task or the reference. `_ng_failure_terminal` true,
`_ng_failure_class` `verifier_error`.

- `task_invalid` — the task spec is invalid.
- `reference_invalid` — the reference contradicts its own stored expectation.

**Not scored, not terminal.** Infrastructure uncertainty, `_ng_failure_class`
`provider_unavailable`. Examples: `reference_error`, `reference_timeout`, `comparator_uncertain`,
`provider_create`, `provider_exec`, `transfer_limit`, `result_invalid`, `queue_timeout`. The
saved response is good, so reverification re-checks it once the server is healthy (see "Recovery").

**Not scored, not terminal, needs a new candidate.** `_ng_failure_class`
`critpt_custom_grader:response_incomplete`.

- `response_incomplete` — the saved response did not complete, or carries an incomplete or error
  envelope, so it has no usable source. The grader rejects it the same way on every pass, so
  reverification never resolves it. Regenerate the row with `gym eval run --resume`. A response
  that DID complete with no error but carries no usable source is instead an empty submission,
  scored `0.0` as `candidate_source_limit`.

Three categories the operator reads as signals about the server, not the task:

- `busy` — a job was already running and the one-job limit was reached. `max_concurrent_jobs` is
  pinned to exactly `1`, so serialize verify calls, or raise `max_queued_jobs` to widen the wait
  queue.
- `ownership_unresolved` — the server could not confirm it deleted its sandboxes. The next job
  reconciles the leftover records first, so a transient provider delay clears on its own. A restart
  also reconciles the journal at startup (see "Recovery").
- `execution_unknown` — an unexpected or uncorrelated execution outcome. A conservative catch-all,
  never a verdict about the task.

## Privacy

The server runs with `request_privacy.private_requests: true`.

**Guaranteed.** Request bodies, validation details, provider exceptions, and task sources are not
logged. Errors return a fixed category-only body. FastAPI auto-instrumentation is not applied, so
no server span records the request URL or headers. The saved response is an opaque, bounded
recovery artifact, not a diagnostic.

**Not guaranteed.** Provider-side storage is outside the server's control. A sandbox provider may
retain data on its own systems. Privacy at the server does not extend to the Daytona backend.

## Integrity model

The sandbox reporting channel is unauthenticated by design. Grading integrity does not rest on
trusting reported values. It rests on the candidate never receiving the expected values or
tolerances, on expectations being task-stored and never derived from a run, and on the comparator
running no task code. A forged candidate value must still equal the hidden stored expectation to
pass, which the candidate cannot know and was always free to return legitimately.

The same-UID Python supervisor is containment machinery, not an authentication boundary. Every
trust claim also requires an operator-enforced runtime that protects the supervisor and its
dependencies from same-UID interference, privilege escalation, and escape. This contribution does
not qualify that runtime.

## Recovery and the at-least-once contract

The system is at-least-once, not exactly-once. Each attempt is scored independently, and the
latest attempt stands, so no double count enters a win rate.

**Restart after a crash.** A crash can leave an unresolved record in the journal directory. On the
next start the server builds the grader and runs reconciliation before it serves any request. A
record whose sandbox is already gone clears itself. A record whose sandbox still exists is deleted
by its exact stable name. A record that cannot resolve yet, such as a create still inside its
settle deadline, is retried before the first job is admitted. The operator does not edit the
journal by hand. Point the restarted server at the same `journal_dir` and let reconciliation run.

**Re-grade a saved candidate.** A `provider_unavailable` row holds a good saved candidate that
the grader could not score. `gym eval reverify` re-sends the saved response with no new model call.
Core has no option that selects failure rows by class. `--judge-failed-only` selects only
`judge_failed` rows, so it does not recover these. Select the rows yourself: take the latest
attempt per rollout, drop rollouts that already succeeded, and keep `provider_unavailable`.

```bash
jq -cn --slurpfile ok <(jq -c '[._ng_task_index, ._ng_rollout_index]' <rollouts>.jsonl) '
  ($ok | map({key: tostring, value: true}) | from_entries) as $done
  | reduce inputs as $r ({}; .[([$r._ng_task_index, $r._ng_rollout_index] | tostring)] = $r)
  | to_entries[] | select($done[.key] | not) | .value
  | select(._ng_failure_class == "provider_unavailable")' \
  <rollouts>_failures.jsonl > <retry>.jsonl

gym eval reverify \
    --config resources_servers/critpt_custom_grader/configs/critpt_custom_grader.yaml \
    --model-type openai_model \
    '+policy_model_name=<model-name>' \
    '+policy_base_url=<model-base-url>' \
    '+policy_api_key=${oc.env:CRITPT_GRADER_MODEL_API_KEY}' \
    '+model_endpoint_readiness_timeout_seconds=0' \
    '+critpt_custom_grader.resources_servers.critpt_custom_grader.execution.snapshot=<snapshot>' \
    '+critpt_custom_grader.resources_servers.critpt_custom_grader.execution.os_user=<os-user>' \
    '+critpt_custom_grader.resources_servers.critpt_custom_grader.execution.owner_id=<owner-id>' \
    '+critpt_custom_grader.resources_servers.critpt_custom_grader.execution.journal_dir=<journal-dir>' \
    --inputs <rollouts>_materialized_inputs.jsonl \
    --rollouts <retry>.jsonl \
    --output <recovered>.jsonl \
    --concurrency 1
```

`gym eval reverify` starts its own servers, so stop `gym env start` first. Only one grader process
may use the `journal_dir`. Reverify has no `--model` or `--model-url` flag. Pass the model with the
`policy_*` overrides instead. Reverify never calls the model, so
`model_endpoint_readiness_timeout_seconds=0` skips the wait for a model endpoint. `gym eval run`
writes `<rollouts>_materialized_inputs.jsonl` next to its output.

Rows that fail again go to `<recovered>_failures.jsonl`. To score the union, start the servers
again with `gym env start` and the same config. Aggregate needs the running head server. Then run:

```bash
gym eval aggregate -i "'<rollouts>.jsonl,<recovered>.jsonl'" -o <merged>.jsonl
```

Keep the inner single quotes. Without them, Hydra rejects the comma list as an ambiguous value. Do
not include `<retry>.jsonl` in that list.

**Regenerate a candidate.** Use `gym eval run --resume` only when no usable saved candidate
exists: a `critpt_custom_grader:response_incomplete` row, or an `agent_run_error` or
`agent_request_failed` row (a lost outer `/run` response, or a `/verify` call that returned an
error body such as a 503). Resume re-dispatches every non-terminal failure row of every class and
calls the model again, so it also regenerates `provider_unavailable` rows. Before you resume,
append `<recovered>.jsonl` to `<rollouts>.jsonl` so that resume skips the rows that reverify
already scored. `verifier_error` rows are terminal and are never retried.

## Tests

Host-safe suites run on any host and cover the schema, codec, comparator, execution controller,
and runner protocol:

```bash
gym env test --resources-server critpt_custom_grader
```

The remote suite `tests/test_runner_remote.py` is different. It executes deliberately hostile
source that burns CPU, forks, signals its parent, and forges result files. It is skipped unless
`NG_CRITPT_RUNNER_REMOTE=1` is set. **Run it only inside a disposable non-root sandbox with
`/proc` mounted. Never run it on a shared host.**

## Known limitations

- **Exception expectations are scored.** A case whose expectation is an exception (the
  `{input, expected_error}` alias, or an `exception` expected kind) passes when the code raises,
  and neither the exception type nor its message is compared. The stored `message` is provenance
  only. A reference that does not raise where an exception is expected is a reference defect
  (`reference_mismatch`). A candidate that does not raise is a wrong answer (`candidate_mismatch`).
- **Daytona only.** No other execution provider is implemented. The backend seam is designed for
  further adapters, and the grading logic does not depend on the provider.
- **One job at a time, with a small wait queue.** `max_concurrent_jobs` is pinned to exactly `1`
  (`ge=1, le=1`), so cleanup and reconciliation never race execution. Do not raise it. `max_queued_jobs`
  is `4`. A second concurrent verify waits in the queue and completes when the running job finishes,
  while its cumulative wait stays within `queue_timeout_s`. Only concurrency past `1 + max_queued_jobs`
  gets `busy`. `--concurrency 1`
  stays recommended, but a run without it queues instead of dropping rows. To admit more waiters,
  raise `max_queued_jobs`, not `max_concurrent_jobs`.

## Licensing

- Code: Apache 2.0.
- Runtime scientific dependencies: SymPy (BSD) and its `mpmath` dependency (BSD), Pydantic (MIT).
  Retain upstream copyright and license notices in any distributed runtime.
