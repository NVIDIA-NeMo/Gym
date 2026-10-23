# Scale-SWE resources server

Verification for [AweAI-Team/Scale-SWE](https://huggingface.co/datasets/AweAI-Team/Scale-SWE):
20,181 software-engineering tasks, entirely Python.

Unlike the SWE-rebench server, there is exactly one test runner here (pytest), so there is no
per-row log-parser dispatch. Each row ships its own prebuilt image (`aweaiteam/scaleswe:...`),
an explicit `workdir`, and a `pre_commands` string that performs the checkout of
`parent_commit` and scrubs the repo's git history so the fix commit is unreachable.

Sandboxes come from `nemo_gym.sandbox`, so the same server runs on OpenSandbox or any other
configured provider.

## The failing tests

A row supplies the tests that must go from failing to passing one of two ways, and 61% of rows
carry both:

- `f2p_script` (92% of rows): a literal pytest file, uploaded to `/tmp` and copied into the
  checkout as `test_fail_to_pass.py` (the name the dataset's own `FAIL_TO_PASS` ids expect). It
  cannot be written into the repo before `pre_commands` runs, because `pre_commands` includes
  `git clean -fd`, which would delete it.
- `f2p_patch` (69% of rows): a diff that adds the tests directly into the existing test tree.

pytest is pointed at the **unique test files** named by `FAIL_TO_PASS ∪ PASS_TO_PASS`, not at
every individual node id — a row can carry hundreds of ids across up to ~69 files, and grading
by id afterwards from the collected output gives the same verdict without risking a pathological
command line.

## Grading

An instance resolves only when **every** `FAIL_TO_PASS` and `PASS_TO_PASS` node id is observed
as `PASSED` in the pytest output. A node id absent from the output counts as not passing —
treating absent as success is the standard way a broken test command (or a collection error)
scores as a resolved instance.

`FAIL_TO_PASS` / `PASS_TO_PASS` arrive as JSON-encoded strings in this dataset, not lists; see
`as_id_list`.

The graded region of the log is delimited by markers, so setup-time noise (pip installs,
`pre_commands` output) never reaches the pytest-output parser.

## Prepare the data

`prepare_scale_swe.py` only writes rows already on `data/supported_instance_ids.txt` -- the
17,696 instances (of 20,181) whose golden patch resolved in all three passes of a full-set 3x
sweep (see Golden-patch validation below). The other 2,485 are excluded because none can be
scored either way: 1,671 never resolve, 709 never produced a verdict (image pull / setup
failures, not row failures), and 105 resolve inconsistently across passes. It streams the Hub
dataset rather than loading the full split, and shuffles the kept rows (fixed seed, reproducible)
before writing, since the Hub's own row order is grouped by repo.

```bash
SCALE_SWE_LIMIT=200 python resources_servers/scale_swe/prepare_scale_swe.py
```

## Golden-patch validation

Grades each task with the dataset's own patch, which measures the dataset rather than a model.
A row whose golden patch does not resolve cannot be used for evaluation or training.

```bash
gym env start \
  --config resources_servers/scale_swe/configs/scale_swe.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

python resources_servers/scale_swe/apply_golden_patch.py \
  +training_jsonl=resources_servers/scale_swe/data/scale_swe_training.jsonl \
  +limit=40 +concurrency=8
```

It prints a resolved rate overall and per language (this set is 100% Python, so that breakdown
is a single row), and writes one row per task. `diagnose_failures.py` separates environment
failures (OOM, disk, network, timeout, no-verdict, zero-tests-collected) from genuine row
failures — see `resources_servers/swe_rebench/diagnose_failures.py` for the shape of evidence
this looks for and why each cause is only reported on direct signal in the captured output, not
inferred from the fact that a task failed.

`aggregate_golden_patch.py` joins repeated golden-patch passes (e.g. a 3x sweep) into
supported / flaky / broken / inconclusive buckets, keeping an infra fault (no verdict) separate
from a genuinely nondeterministic test so neither silently corrupts the other's label.

## Running an agent

`configs/scale_swe_opencode.yaml` wires the opencode sandboxed agent to
`scale_swe_resources_server`, pointed at `data/scale_swe_training.jsonl`. It lives in a separate
file rather than folded into `scale_swe.yaml` so golden-patch validation (which never touches an
agent) doesn't need to pull in the agent's much larger sandbox/permission config.

```bash
gym env start \
  --config resources_servers/scale_swe/configs/scale_swe_opencode.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml
```

## Environment Server sessions

An Environment Server such as `single_agent_turn` seeds with a typed `ResourcesSeedSessionRequest` whose
`task_data` is the row. The reply lends the prepared task sandbox as `sandbox_access`. The session cookie
names the session for `/verify`, and `/close_session` ends it.

- **Seed.** A seed in golden-patch mode is refused with a 422. If `git rev-parse HEAD` does not name a commit
  in the prepared sandbox, the seed fails with a retryable 503 and stops the sandbox: without a base commit,
  the agent's patch could never be captured.
- **Verify once, replay after.** Verification captures the agent's patch, stops the task sandbox and grades
  the patch in a fresh sandbox. The session keeps its first completed verdict. An identical retry gets that
  verdict back, and a different request gets a 409.
- **No verdict is masked, not scored.** A verification without a verdict returns `mask_sample: true`, a
  `failure_kind` (`verifier_error`, `provider_unavailable` or `session_lost`) and a `failure_reason`. Masked
  results are not kept, so a retry gets a 503 that asks the caller to retry the episode. These count as no
  verdict:
  - the capture could not reach the sandbox, or a capture command timed out or ended without an exit status
    (nothing is graded, because an empty patch would read as a completed 0);
  - the verifier exec ended without an exit status for a reason other than a timeout;
  - the eval script exited before it reached the tests.

  A verifier timeout stays a verdict, because a test the patch made hang must still score 0.
- **A repository the agent broke scores 0.** If git itself exits non-zero on the repository the agent left,
  the result is a completed 0 with `mask_sample: false`, and `failure_reason` names the git error. Examples
  are a stale `.git/index.lock` from an interrupted git command, a deleted `.git`, or a pruned base commit.
- **Close.** A repeated close, or a close for a session that was never seeded, succeeds. A seed after a close
  is refused for `session_close_retry_window_seconds`. A close that arrives during setup fences the seed, which
  then stops its own sandbox. Close stops the task sandbox and any verifier sandbox whose stop has not
  succeeded yet.
- **Stops.** The server waits up to `sandbox_stop_timeout_seconds` (default 120) for each stop, but never cuts
  a stop short. A stop that is still running, or that failed, leaves the sandbox with the session. Close then
  raises so the caller can retry, and a retried close or shutdown waits for that stop again or starts it again.
- **One worker.** Sessions live in process memory, so the server refuses `num_workers` other than 1.
- **Shutdown.** When the server shuts down, it stops every sandbox it still holds, for Environment Server
  sessions and cookie sessions alike, with the same bounded wait per stop. A session with a request still in
  progress gets that long to finish, and then its sandboxes are stopped anyway. A stop that fails is logged,
  and the provider's TTL reclaims the sandbox.

The cookie path that an Agent's `/run` uses does not replay or mask, and the stops inside its requests are not
bounded. Only the shutdown cleanup above applies to it.
