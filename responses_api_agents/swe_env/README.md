# swe_env

Shared library for provisioning and grading SWE (software-engineering) task
environments. It is imported by `anyswe_agent` (and usable by any other Gym
agent that self-drives inside a SWE task sandbox); it is **not** a runnable
server and has no config or entrypoint.

Everything is provider-neutral, running over the `nemo_gym.sandbox` providers
(docker / apptainer / opensandbox):

- **`harnesses/`** — per-dataset-family recipes (SWE-bench, SWE-bench
  Multilingual, SWE-bench-ext, R2E-Gym, SWE-rebench, NV-internal). Each builds
  the task sandbox spec, materializes the model patch, runs the evaluation, and
  grades the result host-side via the official per-repo parser (falling back to
  a generic parser only where the official one is unavailable).
- **`sandbox.py`** — async sandbox lifecycle (`AsyncSweEnvironment`,
  `acquire_sandbox`) with always-teardown semantics.
- **`self_drive.py`** — provision a writable sandbox, inject a sandbox-reachable
  model endpoint / egress env, run an opaque agent launch command, and extract
  the resulting `git diff` patch.
- **`verify_task.py`** — grade a patch inline in a fresh sandbox (no separate
  `/verify` server), returning a mask-aware reward.
- **`parsing/`** — test-log parsers (relocated verbatim from `swe_agents`).

## Reward-profiling baseline — SWE-bench Verified gold patches

A gold-patch census validates the grader end-to-end: feed each instance's ground-truth patch
through the flat grader; a correct grader resolves everything except the genuinely broken
instances. On SWE-bench Verified (500 gold patches, `patch_exists` 500/500, **0** infra errors;
an empty patch resolves **0 / 500**):

| grader | resolved | |
|---|---|---|
| **docker-flat** (this library) | **493 / 500** | host-side flat grading, pull-on-demand images |
| **apptainer-flat** (this library) | **491 / 500** | same host-side grader as docker — misses the *identical* 7; the only gap to 493 is the httpbin outage below |
| official `swebench run_evaluation` (nested, docker) | 490 / 500 | the canonical tool, for reference |

> The earlier apptainer figure of **492** was swebench's *nested* `run_evaluation` on a `.sif`
> (a different grader) — not this flat grader. The flat grader resolves **493** on **both** docker
> and apptainer; the numbers above differ only by transient run-time conditions, never by grading.

**docker-flat ≡ apptainer-flat.** Both run the *identical* host-side grader (same eval script +
per-repo parser) and miss the **same 7** instances; only the sandbox the eval script executes in
differs, and apptainer-flat never resolves anything docker doesn't (the over-resolve set is empty).
The apptainer-vs-docker count gap is entirely environmental/operational, not grading:
- **psf/requests-1724, -2317** (the 491→493 gap) — their tests hit the external `httpbin.org`, which
  was returning HTTP 503 during the apptainer run (`curl http://httpbin.org/get` → 503). docker fails
  them identically when httpbin is down (the official audit reproduced the 503s); docker-flat
  resolved them earlier when httpbin was up. External-service flakiness, not a sandbox difference —
  so flat-apptainer ties flat-docker at **493** whenever httpbin is reachable.
- **scikit-learn-14710** — hit the eval timeout under `--concurrency 12`; re-graded in isolation it
  resolves (folded into the 491 above). A concurrency artifact, not a grading difference.

Two apptainer sandbox fixes were required for this parity (see `anyswe_agent/app.py` + `harnesses/`):
- **`--no-mount home`** — apptainer bind-mounts the host `$HOME` by default, leaking the host
  matplotlib font/config cache into the eval and flipping image-comparison tests (e.g. `matplotlib`
  `test_pcolormesh_small[eps]`); docker has no host-home bind.
- **provider-independent eval-command timeout (1800s)** — the eval command otherwise inherited each
  provider's exec default (apptainer **180s** vs docker **3600s**), silently masking long suites
  (scikit-learn / sympy, or any suite slowed under concurrency) as timeouts on apptainer only.

**Is 493 an overcount?** No. docker-flat (493) exceeds the official `run_evaluation` (490) because
**official 4.1.0 runs sphinx-via-tox pytest without `-rA`**, so it cannot see the genuinely-passing
F2P test (`test_empty_all` / `test_needs_extensions`, `tox exit 0`) and *undercounts* sphinx-8595/9711
— which flat correctly resolves. flat is in fact slightly *conservative*: it misses sphinx-8120/8265/8269
(official resolves them; an instance-specific parser quirk, missed by docker **and** apptainer alike).
The true resolvable count is **~496 / 500**; the only genuinely unresolvable gold patches are the 4
upstream env-flaky instances that fail on **every** grader — **astropy-7606/8707/8872** (a distutils /
nose deprecation raised-as-error during collection) and **django-10097** (a real test failure,
confirmed once the 1800s timeout let its ~1900-test suite run to completion).

Reaching the flat baseline also required two flat↔nested **reconstruction** fixes the census surfaced
(445 → 486 → 493):
- **`PYTEST_ADDOPTS=-rA`** — swebench 4.1.0's eval command for some families (sphinx via tox,
  several sklearn) runs pytest without `-rA`, so passing tests print only as dots and the host-side
  parser (`parse_log_pytest_v2`) saw zero passes (445 → 486).
- **drop `GIT_CONFIG_GLOBAL=/dev/null`** — older instance images' git can't parse `/dev/null`, so
  the eval script's `git checkout` + test-patch `git apply` failed and required tests came back
  "absent" (486 → 493). See `harnesses/swebench.py`.

### Reproduce

Run the gold-patch census with `responses_api_agents/anyswe_agent/gold_census.py` (no model, no
agent — it feeds each instance's gold patch through the flat grader and tallies resolves):

```bash
# docker: images pull on demand; --rmi removes each after grading to cap disk
HF_HOME=/tmp/hf_cache python responses_api_agents/anyswe_agent/gold_census.py \
    --provider docker --concurrency 8 --tests-timeout 3600 --rmi
# (quick smoke on a subset)
python responses_api_agents/anyswe_agent/gold_census.py --provider docker --limit 25 --rmi

# apptainer: pre-built local .sif images ...
python responses_api_agents/anyswe_agent/gold_census.py --provider apptainer \
    --container-formatter 'data/sifs/{instance_id}.sif' --concurrency 8 --tests-timeout 3600
# ... or build each missing .sif on-demand from docker:// and delete after grading
HF_HOME=/tmp/hf_cache python responses_api_agents/anyswe_agent/gold_census.py \
    --provider apptainer --apptainer-build --rmi --concurrency 8 --tests-timeout 3600 \
    --container-formatter 'data/sifs/{instance_id}.sif'
```

The script applies the two apptainer parity fixes automatically: `--no-mount home` (host-`$HOME`
isolation) and the eval-command timeout (1800s, or `--tests-timeout`). It checkpoints to
`gold_census_results.json` (resumable) and prints `gold resolved N/500` plus the not-resolved list.
A clean run has **0** `error_kind` (infra) failures; a resolved instance means the gold patch passed
that instance's FAIL_TO_PASS + PASS_TO_PASS tests under the host-side grader.

**For EXACT docker↔apptainer parity** (identical passing set), run both **in the same time window**
(so the external `httpbin.org` state is the same for the `requests` instances) and with
`--tests-timeout 3600` (so no heavy suite is masked by a concurrency-induced timeout). Under those
conditions the two providers resolve the *identical* set — the grader is host-side and
provider-neutral, so the sandbox only changes where the eval script runs, never the verdict.

To cross-check against the canonical *nested* grader (note: `run_evaluation` 4.1.0 undercounts
sphinx-via-tox instances — it runs pytest without `-rA` — and depends on external `httpbin.org` for
the `requests` instances, so it is a lower bound, not ground truth):

```bash
HF_HOME=/tmp/hf_cache python -m swebench.harness.run_evaluation \
    -d princeton-nlp/SWE-bench_Verified -p gold --run_id gold_audit \
    --max_workers 8 --cache_level none --clean True
```

## Tests

The unit tests run against a scripted fake `SandboxProvider`, so they need no
Docker/apptainer and execute in CI:

```bash
ng_test +entrypoint=responses_api_agents/swe_env
```
