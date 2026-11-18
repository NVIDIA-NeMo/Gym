# VisualWebArena

VisualWebArena extends WebArena with visually grounded tasks and reference
images. This integration runs the maintained 908-task population through the
same Gym visual-browser runtime, browser-session provider, `computer_use`
action contract, and distributed agent loop used by WebVoyager and WebArena.
Only the local-site setup, task images, and VisualWebArena evaluator are
benchmark-specific.

## Prepare

From the repository root:

```bash
gym eval prepare --benchmark visualwebarena
python benchmarks/visualwebarena/prepare.py
gym env validate visualwebarena
```

The prepare command downloads the public
[`jayl940712/webarena_benchmarks`](https://github.com/jayl940712/webarena_benchmarks)
archive at commit `6a2977939b157b0ab9de7799bb089c721f1ac115`. It verifies the
source JSONL SHA-256, exactly 908 rows, and all 341 distinct local task images
before producing the model-neutral
`benchmarks/visualwebarena/data/visualwebarena.jsonl`. Source identity is
recorded in `source_lock.json`.

The first command prepares the dataset only. The second reuses the cached
source and writes a private, gitignored `benchmarks/visualwebarena/env.yaml`.
It preserves an existing configuration without overwriting it. Use `--force-env` only when replacing an
existing generated file intentionally:

```bash
python benchmarks/visualwebarena/prepare.py --force-env
```

The named validation command reads `manifest.yaml`, checks the prepared rows
and verifies the component composition. This benchmark uses agent-generated
prompts and remains `experimental: true`; the manifest does not certify scores.

To reuse an offline or shared checkout, set the root containing both
`visualwebarena.jsonl` and the `visualwebarena/` image tree:

```bash
export VISUALWEBARENA_SOURCE_ROOT=/shared/webarena_benchmarks
gym eval prepare --benchmark visualwebarena
python benchmarks/visualwebarena/prepare.py
```

Mount that root read-only at the same absolute path on every distributed
worker. The JSONL retains relative image paths below this root.
When using `--source` with a separately stored JSONL, also pass `--source-root`
if the images are not below the JSONL's parent directory. The generated
`env.yaml` uses that same image root for both the agent and resource server.

## Run

Deploy and reset the VisualWebArena site stack, export the deployment-specific
`WA_*` URLs and approved `WEBARENA_JUDGE_*` credentials, then run:

```bash
export VISUALWEBARENA_STATE_ROOT=/shared/visualwebarena-owned-deployment-state
gym env prefetch --config benchmarks/visualwebarena/env.yaml
gym env start --config benchmarks/visualwebarena/env.yaml
```

In another terminal:

```bash
python -m benchmarks.visualwebarena.run \
  --config benchmarks/visualwebarena/env.yaml \
  --input benchmarks/visualwebarena/data/visualwebarena.jsonl \
  --output-dir results/visualwebarena/clean-state-run \
  --state-root "$VISUALWEBARENA_STATE_ROOT" \
  --reset-url "$VWA_RESET_URL" \
  --status-url "$VWA_RESET_STATUS_URL" \
  --reset-token-file "$VWA_RESET_TOKEN_FILE" \
  --ca-cert "$VWA_RESET_CA_CERT" \
  --resource-health-url "$VWA_RESOURCE_BASE_URL/healthz"
```

Set `VISUALWEBARENA_STATE_ROOT` before starting the servers. It must identify
one shared directory for the owned site deployment, mounted at the same path
on the coordinator and every resource worker. Supply the deployment's HTTPS
reset/status endpoints, bearer-token file and CA certificate; the token is
read at runtime and is not written into the protocol manifest. Repeat
`--resource-health-url` for every resource worker using this site deployment.

The coordinator verifies all 908 task identities against the maintained
population before any reset. It runs 492 read-only tasks, then 408 ordinary
mutating tasks, followed by eight protected tasks in six reset epochs:
`346,678`; `345,751`; `336`; `335`; `340`; `92`. These are runtime row indices;
their original source IDs include `visualwebarena-680` and
`visualwebarena-753` for rows 678 and 751. The coordinator resets the owned
deployment under the exclusive lease before the first ordinary wave as well
as before each protected epoch. A subset can be
selected with `--task-ids`, while `--input` must still contain the full
identity-checked population. The reference epoch numbers remain stable for
subsets. The default concurrency is one; `--concurrency` requires sufficient
isolated browser capacity.

After the initial verified reset, ordinary phases hold a shared filesystem
lease and retain their requested Gym collection concurrency. After downgrading
the exclusive reset lease, the coordinator checks its ownership record again;
if an intervening reset changed ownership, collection aborts. Before protected
epochs, the coordinator releases its
shared lease and waits for exclusive ownership. Every job sharing the site
deployment must use the same state root and cooperate with that lease; it does
not discover unrelated jobs that bypass it. Before each reset and after each
protected phase, every declared worker must report zero active, creating and
pending-cleanup sessions and browser leases. An ambiguous reset response
leads to status polling rather than another reset trigger. Every reset must
show `Reset ongoing` followed by `Ready for duty!` before tasks are dispatched,
even when the trigger returned HTTP 200/202. An immediate ready response can
belong to the prior deployment; polling continues until the actual transition
is observed or the deadline expires. Redirected control requests, service
errors and readiness timeouts abort the run.

Each coordinator invocation acquires exclusive ownership and resets before
starting, so a second run cannot inherit mutations from a completed or failed
run. Separate invocations sharing one deployment wait for existing ordinary
leases before their initial reset. Use one coordinator's collection concurrency
for parallel tasks, or separate site deployments for parallel benchmark runs.
The lease does not provide separate site snapshots to independent runs.

The coordinator attaches the audited read/write classification to task data,
so the existing resource site pool admits simultaneous readers while writers
wait for that site's readers/writers. These annotations do not alter prompts
or evaluator targets. The shared lease excludes resets, not conflicting ordinary writes. Retain the
existing site-pool writer locks and deployment isolation: `local_locks` covers
one resource-server process, so concurrent mutating shards in different
resource processes need isolated site replicas or a deployment-wide writer
scheduler. Each browser must have its own display. To shard a population
across jobs, `--stage read-only`, `--stage mutating` and `--stage protected`
permit explicit job dependencies: all read-only shards finish before writers,
and ordinary waves finish before protected epochs. Each stage reports only
its actual selected denominator. Alternatively, use one coordinator to
dispatch all phases to sufficient isolated worker capacity. Overlapping site
reset epochs with ordinary work requires independent site deployments.

The resource server independently verifies the protected task's intent,
active run/epoch lease and actual initial address/cart or classifieds state
before model actions. Later resets must reproduce the first observed baseline
for that site. Receipts and baseline observations live below the private state
root; rollout verifier metadata carries the receipt hash. Direct collection of
these eight tasks without their coordinator context fails as infrastructure
instead of silently grading a contaminated site.

Use a fresh output directory. Native Gym phase inputs, collector logs,
rollouts and recovery artifacts remain under it. The final merged output and
summary are written only after every requested task has exactly one valid,
unmasked result and protected tasks carry their initial-state receipts. Both
raw and full-success totals/rates are recorded with the actual denominator.
If reset readiness or worker cleanup cannot be proved after a failure, the
state root gets a `requires-recovery-<run-id>.json` record. Reconcile all such
records, the recorded workers and site ownership before starting a fresh run;
an ordinary retry does not clear them or trigger another reset.

Stop the foreground `gym env start` process with Ctrl-C after evaluation.
Each headed Chromium process owns one display and therefore permits one active
session. Scale out by launching isolated resource-server and agent processes
with the site isolation and phase barriers described above.
Run non-state-changing tasks before state-changing tasks, including tasks
that read across multiple sites. Reset the owned site deployment before a new
evaluation; serialization alone does not undo changes from an earlier task or
run. Phase classification and independent site replicas belong to deployment
orchestration, not to task-prompt rewriting.

## Runtime and evaluator boundary

The shared runtime supplies screenshots, headed Chromium, PyAutoGUI-backed
actions, lifecycle handling, artifacts, and browser-session acquisition. The
`webarena_browser` resource adds site login and URL substitution. This stacked
PR adds VisualWebArena's input-image materialization and pinned evaluator,
including fuzzy-image dependencies and before/after collision snapshots.

Prepared rows do not contain model prompts or tool schemas. The default config
selects the Nano Omni policy adapter; another compatible policy can replace
that adapter while retaining the same task population and environment. Model
and runtime results are comparable only when the source lock, site snapshot,
viewport, action profile, evaluator, and judge configuration also match.

The Nano Omni composition uses 100 steps, three recent browser screenshots,
and no additional request-level output-token cap. Context exhaustion and
terminal action errors end interaction but still invoke the evaluator on the
live state before closing the browser. Model tool arguments are not repaired.
Both task-image readers allow 32 MiB, including the maintained task image
that exceeds the previous 25 MiB default; input images are not resized to
work around that limit.

This composition uses the shared agent's existing 100-step limit. If the model
uses the entire budget without a terminal answer, the agent evaluates the
observed state with no fabricated final answer before closing the session.
The optional Nano budget prompt/template feature and automatic display
allocation are reviewed in separate PRs; this integration uses the agent and
browser runtime already available on `main`. Run records must retain the
actual stop behavior and template options alongside the clean-state schedule.
Historical scores retain their original configurations.
