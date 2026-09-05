# GDPVal checkpoint evaluation on HSG

This package takes one Hugging Face checkpoint path and runs the maintained
Super-35/Nemotron GDPVal evaluation from policy rollouts through Stage 0 and
Stage 1 AA-v2 judging.

It is designed to be fast without weakening the final result: completed work is
reused, recovery runs only missing tasks, and expensive provider calls resume in
place. Cheap reproducibility and correctness checks remain fail-closed.

Version 1.4.13 moves the existing-deliverables judging path's executable
package, selected Gym tree and Python environment, patched runtime, Apptainer
installation, GDPVal image, component environments, and software caches to
stable node-local `/raid/scratch` paths. Slurm-spooled scripts perform the
initial copy before any package code is sourced. Lustre retains canonical
inputs, resumable task evidence, immutable receipts, and final results. It also
removes recursive cache-tree scans from controller polling and supports
cross-filesystem Office-to-PDF publication.

The r7 package refresh adds two provider-boundary fixes discovered in a strict
Stage 1 tail. Gemini now enforces Vertex's 50 MB *per-PDF* limit before the
existing aggregate request limit and rasterizes only an oversized PDF. Claude
requests send both LiteLLM timeout fields at 900 seconds so the gateway does
not apply its shorter default ahead of Gym's matching 900-second bound. These
are transport/runtime fixes; completed comparison rows remain reusable.

Version 1.4.12 backports the complete missing-reference and strict-comparison
contracts from PRs #2796 and #2807 into the pinned judging runtime. It also
supports a dedicated Gemini credential and explicitly pins Claude to high
effort. Version 1.4.11 made patched component selection fail closed, shipped the OpenAI
adapter's install marker in the immutable runtime overlay, and losslessly
splits PDF overflow by page when a whole document exceeds a provider's image
budget. Version 1.4.10 added bounded provider-owned 429 backoff and a fingerprinted,
per-campaign Gemini concurrency limit. Version 1.4.9 isolated the generated
occupation distribution inside each judge namespace, binds its absolute path and SHA-256 to the provider-free fingerprint
receipt, freezes it owner-read-only, and revalidates it before provider access.
Fingerprint probing can no longer write into the immutable runtime overlay.
It also permits exact audio/video pairs up to a 495 MB serialized projection
only after media routing narrows them to Gemini, retaining a 5 MB margin below
the observed provider hard limit; non-media routing keeps its existing gates.
Transport admission increases above the legacy ceilings are canonicalized only
for the campaign fingerprint, so a completed Stage 0 can resume under the wider
provider ceiling without changing its evidence or rerunning calibration.
When a preflight-valid request is rejected specifically for provider context
length, that provider is excluded for the whole matchup and all four votes are
replayed from the original seeded RNG state with the remaining eligible panel;
the exclusion is recorded and all other provider errors remain fatal.

Existing-deliverables imports also get six bounded per-file
attempts for known transient Lustre and transport-endpoint read errors. Every
retry restarts the file from byte zero; the original size/hash contract and the
whole-tree post-copy identity check remain mandatory.

## Quick start

Run on HSG:

```bash
E2E=/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/spanev/gdpval_colo/aav2/checkpoint_e2e_true3_v1_4_13/run_checkpoint_e2e.sh
CHECKPOINT=/absolute/path/to/a/new/checkpoint/hf

# Creates and validates deterministic campaign files. Submits nothing.
"$E2E" prepare "$CHECKPOINT"

# Starts rollouts and the autonomous rollout-to-judging controller.
CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS=true \
  "$E2E" submit "$CHECKPOINT"

# Read-only progress check.
"$E2E" status "$CHECKPOINT"

# Print the final validated ELO receipt when complete.
"$E2E" result "$CHECKPOINT"
```

`checkpoint_e2e` may be a convenience symlink to the immutable
`checkpoint_e2e_true3_v1_4_13` installation. The launcher resolves that link to its
physical directory before recording settings and source pins, so a later alias
upgrade cannot redirect an already prepared campaign.

The one-command form is:

```bash
CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS=true \
  "$E2E" all "$CHECKPOINT"
```

`submit` and `all` return after Slurm submission. The controller continues on
HSG after the laptop disconnects.

If a controller ends before completion, continue the same campaign:

```bash
CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS=true \
  "$E2E" resume "$CHECKPOINT"
```

Do not create a new timestamped run directory. The deterministic run root is
what makes rollout and judge resume reliable.

## What runs

### 1. Prepare and preflight

`prepare` checks the HF checkpoint, exact 220-row dataset, local reference
files, six deterministic shards, model profile, containers, and pinned source
files. The controller repeats the exact compute preflight on every start and
before publishing the final receipt, so a stale pass marker cannot bypass
source or input drift.

Reference attachments are content-hashed once during `prepare`. Routine
compute preflights compare their recorded stat signatures; the final controller
preflight and `result` hash every attachment again before accepting ELO.

For judging, `prepare` materializes a campaign-owned runtime from the PR #2588
commit plus the small TRUE3 transport patch. The shared checkout is never
modified. Every executed wrapper, config, helper, patched output, and source
module in this path is hash-pinned; a sparse checkout does not need to appear
globally clean.

The maintained defaults use:

- Slurm account `nemotron_n3_post`.
- The proven rollout checkout at commit
  `626d2c2654912ec2f0c62d2d440888751a3a5b96` and the separate NeMo Gym PR
  #2588 judging runtime at commit
  `d3f146d386c7dfe07d4fabce32c4c8b14c7917d2`.
- The `super35-nemotron` serving profile: TP4, BF16 weights, FP8 KV cache,
  262,144-token context, and the Ultra-v3 reasoning parser.
- The fixed local 220-task GDPVal dataset and nine AA-v2 reference anchors.

### 2. Rollouts

- Six disjoint modulo shards: `37, 37, 37, 37, 36, 36` tasks.
- One TP4 replica per shard, concurrency 20 per replica.
- Up to 120 simultaneous tasks and 24 GPUs while all six shards are active.
- One shared deliverables directory. Each shard gates success on the exact task
  IDs in its own JSONL, so completed sibling outputs are informational rather
  than being miscounted against the shard size.
- A finish marker containing an object or JSON `null` is terminal. Stirrup uses
  `null` for a normal max-turn completion without explicit finish arguments;
  malformed JSON, lists, and scalars still fail closed.
- Immutable shard JSONL files are copied into each rollout run directory before
  Gym starts, so generated `*_prepare.jsonl` and metrics siblings cannot pollute
  the campaign's hash-checked `shards/` directory.
- Every Slurm rotation gets a fresh replica output/readiness directory, so an
  old `server_info/*.env` cannot make a restarted vLLM replica look ready.
- Rollout teardown is idempotent and preserves Gym's original exit code; an
  already-exited Gym or serving process cannot turn a completed shard into a
  failed Slurm job.
- Missing tasks are repacked into fresh residue shards at concurrency up to 8;
  completed tasks are never intentionally rerun.
- Nested recovery submissions scrub the controller allocation's inherited
  `SLURM_MEM_PER_*` and `SBATCH_MEM*` values before `sbatch`, so the controller's
  8 GB limit cannot cap a rollout child or its serving `srun`.
- Up to six bounded residue rounds.

This is the main speedup over a serial or low-concurrency run. Concurrency 20
per TP4 replica is the proven stable point for this profile; pushing it higher
can lose more time to memory pressure and long-tail retries than it saves.

### 3. Office closure

Model-produced Office files are converted once before judging. Conversion uses
eight workers, a 900-second per-file timeout, and at most four attempts. Valid
same-stem PDFs are reused. The source inventory and closure fingerprint must
remain unchanged on resume.

Identical duplicate OOXML ZIP members can be repaired in a staged copy;
conflicting duplicates fail closed. Source files are not modified. Office files
under `reference_files` are reference inputs and are not part of this produced-
artifact conversion gate; their audited sidecars are built in the transport
phase below.

### 4. Transport prebuild

After Office closure, a separate CPU job builds the candidate and nine
reference transport views before provider authorization is required. The job
has a two-hour Slurm limit, each FFmpeg conversion has a 30-minute limit, and
the controller permits at most two attempts across every resume. Publication
is atomic: an interrupted attempt cannot expose a partial view, while a
completed manifest is validated and reused by every judge attempt.

The HSG host does not need media tooling. The prebuild runs FFmpeg, PyMuPDF, and
SoundFile inside the pinned GDPVal container. Large PCM audio, including
WAV/AIFF members inside ZIP files, is losslessly normalized to FLAC and checked
by decoded-sample identity. Reference views reject every unrecorded path;
candidate views additionally allow only the exact dynamic Gym judge-cache
filenames.

Nested benchmark inputs under `reference_files/<asset-id>/` are included in
deterministic relative-path order. Office sources use a provenance-safe PDF
sidecar; missing Office sidecars are rendered with the pinned converter, STEP
inputs are rendered as complete monospaced text PDFs, and PSD inputs are
flattened losslessly into a one-page PDF. Source files are never modified, and
every derivative records its source/output hashes and renderer parameters.

Video files of at least 8 MiB are normalized to H.264, at most 1280×720, CRF 26,
with AAC 128 kb/s audio. This bounded visual proxy is intentionally lossy: it
is required for the GDPVal reel task whose full three-section inline payload
exceeds the provider gateway even after lossless audio conversion and exact
deduplication. Duration, stream presence, dimensions, codecs, tool versions,
and output hashes are validated. The exact profile
`reference-pdf-v1+video-h264-720p-crf26-aac128-min8m+ref-video-bundle8-v1` is
recorded in the transport manifest and in the scientific run fingerprint.
When one recursive reference ZIP contains more than eight videos, all logical
clips are normalized at fixed 1280×720/30 fps with source audio retained (or
deterministic silence when absent), then partitioned in stable path order into
at most eight contiguous MP4 bundles. No clip is sampled or dropped. A text
manifest beside the bundles gives each original archive path, source hash, and
exact half-open time range, and validation binds the source inventory,
boundaries, bundle identities, and hashes. Reserving two physical video slots
keeps the reference plus one A and one B deliverable within Gemini's ten-video
request limit. A shared immutable content-addressed cache under
`$AAV2_ROOT/transport_derivative_cache_v1` derives repeated benchmark inputs
once and hard-links the verified result into later candidate and reference
views. Each key binds the source bytes and extension, output type, complete
derivative profile, transport and Office-converter source hashes, exact GDPVal
container hash, and media-tool identities. Per-key locks make concurrent
campaign builders converge on one atomic object. Every hit is rehashed before
use; a malformed or drifted object is atomically quarantined and rebuilt. The
campaign manifest records the cache identity and hit/miss/corruption counters,
while the published derivative remains subject to the ordinary full transport
validation.

### 5. Multistage judging

- Stage 0 plans 45 calibration tasks against nine references.
- Stage 0 completes normally with 45 rows, or can advance with 41–44 successful
  rows only when PR #2588 records an accepted partial-completion receipt: the
  missing rows must be persisted `timeout_exceeded` or `transient` failures,
  overall success must be at least 90%, and every reference must satisfy the
  configured coverage floor. All other failure classes remain non-waivable.
- Stage 1 evaluates all 220 tasks against the selected top four references.
- Every accepted row has four trials drawn from GPT-5.5, Gemini 3.1 Pro, and
  Claude Opus 4.8.
- The TRUE3 profile rejects any comparison request with fewer than four valid
  votes, any invalid trial, or any reference error before Stirrup can cache the
  row or multistage planning can consume it. The controller's sanitizer remains
  a recovery boundary for legacy or interrupted output written by older
  runtimes.
- The three judges use distinct model servers. Gemini and Claude retain their
  configured high/adaptive reasoning controls.
- PDFs stay native for Gemini and Claude when their provider limits allow it.
  A Gemini PDF above 50 MB is deterministically rasterized page-for-page before
  dispatch; this is distinct from the 495 MB whole-request cap.
  GPT receives page images and text, with deterministic DPI fallback down to
  72 DPI (including the proven 81-DPI boundary) before its exact 30 MB request
  cap is reached. Audio and video route only to Gemini.
- Before provider calls, a deterministic assignment repair moves only
  transport-incompatible or incomplete-reference task/reference pairs,
  preserves every selected reference's exact task count, and freezes the
  repaired plan for resume. If no count-preserving compatible assignment
  exists, the stage fails before a billable request instead of silently
  dropping media or judging an unfinished anchor.
- Stage 1 is strict: exactly 220 rows and 880 valid trials are required for the
  headline ELO. There is no partial final ELO.
- Judge attempts use concurrency `16`, retry once at `16`, then downshift to `8`
  and `4`, always with Gym `--resume` against the same output and cache.
- Experimental persistent lifecycle is opt-in with
  `CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION=true` for import-only judging. It
  starts Ray and the seven Gym services once, then runs strict resume passes at
  `16 -> 8 -> 4 -> 1` inside that same Slurm allocation. The exact result gate
  runs after every pass; exhaustion exits nonzero and never publishes a partial
  final ELO. Unset/false retains the proven one-pass-per-job lifecycle.
- Each judge Slurm job derives a deterministic head-server port and job-scoped
  component-port range from its job ID. It holds a node-local slot lock and
  probes every port before use, moving forward if its first slot is occupied.
  Concurrent campaigns on a shared CPU node therefore do not all contend for
  Gym's process-global default port.
- The internal judge task timeout is 25 minutes, leaving five minutes to persist
  its failure before the 30-minute outer no-progress watchdog can stop the
  attempt and advance to the next concurrency.

### Provider-tail diagnosis

- `429`, `RateLimitError`, or `RESOURCE_EXHAUSTED` indicates quota/backoff.
- Gemini `400 INVALID_ARGUMENT` on an otherwise valid PDF can indicate a
  per-document limit; inspect the largest inline PDF, not only total request
  bytes. The runtime preflight must handle the provider limit before dispatch.
- Claude `Timeout(timeout=360.0)` with local request timeout 900 means the
  upstream LiteLLM timeout won. Pin `timeout` and `request_timeout` on the
  Claude model-server request; lowering effort or counting the row as a loss is
  not an equivalent evaluation.
- Stage 1 remains unpublished until all 220 rows and 880 votes are present.

Before billable calls, the job checks that the configured endpoint advertises
all three exact judge model IDs. It does not rerun a full live canary suite for
every checkpoint.

### Controlled 217+3 fast-tail repair

Use this path only when an imported or interrupted campaign has exactly the
same candidate rollout evidence as an existing judged run. It is an
operator-controlled mixed-runtime repair, not the normal launcher path and not
a generic arbitrary-tail resume feature.

The proven fast path reused 217 strict-valid Stage 1 rows and judged only the
three missing rows. Keeping Ray and all seven Gym services alive inside one
persistent allocation reduced the repair to the provider time for those three
rows instead of repeating all 220 tasks.

Minimum safe workflow:

1. Prove outside the seeder that the source and target use byte-identical
   candidate rollout/deliverable evidence, the same judge policy, and the same
   frozen task-to-reference assignments. Similar checkpoint names or rollout
   roots are not sufficient. `seed_compatible_judgments.py` does not itself
   hash-compare the source and target deliverable trees.
2. Dry-run `seed_compatible_judgments.py` with the source judge directory,
   target judge directory, and target deliverables directory. The target must
   already contain its frozen 220-task Stage 1 plan and exact 220-row
   `preprocessed_datasets/benchmark.jsonl`.
3. Accept only strict-valid source rows: four judged trials, zero invalid
   trials, no judge or reference errors, the expected three-model judge panel,
   and the exact target reference assignment. For this procedure, require the
   summary to report `stage1_rows_after=217` and `stage1_rows_remaining=3`.
4. Apply once. Preserve the generated read-only backup and seed receipt, then
   make a read-only pre-tail snapshot of the corrected 217-row output. The
   dispatcher `_ng_task_index` must be the task ID's row position in the target
   preprocessed dataset, never its position in the Stage 1 plan. The seeder now
   enforces this mapping.
5. Resume only the three missing tasks with the v1.4.11 runtime and
   `CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION=true`. The persistent session keeps
   services warm and uses bounded passes at `16 -> 8 -> 4 -> 1`; downshift only
   after measured lack of progress. Use
   `GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS=1` when overlapping campaigns or after
   provider throttling.
6. Publish nothing until `campaign.py result` proves exact Stage 1 coverage of
   220 tasks and 880 trials with `invalid=0`. Stage 0 may use only the bounded
   PR #2588 partial-completion policy; Stage 1 is never partial.
7. Run `transition_receipt.py` with the old and new runtime manifests, v1.4.11
   package root, seed receipt, pre-tail snapshot, exact three task IDs, frozen
   fingerprint, every contributing Slurm job ID, final JSONL, and strict result
   JSON. Keep its mode-0400 receipt and matching SHA-256 sidecar.

Always pass `--pre-tail-output` to the transition receipt. Without the snapshot,
the receipt cannot prove that the final file preserved all 217 old rows and
added exactly the named three. The receipt tool is intentionally limited to
this 217+3, v1.4.11 transition. Describe its output as a strict-valid mixed-
runtime fast-tail result, not as an all-row v1.4.11 rejudge.

## Why the next run should be much faster

The reusable path incorporates the main lessons from the previous campaign:

1. Start six TP4 replicas immediately instead of running four tasks globally.
2. Keep one stable run root; timestamped roots defeat resume.
3. Recover exact missing rollout rows, not an entire 220-task run.
4. Convert produced Office artifacts once and reuse the closed inventory.
5. Resume judging in place; the pinned runtime does not retain invalid judge
   replies as successful cache entries, and its strict vote gate stops them
   before multistage planning.
6. Downshift judge concurrency only after measured lack of progress, rather
   than restarting speculatively.
7. Allow a tightly bounded Stage 0 timeout/transient tail to advance calibration
   while keeping the full Stage 1 result strict.
8. Skip redundant full-payload audits and live canaries on every checkpoint;
   retain deterministic input, source, coverage, journal, vote, and ELO gates.

This does not make model inference literally 100 times faster. It removes the
large avoidable costs: idle GPUs, accidental four-way rollout concurrency,
whole-run restarts, repeated Office conversion, and manual judge-tail recovery.

For a healthy cluster and provider endpoint, budget roughly four hours for
rollouts plus about one hour for judging, in addition to Slurm queue time. A
the first transport prebuild for a new converter identity can add tens of
minutes (roughly 37–45 minutes was observed for the current nine references).
Later campaigns reuse the shared derivatives and retain one full validation;
the expected warm path is roughly 6–10 minutes, depending on Lustre metadata
latency. Tool timeouts, model startup, provider throttling, or difficult
artifact conversion can extend that.

## Commands and state

| Command | Behavior |
| --- | --- |
| `prepare CHECKPOINT` | Creates or verifies immutable campaign state; no jobs or provider calls. |
| `submit CHECKPOINT` | Submits preflight, six rollout shards, and one controller. |
| `status CHECKPOINT` | Prints read-only rollout/judge progress and the next action. |
| `resume CHECKPOINT` | Restarts a non-live controller using existing rollout and judge state. |
| `result CHECKPOINT` | Revalidates the exact dataset, output, journal, metrics, digest, and completion marker, then prints ELO. |
| `all CHECKPOINT` | Runs `prepare`, then `submit`. |

`submit` and `resume` share a nonblocking per-campaign lock. The controller has
its own singleton lock. Repeated commands therefore fail or reuse state rather
than launching two owners for one campaign.

Hidden `.slurm_submit_intents/` files are crash-recovery metadata for the gap
between Slurm accepting a job and the launcher publishing its job receipt. A
resume adopts exactly one matching Slurm job; zero or multiple matches fail
closed for manual Slurm reconciliation. Never delete an intent merely to make
resume submit again, because the original job may still exist.

Useful status states are:

- `PREPARED`: run `submit`.
- `RUNNING`: the HSG controller is active.
- `AWAITING_JUDGE_AUTHORIZATION`: rerun `resume` with the authorization flag.
- `RETRYABLE`: the controller is no longer live; run `resume`.
- `BLOCKED`: inspect `CONTROLLER_BLOCKED` and the newest log before resuming.
- `PASS`: run `result` for the authoritative receipt.

Slurm `COMPLETED` alone is not a scientific pass. `CAMPAIGN_COMPLETE` is written
only after the final strict validator succeeds.

## Provider authorization and secrets

Provider calls require the exact lowercase opt-in:

```bash
export CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS=true
```

The value is captured when the controller is submitted. If it was omitted,
rollouts, Office closure, and transport prebuild may finish, but judging waits
until an authorized `resume`.

The protected environment file is parsed line by line and only assignments
beginning with `export ` are loaded. It must include:

```bash
export TAVILY_API_KEY=...
export JUDGE_BASE_URL=...
export JUDGE_API_KEY=sk-...
export JUDGE_GEMINI_API_KEY=sk-...
```

`TAVILY_API_KEY` is used by rollout web search. `JUDGE_API_KEY` is the default
LiteLLM credential and `JUDGE_GEMINI_API_KEY` overrides it only for Gemini.
Values are loaded inside jobs and are not written into
campaign settings. Jobs use `--export=ALL`, so remove unrelated credentials
from the submission shell if they should not propagate.

The provider-free fingerprint probe loads the same protected environment as the
live judge. Endpoint and model identity remain fingerprinted; API-key values are
excluded so credential rotation cannot invalidate otherwise identical evidence.

## Common overrides

Defaults point at the maintained HSG installation. Override only when the
replacement is intentional and compatible:

| Variable | Purpose |
| --- | --- |
| `CHECKPOINT_E2E_ROOT` | Parent for deterministic campaign directories; default `$AAV2_ROOT/checkpoint_e2e_true3_v1_4_13_runs`. |
| `CHECKPOINT_E2E_ROLLOUT_GYM_ROOT` | Pinned checkout containing the proven rollout harness. |
| `CHECKPOINT_E2E_EXPECTED_ROLLOUT_GYM_REVISION` | Required full Git revision for the rollout checkout. |
| `CHECKPOINT_E2E_GYM_ROOT` | Pinned NeMo Gym checkout. |
| `CHECKPOINT_E2E_EXPECTED_GYM_REVISION` | Required full Git revision for that checkout. |
| `CHECKPOINT_E2E_DATASET` | Exact local GDPVal JSONL; must contain 220 unique task IDs. |
| `CHECKPOINT_E2E_REFERENCE_OVERLAY` | Nine-reference AA-v2 overlay. |
| `CHECKPOINT_E2E_ENV_FILE` | Protected runtime and judge environment. |
| `CHECKPOINT_E2E_ROLLOUT_SBATCH` | Compatible rollout wrapper override; default is the hash-pinned package-owned wrapper. |
| `CHECKPOINT_E2E_ACCOUNT` | Slurm account; default `nemotron_n3_post`. |
| `CHECKPOINT_E2E_ROLLOUT_CONCURRENCY` | Initial per-replica concurrency; default 20. |
| `CHECKPOINT_E2E_RECOVERY_CONCURRENCY` | Residue per-replica concurrency; default 8. |
| `CHECKPOINT_E2E_JUDGE_NO_PROGRESS_SECONDS` | Judge watchdog; default 1800, maximum 7200. |
| `CHECKPOINT_E2E_MAX_CONTROLLER_REQUEUES` | Controller wall-time requeues; default 2, maximum 4. |
| `CHECKPOINT_E2E_POLL_SECONDS` | Controller polling interval; default 60, maximum 300. |
| `GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS` | Fingerprinted per-campaign Gemini request cap; integer 1–4, default 2. Use 1 when campaigns overlap. |

The checkpoint path plus its inventory determines the run ID. Metadata and
files up to 64 MiB are hashed; large weights use size and nanosecond mtime.
Treat the checkpoint, dataset, overlay, runtime, containers, and prepared run
directory as immutable. Use a different `CHECKPOINT_E2E_ROOT` for an
intentionally different evaluation profile.

## Final acceptance gates

The final result must have:

- Exact 220/220 policy rollout completion coverage.
- Closed produced-Office inventory.
- Stage 0 complete at 45 rows, or a valid PR #2588 partial receipt with 41–44
  rows and only accepted persisted timeout/transient omissions.
- Exact Stage 1 coverage of the canonical dataset: 220 rows and 880 valid
  trials.
- Four valid votes per row, no judge/reference errors, and all three judge
  models represented.
- Journal plans, assignments, fingerprints, partial evidence, top-four
  references, and output rows that agree.
- A validated transport-v3 manifest whose derivative profile matches the
  fingerprinted judge overlay.
- Aggregate metrics whose headline ELO is exactly the complete Stage 1 fit.
- An independent anchored Bradley–Terry recomputation from the persisted Stage
  1 votes that matches the reported headline ELO and reference anchors.
- A mode-0400 final receipt, matching SHA-256 sidecar, and empty mode-0400
  completion marker.

Reasoning-token telemetry is intentionally not an acceptance gate. The serving
parser may expose inline reasoning under generated/output token counters; that
does not affect rollout coverage, judge votes, or the final ELO.

## Scope

The built-in serving profile supports the current Super-35/Nemotron checkpoint
family only. It does not infer tensor parallelism, parser choice, dtype, or
serving flags for an unrelated architecture. Add and validate a separate model
profile before using this launcher for another family.
