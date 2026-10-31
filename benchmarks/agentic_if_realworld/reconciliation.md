# Interrupted-run reconciliation

Use this procedure when the adaptive benchmark stops with an ambiguous dispatch,
interrupted body, missing receipt or missing Gym projection. The judge-receipt
checks also apply to the native Resources Server. A technical interruption is
unscored; it is not an instruction-following failure.

## 1. Stop writers and preserve the run

Stop new submissions and let outstanding tasks settle where possible. Confirm
that the driver and servers using this receipt directory are no longer writing
before making a private backup. Lock files can remain after a process exits;
deleting `.lock` or `score.lock` does not resolve an uncertain dispatch.

Preserve the entire receipt directory, model-call captures, logs, rollout output,
`rollouts_materialized_inputs.jsonl`, `resolved-config.yaml` and any `.run.json`,
summary and aggregate files. Keep the original paths and bytes intact: receipts
contain absolute paths and cross-file hashes. A hard termination may leave some
reports absent or behind the settled journals. Store backups outside Git with
restricted access; they contain real inputs, criteria and answers.

## 2. Locate the interrupted attempt

Match the task ID in the error log to a cell journal's `item_id`. Under the
configured `AGENTIC_IF_RECEIPT_DIR` (or YAML `receipt_dir`), inspect:

| Artifact | What to inspect |
| --- | --- |
| `cells/<task-hash>.json` | `attempts`, `transport_failures`, and any `selected` response. Each policy attempt has a `receipt_id`; a settled attempt also has `receipt_path` and `receipt_sha256`. |
| `model/<receipt_id>.json` | Policy transport status and request binding. Preserve its `.binding.json` companion when present. |
| `judges/<request-fingerprint>/payload.json` | Match `item_id` to the task whose saved answer is being judged. |
| `judges/<request-fingerprint>/attemptN.json` | Judge transport status, plus any cached `result.json` in the same directory. |
| Each transport receipt's `raw_attempts` | All recorded HTTP-body paths and hashes, including earlier bounded retries. Preserve adjacent `.aN.http` files even if a crash left them unindexed. |

Read `status`, `dispatch_status`, `http_status`, `response_body_complete`, `reason`
and the latest raw attempt together. A missing field, empty body, or missing
receipt is not proof that nothing was sent. If necessary, ask the endpoint owner
to correlate the endpoint, recorded time and available request/response IDs with
provider logs. Keep this evidence private; do not paste traces or credentials
into a public issue. Provider logs alone do not update the local receipt state.

## 3. Choose the supported next step

These resume cases require settled, consistent journals and matching hashes.
A lingering `status: started` or missing binding still blocks resume.

| Recorded outcome | Next step |
| --- | --- |
| Confirmed `dispatch_status: not_dispatched`, with a settled error receipt | Fix connectivity, then explicitly resume. Policy generation retries the same seed/token cap; judging retries the saved answer. |
| Judge error receipt with received HTTP 3xx/4xx/5xx headers, including a truncated error body | Fix the service or credentials, then explicitly resume judging. Non-2xx responses do not consume the two-review budget. Transient 429/5xx responses may already have exhausted their bounded transport retries. |
| Complete policy stream already accepted as `status: generated`, with a settled journal and trusted Gym projection | Resume normally. A read error after a saved terminal event and `[DONE]` is already handled by the adapter; keep the first complete answer. |
| Interrupted 2xx body without an accepted complete delivery; `dispatch_status: unknown` without the judge non-2xx evidence above; `status: started`; or missing receipt/projection | Keep the run stopped. This release has no general repair/import command for these states. Preserve the evidence for a maintainer-reviewed recovery. |

Never make a receipt resumable by changing its status, adding `[DONE]`, rewriting
hashes, deleting an attempt, or restoring a snapshot that omits later dispatches.
Do not regenerate a saved complete policy answer to repair a judge failure.

## 4. Resume or record an unresolved run

For a supported resume, repeat the original `gym eval run` command with
`--resume`, the same task selection, output/receipt/capture paths, model settings,
Gym commit and dependencies. The runtime fingerprint includes the commit even
when its file contents are unchanged. Fixing a credential's value is permitted;
changing the endpoint, model or other fingerprinted settings requires a new run.
Resume validates all selected journals before dispatch and preserves accepted
answers and grades. If validation rejects the run, retain the error and stop.

If delivery remains uncertain, record the affected task/attempt, evidence hashes
and decision in a separate private incident note. Keep the original run and its
declared denominator incomplete; report missing/unscored tasks separately from
failures, with complete accuracy null. A replacement evaluation must use fresh
output, receipt and capture locations and its own run identity. Do not combine
its answers or counts with the stopped run or copy ambiguous receipts into it.
Reusing or importing recovered answers requires a separately reviewed recovery;
it is not supported by editing the frozen run's files.
