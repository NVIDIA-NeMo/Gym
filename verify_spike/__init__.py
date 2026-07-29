"""Standalone trajectory-health verification spike for NeMo-Gym artifacts.

Port of the evalpipeline trajectory health checks (nvidia-eval-factory-
benchmarking MR !929) that consumes NeMo-Gym-native artifacts directly:
``evaluator_rollouts.jsonl`` records plus the per-rollout
``model_calls/<rollout_id>.capture.jsonl`` sidecars. No ATIF projection,
no evalpipeline imports, stdlib only.

Doctrine carried over unchanged from the source:

- Flags never gate: every signal beyond the clean-rollout verdict is a
  triage flag, never an automatic failure.
- Fail-soft on missing surfaces: a rollout without a sidecar capture
  still gets a verdict; capture fields stay None.
- Missing metrics is its own signal (missed_metrics_count), never
  conflated with a zero-token step.
- Reconciliation ordering is load-bearing: the success filter runs
  BEFORE dedup, else a failed call and its byte-identical successful
  retry collapse onto the failed row.

Run: ``python3 -m verify_spike <artifacts_dir> [--out DIR]``
"""
