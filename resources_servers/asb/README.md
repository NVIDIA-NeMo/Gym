# Agent Security Bench resources server

This server reproduces Agent Security Bench (ASB) verification for prompt injection,
observation injection, memory poisoning, mixed attacks, and plan-of-thought backdoors. It
scores attack success, legitimate-task utility, workflow validity, refusal, and memory
retrieval provenance from the trajectory produced by `asb_agent`.

The benchmark rows are materialized from the pinned upstream revision rather than vendored.
See [`benchmarks/asb/README.md`](../../benchmarks/asb/README.md) for preparation, execution,
denominators, and protocol deviations.
