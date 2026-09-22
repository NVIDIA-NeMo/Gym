# GSM8K reference scorer

This deterministic resources server reproduces the numeric extraction and exact-match scoring in lm-evaluation-harness GSM8K v3 tasks. It returns both strict `#### <number>` and last-number flexible scores, with strict match as the rollout reward.

```bash
gym env test --resources-server gsm8k_reference
```

The benchmark integration is documented in [`benchmarks/indic/gsm8k`](../../benchmarks/indic/gsm8k/README.md).
