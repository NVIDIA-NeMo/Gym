# GSM8K reference scorer

This resources server scores numerical answers with strict `#### <number>` and
last-number flexible exact match. Strict match is the rollout reward.

The Indic benchmark uses [ai4bharat/indic-gsm8k](https://huggingface.co/datasets/ai4bharat/indic-gsm8k).
The verifier scores final assistant text; separated reasoning items are excluded.

```bash
gym env test --resources-server gsm8k_reference
```

The benchmark integration is documented in [`benchmarks/indic/gsm8k`](../../benchmarks/indic/gsm8k/README.md).
