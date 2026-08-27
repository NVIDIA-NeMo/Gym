# PrimeVul

[PrimeVul](https://github.com/DLVulDet/PrimeVul) evaluates binary vulnerability detection on
textually similar vulnerable and fixed C/C++ function pairs. A pair is correct only when both
members are classified correctly.

- Kind: manifest-backed benchmark
- Integration profile: `custom-gym-verifier`
- Canonical split: upstream paired `test` split (435 pairs, 870 rows)
- Agent: `responses_api_agents/simple_agent`
- Prompt: the published PrimeVul standard-classification protocol in [`prompt.yaml`](prompt.yaml)
- Headline metric: `mean/paired_accuracy` (P-C)

The dataset mirror revision, verifier behavior, secondary metrics, and licensing qualifications are
documented in the [resources-server README](../../resources_servers/primevul/README.md).

## Validate and run

```bash
gym env validate primevul
gym env test primevul
gym env publish primevul

gym eval prepare --benchmark primevul
gym env start --benchmark primevul --model-type vllm_model
gym eval run --no-serve \
    --benchmark primevul \
    --output results/primevul.jsonl \
    --num-repeats 1
```

The committed benchmark JSONL contains five synthetic rows for offline validation. `gym eval
prepare` replaces it with the pinned canonical split. Do not use `--limit`, because it may cut a pair
in half; pass `+prepare_script_args.max_pairs=N` to preparation when a whole-pair subset is needed.
