# Pi with Nemotron 3.5 Super

`pi-nemotron.yaml` runs Pi against the hosted Nemotron text-preview endpoint
directly. Gym supplies MMLU-Pro tasks and grades answers with its MCQA resources
server. No Gym model server or sandbox provider is required for this benchmark.

Install Pi 0.87.1 and ensure it is the `pi` executable on PATH. Gym's existing
Pi installer does not replace an executable already on PATH, even when a
different `pi_version` is configured.

Export `NVIDIA_INFERENCE_HUB_API_KEY` in the shell that starts Gym. The config
passes the environment-variable reference to Pi rather than putting the key
in YAML. Reasoning controls are omitted pending a verified endpoint contract.
The 32,768 context budget is conservative; it is not the model's maximum
context claim. The output budget is 8,192 tokens per model call. Tools are
disabled for this multiple-choice benchmark.

```bash
gym eval prepare --benchmark mmlu_pro

gym env start --config benchmarks/mmlu_pro/pi-nemotron.yaml
```

From another shell, collect a five-task integration smoke:

```bash
gym eval run --no-serve \
  --agent mmlu_pro_pi_agent \
  --input benchmarks/mmlu_pro/data/mmlu_pro_benchmark.jsonl \
  --prompt-config benchmarks/prompts/eval/aai/mcq-10choices.yaml \
  --output results/mmlu-pro-pi-nemotron.jsonl \
  --limit 5
```

The local smoke on 2026-09-28 used this config with real Gym PiAgent and MCQA
HTTP servers at Gym commit `8320fad15`, Pi 0.87.1, and the first five upstream
test rows. It produced five graded attempts, three correct. That is an
integration check, not a full MMLU-Pro estimate or evidence of coding-benchmark
performance. The smoke runner started the servers programmatically; the
standard CLI config was separately checked with `gym env resolve`.

For DeepSWE, Pi still calls Nemotron directly, but the agent must run in a task
sandbox with the correct repository and interpreter. Gym's existing DeepSWE
resources server then collects the committed patch and grades it in a fresh
sandbox. That coding-benchmark integration is separate from this MCQA recipe.
