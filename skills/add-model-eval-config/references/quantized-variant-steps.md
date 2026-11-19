# A quantized variant: the six steps

Moved from section 9 of [SKILL.md](../SKILL.md), with semicolons replaced by commas or periods and no other change. Section 9 there keeps the rule and the smoke check.

1. Check that the export came from the BF16 checkpoint before you write anything: `config.json` matches
   the BF16 one outside `quantization_config`, and a few tensors the export keeps unquantized
   (`exclude_modules` in `hf_quant_config.json`, such as the embeddings or `lm_head`) are byte-identical
   to the BF16 checkpoint's. Also compare `generation_config.json`, including whether it exists: with
   `--generation-config auto` its values are the defaults for every sampling setting a request leaves unset,
   so a different file changes the sampling too. If the two differ, stop and ask the export's owner to
   align them. Do not override the benchmark's sampling to compensate. Record what you compared.
2. Copy the BF16 config and change only these fields, each with a comment naming its source:
   - `model` (and its mount): the export.
   - `served_model_name`: the model's name without its precision suffix plus the export's actual
     precision, such as `<model>-nvfp4` or `<model>-fp8`, never `<model>-bf16-nvfp4`, so the two runs never
     merge.
   - The KV-cache and state-cache dtypes (`--kv-cache-dtype`, a hybrid model's Mamba cache dtypes), only
     where the export's serving recipe sets them. With no serving recipe, add none and ask the export's
     owner for it. Never copy cache flags from another model's config: an uncited flag is a guess.
3. Keep everything else as in BF16: the benchmark settings (section 8, left column), the parallelism, the
   parsers, the chat template, `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`, the image,
   `env`, the client and `driver.gym_install.ref`. Do not add `--quantization`: vLLM detects a quantized
   export. If the export ships its own chat template, diff it against the BF16 one, serve the BF16 one and
   report the difference to the export's owner. A slower load raises `health_check.timeout_seconds` in the
   BF16 config, so both runs keep one limit.
4. Resolve both (`gym eval submit --config <file> --resolve-only`) and diff them. Only these may differ:
   the fields of step 2, each benchmark's `run.policy_model_name` (submit derives it from the served name),
   and the MLflow fields of steps 5 and 6 (the precision tag, the `quantization_run_*` tags and the experiment name).
   The `pair_id` and every other serving, benchmark and scoring setting stay identical. Run both at the same
   Gym commit with the same `num_repeats`, and keep the policy timeout on the model (run-and-babysit-eval,
   section 6). Repeat the diff after any change to the BF16 config.
5. Tag both runs so they pair in MLflow: `mlflow_run_tags` with the precision and an id the two runs share,
   for example `{precision: nvfp4, pair_id: <id>}`. The served name is not the pairing tag. Put the tags
   where the BF16 config already tracks the run: under `driver.benchmarks.<name>.run` for live tracking, or
   in the MLflow export it runs after the rollouts (`job.export.mlflow`, or a benchmark's
   `driver.benchmarks.<name>.export.mlflow`). The keys are the same in both places. Merge them into the
   existing tags and do not add a second exporter. Tags alone do not turn tracking on (it needs the tracking
   URI, experiment and run name): if the BF16 config does not track, say that the pair is not recorded in
   MLflow.
6. Carry the export's provenance. When the quantization run was tracked (an MLflow run, for example) and
   the requester gives its name, id or URL, add them as the tags `quantization_run_name`,
   `quantization_run_id` and `quantization_run_url`, in the same place as step 5, and use the experiment
   the requester names as `mlflow_experiment_name`. The evaluation run stays on your own tracking server and
   keeps its own run name. The tags only link it to the quantization run. Do not copy that run's tracking URI
   or experiment id. With no such run, there is nothing to carry.
