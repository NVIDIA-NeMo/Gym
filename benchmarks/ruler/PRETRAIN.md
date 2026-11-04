# RULER for pretraining and midtraining checkpoints

This benchmark flavor reproduces RULER's text-completion protocol for base,
pretraining, and midtraining checkpoints. It uses the existing RULER verifier;
only data preparation and model serving differ from the chat-model benchmark.
It requires the native vLLM completions support introduced by #1300.

Prepare one sequence length with the tokenizer for the checkpoint under test:

```bash
gym eval prepare --benchmark ruler/config_pretrain \
  "+prepare_script_args.model='/path/to/checkpoint'" \
  "+prepare_script_args.length=262144" \
  "+prepare_script_args.data_format=default"
```

`default` appends RULER's `answer_prefix` directly to the question, matching
NeMo Skills' `start_assistant_response_key=generation` behavior. `base` inserts
one newline before the prefix. Per-task generation limits are embedded in each
prepared request: 128 tokens for NIAH, 30 for variable tracking, 120 for
common-word extraction, 50 for frequent-word extraction, and 32 for QA.

Run against a Gym-managed local vLLM server:

```bash
gym eval run \
  --benchmark ruler/config_pretrain \
  --model /path/to/checkpoint \
  --output results/ruler_pretrain.jsonl \
  --split benchmark \
  --temperature 0 \
  --top-p 1 \
  --resume \
  ++overwrite_metrics_conflicts=true \
  ++reuse_existing_data_preparation=true \
  ++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.max_model_len=262144
```

Override tensor parallelism for the available hardware. The bundled model
profile sends the prepared prompt through vLLM's `/v1/completions` endpoint,
defaults to one GPU, and deliberately leaves cluster-specific CUDA, cache, and
scheduler settings to the launcher.
