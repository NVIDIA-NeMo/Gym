# Serving the HarmBench classifier for this environment

The verifier calls `cais/HarmBench-Llama-2-13b-cls` through NeMo Gym's `inference_provider`
model server, i.e. an OpenAI-compatible `/v1/chat/completions` endpoint. Upstream HarmBench feeds
the classifier a *raw* Llama-2 `[INST] <<SYS>> ...` string through vLLM's `LLM.generate`, so the
chat endpoint must not add any chat formatting of its own. `harmbench_classifier_chat_template.jinja`
is a passthrough template that emits `bos_token`, a space, and the user content verbatim; with it the
chat request tokenizes identically to upstream's raw prompt (`[1, 518, 25580, ...]`, verified with
vLLM `/tokenize`).

Reference vLLM launch (any host; a Modal web endpoint in the FDR environment was used for the
reference run):

```bash
vllm serve cais/HarmBench-Llama-2-13b-cls \
  --revision bda705349d1144fa618770bea64d99ce54e3835b \
  --served-model-name cais/HarmBench-Llama-2-13b-cls \
  --dtype bfloat16 --max-model-len 2048 --generation-config vllm \
  --chat-template resources_servers/harmbench/classifier/harmbench_classifier_chat_template.jinja
```

Then point the environment at it:

```bash
export HARMBENCH_CLASSIFIER_BASE_URL=http://<host>:8000/v1
export HARMBENCH_CLASSIFIER_API_KEY=<key or placeholder>
```

`benchmarks/harmbench/calibrate.py` re-scores every generation through the raw `/v1/completions`
path of the same server and checks tokenization parity, so a misconfigured template shows up as a
calibration failure rather than a silently shifted ASR.
