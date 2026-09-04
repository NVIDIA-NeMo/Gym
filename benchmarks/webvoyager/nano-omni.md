# Nano Omni on WebVoyager

Nano Omni is one policy protocol on the common `visual_browser` runtime and
the common 552-task dataset. It does not select a separate browser harness.

```text
552-task dataset
  -> web_agent policy_protocol=nano_omni_toolcall
  -> OpenAI-compatible policy endpoint
  -> visual_browser headed Chromium/PyAutoGUI runtime
  -> WebVoyager Gemini trajectory judge
```

The reproducibility contract is pinned in `nano_omni_recipe_lock.json`:

- model-specific tokenizer, chat template, multimodal processor, and parsers
  are policy-server assets, not browser assets;
- generation uses temperature 0.1, top-p 0.95, and 16384 output tokens;
- `chat_template_kwargs={"truncate_history_thinking": false}` is passed to
  vLLM Chat Completions;
- the policy sees three recent screenshots and may take up to 100 browser
  steps;
- the browser and judge use the same proxy/CAPTCHA and evidence behavior as
  the Qwen profile.

The checked-in public-model serving profile is
`responses_api_models/local_vllm_model/configs/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16-alignment.yaml`.
Do not combine a checkpoint with tokenizer or template assets reconstructed
for a different model.

The policy adapter applies bounded syntax recovery to outputs the model already
chose: JSON-string decoding, one missing closing bracket, and known tool-name
aliases. It does not rewrite tasks or add a browser strategy. The shared
executor clamps pathological scroll amounts to protect the worker from a
model-generated `scroll 100000` stall.

For setup, smoke, full execution, and reconciliation, use [runbook.md](runbook.md).

## Validation evidence

A previous reference-aligned Gym control completed the maintained population
at 428/552, while the maintained golden was 429/552. A later hash-sealed PR
candidate completed 421/552 with all 552 task IDs accounted and no unresolved
infrastructure rows. These numbers describe their recorded code, model, proxy,
CAPTCHA, and live-site state; they are not guarantees for a later public-site
run.
