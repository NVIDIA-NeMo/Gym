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

The model server's reasoning and tool-call parsers are authoritative. The
policy adapter decodes the standard API transport envelope, validates the
declared tool schema, and executes valid calls without repairing malformed
JSON, completing missing delimiters, or guessing tool aliases. Invalid parser
output follows the ordinary bounded parse-retry or failure path. The shared
executor rejects pathological scroll amounts instead of silently changing
them.

For setup, smoke, full execution, and reconciliation, use [runbook.md](runbook.md).

## Validation evidence

A pair of sequential full-population runs exercised the parser-faithful
contract above with the tuned `iter_0004622` checkpoint:

| Repetition | Result | Completeness |
| --- | ---: | --- |
| r1 | 415/552, 75.18% | 552 valid unique; no missing, malformed, or duplicate tasks |
| r2 | 420/552, 76.09% | 552 valid unique; no missing, malformed, or duplicate tasks |

The repetitions used the same frozen source, model serving recipe, task data,
browser runtime, proxy/CAPTCHA service, judge, and generation parameters. The
mean was 417.5/552, or 75.63%, and the runs differed by five tasks.

A previous reference-aligned Gym control completed the maintained population
at 428/552, while the maintained golden was 429/552. A later hash-sealed PR
candidate completed 421/552 with all 552 task IDs accounted and no unresolved
infrastructure rows. Those historical recipes enabled bounded post-parser
recovery and therefore are not results for the parser-faithful profile described
above. These numbers describe their recorded code, model, proxy, CAPTCHA, and
live-site state; they are not guarantees for a later public-site run.
