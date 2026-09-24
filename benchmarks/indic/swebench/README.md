<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Indic SWE-bench

Translated SWE-bench Verified from `ai4bharat/indic-swe-bench`: 500 issues per
language across all 14 Indic languages (7,000 tasks). Uses the
[English Verified configuration](../../swebench/verified/opencode.yaml)'s prompt,
OpenCode agent, sandbox images, test verifier, and three attempts per issue.
Only the problem statement changes; repository commits, patches, and tests are preserved.

Indic has separate agent and resources-server names, so English and Indic can
run in the same Gym stack without replacing either dataset.

## Prepare data

Preparation loads the dataset's `test` split with Hugging Face `load_dataset`.
All languages are included by default: `as`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`,
`ne`, `or`, `pa`, `sa`, `ta`, `te`, and `ur`.

```bash
gym eval prepare --benchmark indic/swebench +use_cached_prepared_benchmarks=false
```

For per-language evaluation, select a language during preparation. English is
also available as `en`:

```bash
gym eval prepare --benchmark indic/swebench \
  '+prepare_script_args={languages:[hi]}' \
  +use_cached_prepared_benchmarks=false
```

## Collect rollouts

Complete the [OpenCode prerequisites](../../../responses_api_agents/opencode_sandboxed_agent/README.md)
and configure the model endpoint and OpenSandbox access. Set OpenCode's
`opencode_max_context_window` to match the served model's context limit.

```bash
gym eval run --benchmark indic/swebench \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type vllm_model \
  --model MODEL_NAME \
  --model-url http://HOST:PORT/v1 \
  --model-api-key dummy \
  --split benchmark \
  --output results/indic_swebench/rollouts.jsonl \
  --concurrency 1 \
  +use_cached_prepared_benchmarks=true
```

Use the same model and generation settings for English and Indic comparisons.
For thinking models, configure vLLM's reasoning and tool-call parsers to separate
post-thinking content and structured tool calls. The SWE-bench verifier scores
the executed repository patch against the original tests.

Start with `--limit 1 --num-repeats 1` to check sandbox execution and verification.
The dataset declares Apache-2.0; underlying repositories retain their own licenses.
Prepared data is generated locally and excluded from Git.
