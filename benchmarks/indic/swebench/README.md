<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Indic SWE-bench

Translated SWE-bench Verified from `ai4bharat/indic-swe-bench`: 500 software issues
in each of 14 Indic languages. Reuses the [English Verified configuration](../../swebench/verified/opencode.yaml),
its prompt, OpenCode agent, repository images, and test verifier. Only the issue
text changes; repository commits, patches, and test lists remain unchanged.

## Configuration

The default is three attempts per issue, matching English Verified. Each attempt
receives reward 1 when the existing SWE-bench harness reports `resolved`, otherwise
0. Report mean resolved rate for comparison with English; success in any of three
attempts is pass@3 and is a different metric. Compare languages using identical
model, sampling, agent, context, and sandbox settings. Check `evaluation_completed`
and rollout coverage before reporting scores.

Configure model access and follow the existing [OpenCode prerequisites](../../../responses_api_agents/opencode_sandboxed_agent/README.md).
OpenCode's context window must match the model being evaluated.

## Prepare data

```bash
gym eval prepare --benchmark indic/swebench
```

Defaults to `as`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `ne`, `or`, `pa`, `sa`, `ta`,
`te`, and `ur` (7,000 tasks). For per-language results, prepare and run one language
at a time. English is available as `en`:

```bash
gym eval prepare --benchmark indic/swebench \
  '+prepare_script_args={languages:[hi]}' \
  +use_cached_prepared_benchmarks=false
```

Use `instance_ids` in `prepare_script_args` to select tasks for a smoke test.
Source data is pinned to revision `f03e95b7749c7f06b8aa1a2ba75360d9fcb26a43`.
Missing translations are rejected. Generated data remains local.

## Collect rollouts

```bash
gym eval run --benchmark indic/swebench \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type vllm_model \
  --model MODEL_NAME \
  --model-url http://HOST:PORT/v1 \
  --model-api-key dummy \
  --split benchmark \
  --output results/indic_swebench/rollouts.jsonl \
  --concurrency 1
```

The dataset declares Apache-2.0. Underlying repositories retain their own licenses.
