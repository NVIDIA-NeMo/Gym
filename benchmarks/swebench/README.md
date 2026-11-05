# SWE Bench

Adds multiple SWE benchmarks to Gym. These are useable across agents/harnesses.

## Details

- Data source: `princeton-nlp/SWE-bench_Verified`
- SWE Bench Multimodal (`benchmarks/swebench/multimodal/opencode.yaml`): the dev split of
  `SWE-bench/SWE-bench_Multimodal`, 100 JavaScript tasks. The prepare script downloads each
  issue's images and attaches them to the prompt, and stores the test patch's reference images
  for the verifier, so it needs internet access; rollouts and verification do not. See
  [SWE Bench Multimodal grading](../../resources_servers/swebench/README.md#swe-bench-multimodal-grading).
  This recipe uses `single_agent_turn` and `opencode_agent`; SWE-bench resources own the
  task sandbox and OpenCode borrows it. The agent instance is
  `swebench_multimodal_opencode_agent`, and its dataset is now declared on resources
  with taskset `swebench:multimodal`. Use a model that supports images and tool calls.
  Training token-ID capture is opt-in; rebaseline scores when changing harnesses.
  The test split is not prepared, because the pinned SWE-bench harness (swebench 4.1.0) has no
  environment specs for its repositories.

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --config benchmarks/swebench/verified/opencode.yaml

# Running servers
gym env start \
    --model-type vllm_model \
    --config benchmarks/swebench/verified/opencode.yaml

# Collecting rollouts
gym eval run --no-serve \
    --agent mini_swe_agent_2_swebench_verified \
    --input benchmarks/swebench/data/swebench_verified_benchmark.jsonl \
    --output results/swebench_verified/rollouts.jsonl
```

## Multimodal with OpenCode sessions

Configure your model endpoint and sandbox provider, then prepare and run the taskset:

```bash
gym eval prepare --config benchmarks/swebench/multimodal/opencode.yaml

gym eval run --model-type vllm_model \
    --config benchmarks/swebench/multimodal/opencode.yaml \
    --split benchmark \
    --output results/swebench_multimodal/rollouts.jsonl
```

The recipe selects the prepared benchmark via `--split`; a flat `--input` file is
not the taskset entrypoint. It keeps the dev tasks, prompts, image bytes, verifier,
and three repeats per task. The OpenCode run timeout is 10,800 seconds, the episode
deadline is 21,600 seconds, and the resource sandbox TTL is 25,200 seconds. Issue
images are staged outside the task repository. Reference test images are supplied
only to the fresh verifier sandbox, after the agent closes.
