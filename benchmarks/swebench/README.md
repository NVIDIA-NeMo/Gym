# SWE Bench

Adds multiple SWE benchmarks to Gym. These are useable across agents/harnesses.

## Details

- Data source: `princeton-nlp/SWE-bench_Verified`
- SWE Bench Multimodal (`benchmarks/swebench/multimodal/opencode.yaml`): the dev split of
  `SWE-bench/SWE-bench_Multimodal`, 100 JavaScript tasks. The prepare script downloads each
  issue's images and attaches them to the prompt, and stores the test patch's reference images
  for the verifier, so it needs internet access; rollouts and verification do not. See
  [SWE Bench Multimodal grading](../../resources_servers/swebench/README.md#swe-bench-multimodal-grading).
  The test split is not prepared, because the pinned SWE-bench harness (swebench 4.1.0) has no
  environment specs for its repositories.

## Harnesses

Use the `opencode.yaml` recipe of Verified, Multilingual or Pro. It runs `opencode_agent` through
`single_agent_turn`, as `verified/hermes.yaml` runs Hermes: the Resources Server starts the task sandbox, lends it to
the agent session, and grades the captured patch after the agent closes. Rows reach it through the recipe's taskset
(`swebench:verified`, `swebench:multilingual` or `swebench:pro`).

The `opencode_legacy_agent.yaml` recipes of Verified, Multilingual and Pro are deprecated. They run
`opencode_sandboxed_agent` through the `legacy_agent` relay, and that agent
[will be removed](../../fern/versions/latest/pages/agent-server/integrate-existing-agents.mdx) after the Environment
Server migration. They stay to reproduce existing results, such as the
[Nemotron 3.5 Super](../nemotron_3.5_super/README.md) recipes. The two OpenCode adapters use different defaults, so
treat their scores as separate measurements.

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --config benchmarks/swebench/verified/opencode.yaml

# Start the servers, collect rollouts from the prepared benchmark split, and stop the servers
gym eval run --model-type vllm_model \
    --config benchmarks/swebench/verified/opencode.yaml \
    --split benchmark \
    --output results/swebench_verified/rollouts.jsonl
```

## SWE-bench Pro with Copilot

The public SWE-bench Pro pairing uses the `single_agent_turn` Environment Server
and a Resources-owned task sandbox. See the
[Copilot composition instructions](../../responses_api_agents/copilot_agent/README.md#configure-and-run)
and [SWE-bench Pro data preparation](../../resources_servers/swebench_pro/README.md).

| Harness | Config import | Agent Server | Resources tool grants |
| --- | --- | --- | --- |
| [Copilot](../../responses_api_agents/copilot_agent/README.md) | `responses_api_agents/copilot_agent/configs/copilot_agent.yaml` | `copilot_agent` | `[]` |

The Resources config declares Copilot in `allowed_agents`. Supply a reachable
Gym Model Server and sandbox provider, then use the `swebench_pro` taskset and
`swebench_pro_resources_server` binding. This is an evaluation pairing; token-ID
capture for training is not supported.
