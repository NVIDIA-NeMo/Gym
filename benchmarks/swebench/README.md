# SWE Bench

Adds multiple SWE benchmarks to Gym. These are useable across agents/harnesses.

## Details

- Data source: `princeton-nlp/SWE-bench_Verified`

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
