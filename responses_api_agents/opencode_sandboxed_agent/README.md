# OpenCode Sandboxed Agent
```bash
# In terminal 1
gym env start \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --config responses_api_agents/opencode_sandboxed_agent/configs/opencode_sandboxed_agent.yaml \
    --config resources_servers/swebench/configs/swebench.yaml

# In terminal 2
python responses_api_agents/opencode_sandboxed_agent/client.py \
    +benchmark_jsonl=benchmarks/swebench/data/swebench_verified_benchmark.jsonl
```

For E2E functional testing, run as above and remove the actual opencode run command from the exec.

## Turn constraints

Set `turn_constraint` in the agent configuration to cap policy-model requests and
send the model a reminder of its remaining budget:

```yaml
turn_constraint:
  enforcement: proxy
  limit: 31
  scope: session
  reminder:
    trigger: per_turn
    position: system_message
```

Each rollout has an independent budget. One turn is an attempted policy-model
HTTP POST, including retries, compaction, and subagent requests routed through the
same provider. Multiple tool calls from one response consume one turn. The first
`limit` requests reach the model; the next receives a non-retryable
`session_budget_exhausted` error. The sandbox's partial work is still verified.
The rollout records the requested constraint, observed attempt count, and whether
the budget was exhausted. An exhausted count includes the rejected request.

Reminders use `per_turn`, `threshold` (80% and 95% of the budget), or `auto`.
No additional turn is reserved for a final answer. Omit `turn_constraint` to keep
the existing behavior. Native `steps`/`maxSteps` limits and provider/model overrides
cannot be combined with the proxy constraint.

## Prefetch OpenCode binary and upload to S3
```bash
curl -L https://opencode.ai/install -o opencode_install.sh

APP=opencode
archive_ext=".tar.gz"
os=linux
arch=x64
target="$os-$arch"
requested_version=1.17.11
filename="$APP-$target$archive_ext"
url="https://github.com/anomalyco/opencode/releases/download/v${requested_version}/$filename"
curl -L $url -o $filename
tar -xzf "$filename" -C "./"

aws s3 cp opencode_install.sh /path/to/folder/opencode/install.sh

aws s3 cp opencode /path/to/folder/opencode/$APP-$target

# Double check they are uploaded properly.
aws s3 ls /path/to/folder/opencode/
```
