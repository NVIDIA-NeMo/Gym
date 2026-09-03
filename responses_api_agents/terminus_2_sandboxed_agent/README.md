# Harbor Terminus 2 Agent

This agent runs Harbor's `Terminus2` control loop in the task sandbox supplied by
the NeMo Gym resources server. It adapts the small Harbor environment interface
that Terminus uses (`exec` and `is_dir`) to `AsyncSandbox`; task state therefore
remains owned by the resources server.

```bash
gym env start \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --config responses_api_agents/terminus_2_sandboxed_agent/configs/terminus_2_sandboxed_agent.yaml \
    --config resources_servers/swebench/configs/swebench.yaml
```

To run one row from a benchmark JSONL after starting the servers:

```bash
python responses_api_agents/terminus_2_sandboxed_agent/client.py \
    +benchmark_jsonl=benchmarks/swebench/data/swebench_verified_benchmark.jsonl
```

The agent calls the configured model server exclusively through the Responses
API. Its returned response contains every model request and response from the
Terminus trajectory. Set `dump_trajectory: true` to also have Harbor write its
per-turn JSON trajectory files; it is `false` by default.

## Download and mount Tmux binary
```bash
curl -fL \
  -o tmux-3.7c-linux-x86_64.tar.gz \
  https://github.com/tmux/tmux-builds/releases/download/v3.7c/tmux-3.7c-linux-x86_64.tar.gz
tar -xzf tmux-3.7c-linux-x86_64.tar.gz
mv tmux tmux-3.7c-linux-x86_64

# e.g. copy to mounted S3 bucket
aws s3 cp tmux-3.7c-linux-x86_64 \
  s3://tmux/tmux-3.7c-linux-x86_64
```
