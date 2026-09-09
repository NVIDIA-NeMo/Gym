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

## Agent identity (`agent_user`)

By default (`agent_user: null`) the Terminus 2 tmux session and every agent command run as
the image default user, exactly as before. `agent_user` (an account name for `su`, or a uid;
the digit-only string `"1000"` is normalized to the int `1000`) lets the agent run
unprivileged while the verifier keeps running as the image default (root on the supported
images), which is what the Terminal-Bench `*-userroot` image derivatives are built for.

Lane-level override (applies to every row with every resources server):

```yaml
terminus_2_sandboxed_agent:
  responses_api_agents:
    terminus_2_sandboxed_agent:
      agent_user: agent
```

Per-row override: a dataset row may carry `agent_user`. The resources server echoes it from
`/seed_session` (`terminal_bench_2_1` today) and this agent reads it there; the row override is
honored only when the resources server echoes it, whereas the lane default applies with every
resources server. Precedence: row value if not `null`, otherwise the lane config. A row value
of `"root"` (or `0`) beats a lane `agent` and is the explicit per-row image-default escape
hatch: it skips the identity check, and the provider passes `"root"` through as the image
default (only the int `0` forces uid 0). For mixed pools leave the lane at `null` and set
`agent_user` per row. The effective identity is posted to `/verify` as `agent_user`.

Requirements and fail-closed check: a non-root `agent_user` requires a root-default image in
which that account exists. Before any Harbor setup the agent runs `id -u` as `"root"` (must
print `0`) and `id -u && id -g` as `agent_user` (must succeed and resolve to a non-root
uid/gid; for a uid it must match). Any failure raises `RuntimeError` naming the requested
identity and the observed output, the sandbox is stopped, and the rollout fails rather than
silently running as root. When `agent_user` is non-root the agent also runs `chmod 777
/logs/agent` (as the image default) so the su'd tmux `pipe-pane` can write its pane log;
`/logs/verifier` is never touched.

Scope and residual risk:

- `agent_user` is honored by `terminus_2_sandboxed_agent` and `opencode_sandboxed_agent`; other
  allowed agents run as the image default.
- The verifier always runs as the image default (root on the supported images), regardless of
  `agent_user`.
- Agent processes are not terminated before verification (service tasks need them alive while
  the tests run). Under a non-root `agent_user` the agent's processes (uid 1000) therefore stay
  alive during root verification and can read `/tests` (0755) and modify the workspace, but
  cannot write `/tests`, `/logs/verifier` or system paths.
- Containment is only as strong as the image's sudoers/setuid configuration; the
  `-userroot` derivatives are probed for `sudo -n` in the opt-in integration test
  (`tests/test_agent_user_integration.py`).

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
