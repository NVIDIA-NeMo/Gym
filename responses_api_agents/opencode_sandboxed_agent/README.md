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

## Agent identity (`agent_user`)

By default (`agent_user: null`) the in-sandbox opencode process runs as the image default
user, exactly as before. Because opencode IS the agent, its identity is the identity of every
tool call the model makes (the `bash`, `edit`, `write` and other tools run inside that
process), so on a root-default image the model runs as root. `agent_user` (an account name for
`su`, or a uid; the digit-only string `"1000"` is normalized to the int `1000`) lets the agent
run unprivileged while the verifier keeps running as the image default (root on the supported
images), which is what the Terminal-Bench `*-userroot` image derivatives are built for.

Lane-level override (applies to every row with every resources server):

```yaml
opencode_sandboxed_agent:
  responses_api_agents:
    opencode_sandboxed_agent:
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
which that account exists. Before the install the agent runs `id -u` as `"root"` (must print
`0`) and `id -u && id -g` as `agent_user` (must succeed and resolve to a non-root uid/gid; for
a uid it must match). Any failure raises `RuntimeError` naming the requested identity and the
observed output, the sandbox is stopped, and the rollout fails rather than silently running as
root.

What runs as `agent_user`: the install + `opencode run` command, the `opencode session list`
and `opencode export` commands, and the observation database snapshot. Under a non-root
identity the first three commands begin by resetting `$HOME` to the identity's passwd home
(the provider's uid path does not go through `su`, so it would otherwise inherit the image
default's `/root`), so opencode's install lands in `/home/<agent_user>/.opencode/bin` and
its session store in `/home/<agent_user>/.local/share/opencode` (or the per-rollout
`XDG_DATA_HOME` when observations are collected), which is why the list, export and snapshot
commands must use the same identity. The export download uses the provider's file API as the
image default (root can read agent-owned files).

Scope and residual risk:

- The verifier always runs as the image default (root on the supported images), regardless of
  `agent_user`.
- Agent processes are not terminated before verification (service tasks need them alive while
  the tests run). Under a non-root `agent_user` the opencode process and anything it spawned
  (uid 1000) therefore stay alive during root verification and can read `/tests` (0755) and
  modify the workspace, but cannot write `/tests`, `/logs/verifier` or system paths.
- Containment is only as strong as the image's sudoers/setuid configuration; a permissive
  `sudo -n` or a setuid helper in the image gives the model root back.

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
