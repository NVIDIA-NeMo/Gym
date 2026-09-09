# Harbor Terminus 2 Agent

This agent runs Harbor's `Terminus2` control loop in the task sandbox supplied by
the NeMo Gym resources server. It adapts the small Harbor environment interface
that Terminus uses (`exec` and `is_dir`) to `AsyncSandbox`; task state therefore
remains owned by the resources server.

Before launching, complete [OpenSandbox access and setup](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox/opensandbox#setup)
for service credentials, a reachable endpoint, network requirements, and resource
limits. Sandbox-service, model, and image-registry credentials are separate.

```bash
gym env start \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --config responses_api_agents/terminus_2_sandboxed_agent/configs/terminus_2_sandboxed_agent.yaml \
    --config resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml \
    ++terminus_2_sandboxed_agent.responses_api_agents.terminus_2_sandboxed_agent.resources_server.name=terminal_bench_2_1_resources_server
```

To run one row from a benchmark JSONL after starting the servers:

```bash
gym eval prepare --config benchmarks/terminal_bench_2_1/terminus_2.yaml

python responses_api_agents/terminus_2_sandboxed_agent/client.py \
    +benchmark_jsonl=benchmarks/terminal_bench_2_1/data/benchmark.jsonl
```

The agent calls the configured model server exclusively through the Responses
API. Its returned response contains every model request and response from the
Terminus trajectory. Set `dump_trajectory: true` to also have Harbor write its
per-turn JSON trajectory files; it is `false` by default.

## Tmux binary: online or pre-staged

Pre-staging is optional. With `remote_tmux_binary_path: null`, the agent delegates
setup to [Harbor's tmux setup](https://github.com/laude-institute/harbor/blob/v0.22.0/src/harbor/agents/terminus_2/tmux_session.py).
If tmux is absent, that setup attempts package-manager installation and a source
build fallback. The task sandbox needs appropriate installation permissions,
package repositories and dependencies; the fallback also needs GitHub/release
asset egress. A usable tmux already on the sandbox's `PATH` avoids this download.
Agree any DNS/TLS/egress rules with the operator; do not relax a benchmark's
network restrictions just to install tmux. The online path does not pin tmux to
the pre-staged example version below.

### Pre-stage for restricted-egress sandboxes

1. On an approved internet-connected staging host, fetch an approved release
   matching the **task sandbox's** OS/architecture and runtime dependencies.
   This example is Linux x86-64, not a binary to run on a macOS staging host.
   Retain its source URL and SHA-256 checksum for reproducibility.

   ```bash
   set -eu
   tmux_stage_dir=$(mktemp -d)
   cd "$tmux_stage_dir"
   curl -fL \
     -o tmux-3.7c-linux-x86_64.tar.gz \
     https://github.com/tmux/tmux-builds/releases/download/v3.7c/tmux-3.7c-linux-x86_64.tar.gz
   tar -xzf tmux-3.7c-linux-x86_64.tar.gz
   chmod +x tmux
   shasum -a 256 tmux
   ```

2. Arrange an operator-provisioned read-only mount or an approved task-image
   build that exposes the binary at a fixed path in **every task sandbox**.
   For OpenSandbox, use the operator's deployment-specific
   [`provider_options.volumes`](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox/opensandbox#sandboxspec-provider-options)
   definition under
   `terminal_bench_2_1_resources_server.resources_servers.terminal_bench_2_1.sandbox_config.provider_options`:
   the resources server creates the sandbox and the agent reconnects to it.
   Setting only the agent's sandbox config will not add a mount to that existing
   sandbox. Do not replace benchmark task contents while adding the tool.

   For an S3-backed deployment, the following publishes the artifact to an
   **existing approved** bucket/prefix; it does not provision storage or mount it:

   ```bash
   tmux_asset_uri=s3://YOUR-APPROVED-BUCKET/gym-assets/tmux/3.7c
   aws s3 cp tmux "$tmux_asset_uri/tmux-3.7c-linux-x86_64"
   aws s3 ls "$tmux_asset_uri/"
   ```

   The uploader needs `s3:PutObject` on the prefix and `s3:ListBucket` for the
   listing. The mount/transfer identity needs `s3:GetObject` and any applicable
   KMS permissions. Use approved identity configuration, not keys in YAML.
   The operator must map that object to the example path below and preserve
   executable permissions. As an alternative for custom resources-server setup,
   [`await sandbox.upload(local_path, remote_path)`](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox#startup-files-and-file-transfer)
   transfers a local file to the same task sandbox before the agent runs. The
   agent has no S3 downloader or automatic pre-run upload hook; a file uploaded
   into a different sandbox is not shared.

3. Save this override as `offline-assets.yaml`, adapt the path to the actual
   mount/image, and append `--config offline-assets.yaml` to server startup:

   ```yaml
   terminus_2_sandboxed_agent:
     responses_api_agents:
       terminus_2_sandboxed_agent:
         remote_tmux_binary_path: /opt/gym-assets/tmux/3.7c/tmux-3.7c-linux-x86_64
   ```

   This is a sandbox filesystem path, **not** an `s3://` URI. The agent copies
   it to `/usr/local/bin/tmux`, sets execute permission and runs `tmux -V` before
   Harbor setup. The sandbox execution user therefore needs write access to
   `/usr/local/bin`. The command appends that directory to `PATH`; a pre-existing
   tmux earlier on `PATH` can still take precedence.

4. In the existing task sandbox, use the operator's exec tooling or
   `sandbox.exec(...)` to check the binary before rollout:

   ```bash
   test -r /opt/gym-assets/tmux/3.7c/tmux-3.7c-linux-x86_64
   /opt/gym-assets/tmux/3.7c/tmux-3.7c-linux-x86_64 -V
   ```

   Expect `tmux 3.7c` and successful exits. After installation, inspect both
   `/usr/local/bin/tmux -V` and `command -v tmux; tmux -V` as the agent's sandbox
   user to catch `PATH` shadowing. Missing files mean the mount/transfer is
   incomplete; permission or loader errors need an executable compatible mount
   or binary. This validates tool availability, not the full Terminal-Bench
   agent trajectory or verifier, and does not itself allocate a sandbox.
