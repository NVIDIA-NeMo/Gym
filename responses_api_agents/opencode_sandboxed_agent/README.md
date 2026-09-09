# OpenCode Sandboxed Agent

## Prerequisites

Complete [OpenSandbox access and setup](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox/opensandbox#setup)
before launching: obtain a service-operator-issued key (or deploy your own server),
configure a reachable endpoint, and confirm network access and resource limits.
The sandbox-service key is separate from model and image-registry credentials.

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

## OpenCode binary: online or pre-staged

Pre-staging is optional. With the shipped `remote_opencode_*` fields set to
`null`, the agent downloads the [upstream installer](https://opencode.ai/install)
inside each task sandbox and requests `opencode_version` (currently `1.17.11`).
The sandbox needs Bash, curl, archive-extraction tools, a writable home directory,
DNS/TLS access to `opencode.ai`, GitHub and its release-asset redirect hosts.
Ask the sandbox operator to approve the required egress; do not loosen the
benchmark's network policy merely to install a harness. Model connectivity and
task dependencies remain separate requirements.

### Pre-stage for restricted-egress sandboxes

The `remote_*` settings refer to **absolute paths already visible inside the
task sandbox**, not local paths, S3 URIs or instructions to mount a bucket.
Both `remote_opencode_install_script_path` and `remote_opencode_binary_path`
must be set; supplying only one still selects the online-download branch.

1. On an approved internet-connected staging host, download and review the
   installer and the release binary. This example targets Linux x86-64/glibc;
   choose the appropriate upstream asset for the sandbox's architecture, libc
   and CPU capabilities, not the staging host's. Retain the reviewed installer,
   source URLs and SHA-256 checksums in your artifact record.

   ```bash
   set -eu
   opencode_stage_dir=$(mktemp -d)
   cd "$opencode_stage_dir"
   opencode_version=1.17.11
   opencode_target=linux-x64
   curl -fL https://opencode.ai/install -o install.sh
   curl -fL \
     "https://github.com/anomalyco/opencode/releases/download/v${opencode_version}/opencode-${opencode_target}.tar.gz" \
     -o opencode.tar.gz
   tar -xzf opencode.tar.gz
   chmod +x opencode
   shasum -a 256 install.sh opencode
   ```

2. Make these exact files available in **every resources-server-owned task
   sandbox before the agent runs**. Use an operator-provisioned read-only mount,
   or include them in an approved task image without changing task contents.
   OpenSandbox accepts operator-supplied `volumes` through
   [`SandboxSpec.provider_options`](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox/opensandbox#sandboxspec-provider-options).
   For this SWE-bench recipe, sandbox creation options belong to
   `swebench_resources_server.resources_servers.swebench.sandbox_config.provider_options`,
   not just the agent's fallback sandbox config. The mount definition and
   backing-storage access are deployment-specific; obtain them from the operator.

   If your deployment uses S3-backed storage, the upload below is only the
   artifact-publication step. Replace the URI with your approved existing
   bucket/prefix. The staging identity needs `s3:PutObject` on that prefix and
   `s3:ListBucket` to list it; the mount/transfer identity needs `s3:GetObject`
   (and any required KMS permissions). Do not put AWS keys in agent config.

   ```bash
   opencode_asset_uri=s3://YOUR-APPROVED-BUCKET/gym-assets/opencode/1.17.11
   aws s3 cp install.sh "$opencode_asset_uri/install.sh"
   aws s3 cp opencode "$opencode_asset_uri/opencode-linux-x64"
   aws s3 ls "$opencode_asset_uri/"
   ```

   An S3 upload does not create a mount. The operator must explicitly map those
   objects to, for example, `/opt/gym-assets/opencode/1.17.11/` in each task
   sandbox. Alternatively, custom resources-server setup can use the existing
   [file-upload API](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox#startup-files-and-file-transfer):
   `await sandbox.upload(local_path, remote_path)` on the **same** sandbox handle
   that will be given to the agent. These agents do not provide an automatic
   S3 download or pre-run upload hook; uploading to another sandbox has no effect.

3. Save the following override as `offline-assets.yaml`, adapting the paths to
   the agreed mount/image, and append `--config offline-assets.yaml` to the
   server-start command:

   ```yaml
   opencode_sandboxed_agent:
     responses_api_agents:
       opencode_sandboxed_agent:
         opencode_version: 1.17.11
         remote_opencode_install_script_path: /opt/gym-assets/opencode/1.17.11/install.sh
         remote_opencode_binary_path: /opt/gym-assets/opencode/1.17.11/opencode-linux-x64
         remote_opencode_musl_binary_path: null
   ```

   This invokes the staged installer with `--binary`; the staged binary itself
   determines the installed version, so `opencode_version` does not validate or
   replace it. For mixed glibc/musl task images, a non-null
   `remote_opencode_musl_binary_path` instead invokes the installer with
   `--glibc-binary` and `--musl-binary`. That requires an operator-supplied,
   reviewed compatible installer; the upstream installer above does not accept
   those flags. Do not enable this field with the upstream script.

4. Before a rollout, run the following **inside the task sandbox**, using the
   operator's exec tooling or `sandbox.exec(...)` on its existing handle. It
   checks the mounted files without allocating another sandbox:

   ```bash
   test -r /opt/gym-assets/opencode/1.17.11/install.sh
   test -r /opt/gym-assets/opencode/1.17.11/opencode-linux-x64
   bash -n /opt/gym-assets/opencode/1.17.11/install.sh
   /opt/gym-assets/opencode/1.17.11/opencode-linux-x64 --version
   ```

   Expect version `1.17.11` and successful exits. After agent installation,
   check `"$HOME/.opencode/bin/opencode" --version` as the same sandbox user.
   A missing file means the mount/transfer is incomplete; permission denied
   can indicate a non-executable mount or lost execute bits; loader errors or
   illegal instructions indicate an incompatible binary. Fix the staging or
   image rather than silently falling back to downloads. These checks establish
   binary readiness only, not a successful agent rollout or verifier result.
