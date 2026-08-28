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

## Running locally (no container)

Swap in the `local` sandbox provider to run OpenCode as a host subprocess:

```bash
gym env start --config environments/opencode_math/config.yaml
```

This suits resources servers that do not provision their own sandbox. A server that returns a
sandbox id from `/seed_session`, such as swebench, needs a provider that supports `connect`,
so pair those with a container provider instead.

Install is skipped when `opencode` is on PATH. For an external OpenAI-compatible endpoint, leave
`model_server` unset, put `model`/`provider` in `opencode_config`, and turn `token_id_capture` off.

No isolation here: OpenCode's bash tool has your full user access and picks up your personal
OpenCode config and skills. Use a container provider for untrusted tasks.

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
