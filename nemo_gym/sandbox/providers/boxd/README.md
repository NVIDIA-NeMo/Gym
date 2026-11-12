# boxd Sandbox Provider

The `boxd` provider runs each NeMo Gym sandbox on its own [boxd](https://boxd.sh?ref=Gym) machine
through the `boxd` Python SDK. With `SandboxSpec.image` set, the image runs as a container on the
machine's Docker daemon and commands run inside it. Without an image, commands run on the machine
(Ubuntu 24.04, user `boxd` with passwordless sudo).

## Setup

```bash
uv sync --extra boxd
export BOXD_API_KEY=<your-api-key>
```

For a package install, use `pip install "nemo-gym[boxd]"`. The provider requires `boxd>=0.3.1`.

The shipped config is `nemo_gym/sandbox/providers/boxd/configs/boxd.yaml`. It reads `BOXD_ORG`
and `BOXD_REGION` (`fra` or `nyc`), and leaves the API key to the SDK, which reads
`BOXD_API_KEY`. Add it beside the agent and model configs:

```bash
gym env start \
  --config responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml \
  --config nemo_gym/sandbox/providers/boxd/configs/boxd.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml
```

Machines are isolated by default, `stop()` deletes them, and `pause()`/`resume()` hibernate and
wake them with memory kept. boxd also deletes each machine when its lifetime ends (`ttl_s`, else
`create.max_lifetime_s`, one hour by default), so a crashed run leaves nothing behind. See `fern/versions/latest/pages/infrastructure/sandbox/boxd.mdx` for
every option, the resource mapping, timeouts and failure kinds.
