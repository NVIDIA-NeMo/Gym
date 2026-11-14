# VIALS: Visual Interpretation of Artifacts in the Life Sciences

VIALS contains 161 scientific image questions and 174 image panels. The default
integration exports Harbor tasks and executes them through Gym's shared
`harbor_agent_general` bridge. The VIALS vision agent sends the original question
and images to an OpenAI-compatible policy endpoint once, with no tools, and saves
the answer. Harbor runs the independent judge in a separate verifier container.
No VIALS Gym resources server or experiment audit proxy is used in this path.

## Install and prepare

Use Gym's development setup with Python 3.13.14+. A working Docker-compatible
engine and Docker Compose are needed on the execution host. Rootless Podman can
provide the Docker API, but its networking and health checks must work on that
host. An interactive Slurm allocation alone does not install a container engine.
The policy may run on another reachable host; the Harbor worker needs no GPU.

```bash
uv venv --python 3.13.14
source .venv/bin/activate
uv sync --extra dev
# Install the shared bridge's declared dependencies, including Harbor and Ray.
uv pip install -e responses_api_agents/harbor_agent_general

docker build -t vials-inputs:local \
    -f benchmarks/vials/harbor/Dockerfile benchmarks/vials/harbor

gym eval prepare --benchmark vials
gym env validate --manifest benchmarks/vials/manifest.yaml
```

Preparation downloads `handshake-ai-research/VIALS`, configuration `tasks`, split
`test`, at revision `9657bf84c6f6e470e1c31123afe3f9152c3655bd`. No S3 credentials
are required. It generates `data/harbor/` and a five-task example index. Set
`VIALS_HARBOR_IMAGE` before preparation to use a different shared image name.
The exporter can also be used independently:

```bash
python -m benchmarks.vials.harbor.adapter --docker-image vials-inputs:local \
    --output-dir /path/to/tasks --limit 5
```

Each task has instructions, a task configuration, policy-side images and a public
`vials_input.json`, plus private verifier code, references and trusted image
copies. The public manifest contains only original prompts and ordered image
filenames. References and numeric bounds never enter the policy input.

## Configure models

Copy [env.example.yaml](env.example.yaml) into your local root `env.yaml`, merging
with existing settings, and supply the actual URLs and served model IDs. Export
`VIALS_POLICY_API_KEY` and `VIALS_JUDGE_API_KEY`. Never commit credentials.
The policy endpoint must support direct image input and the requested output
limit plus the prompt within its context window. Gym does not serve weights here.
The default judge is `gpt-5-mini`. NVIDIA's gateway uses
`https://inference-api.nvidia.com/v1` and `openai/openai/gpt-5-mini`.

The Harbor worker calls the policy endpoint directly. Do not add `--model-type`;
this configuration does not start a Gym policy model server.

## Run without tools

```bash
gym eval run --benchmark vials --split benchmark \
    --num-repeats 8 --concurrency 4 --output results/vials/rollouts.jsonl \
    ++vials_max_output_tokens=32768
```

The single model request explicitly sets `tool_choice: none` and omits `tools`
(an empty array is rejected by some vLLM versions). No terminal, Python, OCR,
MCP, or tool loop is exposed. Reading images and saving `/task/answer.txt` are
harness file transfers, not model-selected tools. The agent writes a standard
ATIF `trajectory.json`; Gym's shared bridge returns its reward and usage.

Generation settings are forwarded through `harbor_agent.kwargs`. Use
`++vials_max_output_tokens=131072` or `262144` for the longer Nemotron budgets.
This includes reasoning tokens. Do not use Gym's `--max-output-tokens` flag for
this external agent: it does not control Harbor's agent configuration. Defaults
are temperature 1 and top-p 1; override `vials_temperature` and `vials_top_p` as
needed. For Nemotron/Qwen thinking, append:

```bash
++vials_chat_template_kwargs.enable_thinking=true
```

For DeepSeek, use the endpoint's native template settings:

```bash
++vials_chat_template_kwargs.thinking=true \
++vials_chat_template_kwargs.reasoning_effort=high
```

The task environment has no network; the host-side agent calls the policy API.
The separate verifier has public networking for judge calls. Policy and judge
credentials are installed in the Harbor worker environment; only judge variables
are declared in the verifier container configuration.

For five representative examples:

```bash
# Terminal 1
gym env start --benchmark vials

# Terminal 2
gym eval run --no-serve --benchmark vials --agent vials_agent \
    --input benchmarks/vials/data/harbor_example.jsonl \
    --output results/vials/examples.jsonl --num-repeats 1 --concurrency 1
```

On Slurm, request an interactive CPU allocation using your cluster's account and
partition, configure the container engine on the allocated node, and ensure both
endpoint URLs are reachable. Request GPUs only when also serving the policy.

## Scoring and evidence

The policy's visible answer must end in `ANSWER: <final answer>`. The standalone
verifier uses the original VIALS prompts, answer extraction and inclusive numeric
bounds. An unverifiable text verdict triggers grading with original image panels.
Missing final answers score zero; exhausted transport failures are trial errors.
Three received but unparseable verdicts score zero with a diagnostic flag.

Report pass@1 accuracy averaged over eight complete repeats, with the sample SD
of the eight accuracy percentages. This is not pass@8. Keep all 161 questions,
including output-truncated answers, in the denominator. Resolve infrastructure
and judge failures before reporting complete results. Record the model revision,
thinking settings and output cap.

Harbor artifacts live under `results/vials/harbor/`; rollouts are at the selected
output path. Inspect `agent/trajectory.json`, `agent/policy-response.json`, and
verifier `reward.json`/`info.json` to audit image inputs, finish reasons and grading.
The benchmark remains experimental (`experimental: true` in its manifest).
The current integration profile is `external-agent-loop`.

## Contributor checks

```bash
pytest benchmarks/vials/tests
gym env validate --manifest benchmarks/vials/manifest.yaml
pre-commit run --all-files
```

A real container/policy/judge smoke is required in addition to unit tests. Harbor
and container runtime requirements apply to live trials; optional Harbor unit
tests skip when Harbor is absent. VIALS uses only the shared Harbor integration; no benchmark-specific Gym resources
server or alternate direct-Gym configuration is required.

## Sources and licenses

- VIALS code: `002a0d25031f02a760da5d14f2c29bbd26eb06e4` (MIT).
- Dataset: `9657bf84c6f6e470e1c31123afe3f9152c3655bd` (CC-BY-4.0).
- Judge prompts/parser retain upstream notices in the verifier's `upstream.py`.
- See [UPSTREAM_LICENSE](UPSTREAM_LICENSE), [ATTRIBUTIONS.md](../../ATTRIBUTIONS.md)
  and [attributions.csv](attributions.csv) for code and image attribution.

Evaluation only: upstream specifies that VIALS must not appear in training data.
