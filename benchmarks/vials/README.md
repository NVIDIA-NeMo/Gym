# VIALS: Visual Interpretation of Artifacts in the Life Sciences

VIALS contains 161 scientific-image questions and 174 image panels. Preparation
exports native Harbor task folders. Gym's `environment_servers/harbor` owns
sandbox setup, verification, and cleanup; the reusable `direct_vision_agent`
reads public question/image files and calls a Gym Model Server once, without
model tools. The separate verifier calls an independently configured judge.

This path uses Gym's execution runtime and does not depend on the Harbor Python
package, `harbor_agent_general`, or a VIALS resources server.

## Execution path

Harbor defines the task-folder format; Gym executes the tasks. The components
used by the evaluation command below are:

| Component | Responsibility |
| --- | --- |
| `gym eval run <task-folder>` | Loads the prepared dataset and wires the Harbor environment server, selected agent server, model server, and sandbox provider. |
| [Harbor environment server](../../environment_servers/harbor/README.md) | Owns each episode: creates the task sandbox, lends access to the agent, runs the separate verifier, collects results, and cleans up. |
| [Direct vision agent](../../responses_api_agents/direct_vision_agent/README.md) | Reads the public prompts and ordered images, makes one policy call through Gym's model server, and writes `/task/answer.txt`. It exposes no model tools and does not own the sandbox. |
| `tests/test.sh` and `tests/grade.py` in each task | Run in the separate verifier sandbox with private references and judge credentials, call the configured judge, and write the reward. |

`--agent direct_vision_agent` selects the policy harness; `--model-type openai_model`
selects its model transport, including for a vLLM-served policy. The task-folder
CLI selects `environment_servers/harbor`. The benchmark's `config.yaml` supplies
judge secrets to that environment server; it does not select the deprecated
`responses_api_agents/harbor_agent_general` bridge.

The earlier bridge delegated execution to Harbor's Python runtime and a
VIALS-specific Harbor agent. This implementation replaces that path. Use the
task-folder commands below, without a benchmark manifest or a `gym.jsonl` index.

## Install and prepare

Use Gym's development setup with Python 3.13.14+ and its supported uv version.
The worker needs an existing, configured Gym sandbox provider. Follow Gym's
[Docker provider setup](../../fern/versions/latest/pages/infrastructure/sandbox/docker.mdx)
or [OpenSandbox provider setup](../../fern/versions/latest/pages/infrastructure/sandbox/opensandbox.mdx).
VIALS supplies task Dockerfiles and verifier files; Gym's existing provider owns
container creation, file transfer, execution, and cleanup. The benchmark does
not install a container engine or override provider settings.

The following example uses Gym's stock Docker provider with a working Docker
daemon and Docker CLI. Check that service before preparing or evaluating tasks:

```bash
uv sync --extra dev --extra sandbox
source .venv/bin/activate

docker info
docker build -t vials-inputs:local \
    -f benchmarks/vials/harbor/Dockerfile benchmarks/vials/harbor
python -m benchmarks.vials.prepare
```

Preparation downloads `handshake-ai-research/VIALS`, configuration `tasks`, split
`test`, revision `9657bf84c6f6e470e1c31123afe3f9152c3655bd`, without filtering or
S3 credentials. It writes `benchmarks/vials/data/vials/`, including `dataset.toml`,
161 tasks, private reference solutions, and image attribution. Generated tasks
are excluded from Git. Set `VIALS_HARBOR_IMAGE` to use another shared base image.
For an independent export or a small fixture:

```bash
python -m benchmarks.vials.harbor.adapter --docker-image vials-inputs:local \
    --output-dir /path/to/vials --limit 5
```

Each task contains public instructions, prompts, ordered image panels, a task
Dockerfile, private verifier inputs, and a reference solution. References and
numeric bounds are never sent to the policy. Only the oracle receives
`solution/`; ordinary policy sessions receive the public environment files.

## Configure endpoints

Merge `env.example.yaml` into your local root `env.yaml`. Supply endpoint URLs,
model names, and credentials through the named environment variables. The default
judge is GPT-5-mini; NVIDIA's gateway uses `https://inference-api.nvidia.com/v1`
and `openai/openai/gpt-5-mini`. Policy and judge may use different providers.

The policy endpoint must accept image inputs through Chat Completions. Gym's
`openai_model` transports those requests; it does not launch policy weights.
The model can be served by vLLM on another reachable host. Judge credentials
are passed through the Harbor environment server's verifier secrets, not into
the policy sandbox or model input.

## Validate and run

Task-folder datasets use the native folder commands rather than a legacy
benchmark manifest or compatibility JSONL index:

```bash
# Exercise reference answers through the separate verifier first.
gym dataset validate benchmarks/vials/data/vials --sandbox docker \
    --config benchmarks/vials/config.yaml --limit 5

# Real policy evaluation: 161 questions, eight independent attempts each.
gym eval run benchmarks/vials/data/vials --agent direct_vision_agent \
    --sandbox docker --model-type openai_model \
    --config benchmarks/vials/config.yaml \
    --num-repeats 8 --concurrency 4 --output results/vials/rollouts.jsonl
```

The native loader builds and records task and verifier images. Run these commands
on the container-capable worker. On ARM workers, select
`++builds.platform=linux/arm64`; the default build target is AMD64. Other sandbox
providers may need a registry accessible from their workers. Use the selected
provider's documented deployment and image configuration; VIALS requires no
backend changes. A Slurm allocation alone does not provide a configured sandbox
service.

Use `--limit 5 --num-repeats 1` for a policy smoke. Increase the output limit with:

```bash
++direct_vision_agent.responses_api_agents.direct_vision_agent.max_output_tokens=131072
```

Use `262144` for the 256k output budget, subject to model context limits. For
Nemotron/Qwen thinking, append:

```bash
++direct_vision_agent.responses_api_agents.direct_vision_agent.chat_template_kwargs.enable_thinking=true
```

For DeepSeek use `chat_template_kwargs.thinking=true` and
`chat_template_kwargs.reasoning_effort=high` at the same agent configuration path.
Defaults are 32,768 generated tokens, temperature 1, top-p 1, a 14,400-second
agent timeout, and a 600-second verifier timeout. Output limits include reasoning.

No shell, Python, OCR, MCP, or retrieval tools are exposed to the policy. Image
reading and answer writing are harness file transfers. The native loader
currently refuses task-environment `no-network`, so tasks declare public
networking. This is not an enforced network-isolation claim: the policy has no
tool through which to use that network. The separate verifier needs judge access.

## Scoring and evidence

The policy's visible answer must end in `ANSWER: <final answer>`. The standalone
verifier retains upstream extraction, judging prompts, and inclusive numeric
bounds. An unverifiable text verdict triggers grading with the original image
panels. Missing final answers score zero. Exhausted judge transport failures
make `test.sh` fail; the native environment server records the verifier failure.
It must be resolved before reporting complete benchmark accuracy.

Report pass@1 accuracy averaged across eight complete repeats, with the sample
standard deviation of the eight accuracy percentages. This is not pass@8. Keep
all 161 questions, including output-truncated answers, in each denominator.
Audit failure diagnostics and resolve infrastructure/judge failures before
publishing scores. Record the model revision, thinking settings, and output cap.

The Gym response includes prompt text, ordered image paths, reasoning, visible
answer, and policy token usage. Complete image-bearing requests and raw responses
are saved as `policy-input.json` and `policy-response.json` under `/logs/artifacts`,
which the native environment server collects alongside verifier logs and ATIF.
The native ATIF exporter supports text only, so the transcript references the
input artifact and session close records an `input_media_in_artifact` gap.
Enable Gym model-call capture for additional policy-call evidence. The direct
vision agent currently reports an observation-capture gap at session close;
this integration claims Evaluation capability only. The dataset has
`verified = false` in `dataset.toml`.

The earlier eight-repeat results used the deprecated Harbor runtime bridge.
They do not by themselves validate this native environment-server migration.

## Contributor checks

```bash
pytest benchmarks/vials/tests responses_api_agents/direct_vision_agent/tests
pre-commit run --all-files
```

Also run the native oracle and real policy smoke above and inspect the returned
answer, judge reward, failure metadata, model input, and token usage. Unit tests
use sandbox and model fakes; they do not establish live provider compatibility.

## Sources and licenses

- Upstream code: `002a0d25031f02a760da5d14f2c29bbd26eb06e4` (MIT).
- Dataset: `9657bf84c6f6e470e1c31123afe3f9152c3655bd` (CC-BY-4.0).
- Judge prompts/parser retain upstream notices in the verifier's `upstream.py`.
- See `UPSTREAM_LICENSE`, root `ATTRIBUTIONS.md`, and `attributions.csv`.

Evaluation only: upstream specifies exclusion from training data.
