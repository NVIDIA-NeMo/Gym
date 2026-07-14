# Agent Skills Benchmark

This benchmark measures whether a coding-agent skill improves task completion while holding the model, agent, fixture, and hidden verifier constant.

The first task evaluates `nemo-gym-create-environment`: the agent must create a native SQL-generation resources server from existing task data.

## Build the fixture image

Build from the Gym repository root and pin the Gym revision used by both arms. The revision must be reachable from `NVIDIA-NeMo/Gym` (push the branch before building, or override `GYM_REPOSITORY`):

```bash
GYM_REVISION="$(git rev-parse HEAD)"
IMAGE="nemo-gym-agent-skills:create-environment-${GYM_REVISION:0:12}"

docker build \
  --file benchmarks/agent_skills/fixtures/create-environment-sql/Dockerfile \
  --build-arg GYM_REVISION="${GYM_REVISION}" \
  --build-arg CLAUDE_CODE_VERSION=2.1.207 \
  --build-arg UV_VERSION=0.10.4 \
  --tag "${IMAGE}" \
  .
```

The image contains a clean Gym checkout, a runnable `/task_inputs/shop.db`, and pinned Claude Code and uv versions, but removes project-local skill directories. The agent runner rejects dirty or skill-contaminated images. For a release run, also override `PYTHON_IMAGE` with an immutable image digest and record the resulting fixture image digest in the experiment manifest.

## Run the automated experiment

Update the sandbox image in:

```text
benchmarks/agent_skills/configs/experiments/create_environment_v1.yaml
```

Preview the locked commands and skill snapshot without running inference:

```bash
python benchmarks/agent_skills/scripts/run_experiment.py \
  --config benchmarks/agent_skills/configs/experiments/create_environment_v1.yaml \
  --output-dir results/agent-skills/create-environment-v1 \
  --dry-run
```

Use `.venv/bin/python` rather than `uv run python`: Ray server subprocesses use server-specific working directories, which are incompatible with inheriting the repository-root uv project hook.

Run all configured arms and generate the paired comparison:

```bash
.venv/bin/python benchmarks/agent_skills/scripts/run_experiment.py \
  --config benchmarks/agent_skills/configs/experiments/create_environment_v1.yaml \
  --output-dir results/agent-skills/create-environment-v1 \
  --resume
```

The dry run creates the immutable experiment lock. The real invocation uses `--resume` to execute that exact plan. The same flag safely continues interrupted experiments and forwards Gym's rollout-resume behavior for arms with partial output.

The driver refuses a dirty Git checkout by default, snapshots treatment skills, resolves the sandbox image to an immutable digest, verifies that the manifest dataset is bound by the benchmark config, randomizes arm order deterministically, invokes `gym eval run`, validates complete `skills_ref` provenance, and writes:

```text
results/agent-skills/create-environment-v1/
├── experiment.lock.json
├── skill-bundles/
├── bare_baseline/rollouts.jsonl
├── discovery_control/rollouts.jsonl
├── treatment/rollouts.jsonl
├── comparison.json
└── report.md
```

Use `--allow-dirty` only for development; the lock records that the source checkout was dirty.

## Manual arm commands

The commands below show the underlying public Gym interface used by the driver.

## Create an immutable treatment bundle

`skills.path` must point to a directory containing skill directories:

```bash
BUNDLE="$(pwd)/results/agent-skills/create-environment-bundle"
rm -rf "${BUNDLE}"
mkdir -p "${BUNDLE}"
cp -R .agents/skills/nemo-gym-create-environment "${BUNDLE}/"
```

## Run the discovery control

```bash
gym eval run \
  --benchmark agent_skills \
  --agent agent_skills_create_environment_claude_code_agent \
  --split benchmark \
  --output results/agent-skills/discovery-control.jsonl \
  --num-repeats 3 \
  --concurrency 4 \
  +agent_skills_sandbox_image="${IMAGE}"
```

The configured agent uses native discovery (`bare: false`) but receives no target skill.

## Run the treatment

```bash
gym eval run \
  --benchmark agent_skills \
  --agent agent_skills_create_environment_claude_code_agent \
  --split benchmark \
  --output results/agent-skills/treatment.jsonl \
  --num-repeats 3 \
  --concurrency 4 \
  +agent_skills_sandbox_image="${IMAGE}" \
  +skills.path="${BUNDLE}"
```

Both arms use the same task rows, sandbox image, resources server, hidden check suite, and sampling configuration. Treatment rollouts include the content-hashed `skills_ref`.

## Development smoke task

Start the development environment:

```bash
gym env start \
  --config resources_servers/agent_skills/configs/agent_skills.yaml \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
  +agent_skills_sandbox_image="${IMAGE}"
```

Then use the checked-in development row directly:

```bash
gym eval run \
  --no-serve \
  --agent agent_skills_claude_code_agent \
  --input benchmarks/agent_skills/data/create_environment_development.jsonl \
  --output results/agent-skills/development.jsonl \
  --num-repeats 1
```

## Current scoring

The hidden verifier reports:

- `task_success`;
- `correctness`;
- `completeness`;
- `convention_compliance`;
- verifier status and bounded diagnostics;
- token usage and verifier latency through aggregate metrics.
