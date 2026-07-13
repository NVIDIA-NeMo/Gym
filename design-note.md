# Design: Sandboxed Evaluation of NeMo Gym Agent Skills

## Summary

NeMo Gym is adding foundational Agent Skills that teach coding agents how to use Gym correctly. We need a reproducible benchmark that measures whether these skills improve agent output compared with the same agents operating without them.

This design proposes:

- a skill-agnostic coding-task benchmark;
- isolated agent and verifier sandboxes;
- deterministic checks as the primary quality signal;
- paired baseline and treatment runs;
- a benchmark-local experiment driver that records provenance and produces comparison reports;
- an initial vertical slice for `nemo-gym-create-environment` that can be extended to the remaining foundational skills.

The benchmark will use Gym's existing skill-loading, rollout, sandbox, and reward infrastructure rather than introducing a separate evaluation framework.

## Motivation

The foundational skills described in [#1235](https://github.com/NVIDIA-NeMo/Gym/issues/1235) are intended to improve coding agents at workflows such as:

- creating a Gym environment;
- integrating an external benchmark;
- configuring RL training;
- evaluating models and agents;
- implementing verification and scoring;
- using Gym's asynchronous and concurrency patterns;
- integrating other environment frameworks.

The issue proposes evaluating agents with and without skills loaded and comparing output quality across four dimensions:

1. correctness;
2. completeness;
3. convention compliance;
4. quality of guidance.

Prompt-only judging is insufficient for these workflows. Most outputs are code, configuration, tests, datasets, or runnable commands. The benchmark should execute those artifacts wherever possible and reserve LLM judging for dimensions that cannot be checked deterministically.

## Current State

### Canonical skill source

Agent skills now have one canonical source under:

```text
.agents/skills/<skill-name>/
```

Claude Code and legacy Codex discovery paths use compatibility links. Agent-specific metadata such as `agents/openai.yaml` remains colocated with the canonical skill.

### Runtime skill loading

NeMo Gym already supports skills as a run-level evaluation variable:

```bash
gym eval run ... +skills.path=<skill-bundle>
```

The current implementation:

- validates Agent Skills directories;
- computes a content hash;
- stamps `skills_ref` onto materialized inputs and rollout results;
- stages skills into a per-request Claude Code configuration directory;
- preserves skill provenance in output artifacts.

The source dataset remains skill-agnostic.

### Sandbox infrastructure

`nemo_gym.sandbox` already provides provider-neutral lifecycle and execution APIs:

- `Sandbox` and `AsyncSandbox`;
- `SandboxSpec`;
- command execution;
- file upload and download;
- Docker, Apptainer, OpenSandbox, Daytona, and ECS Fargate providers;
- cleanup and TTL support where available.

Mini-SWE and other coding environments provide examples of clean workspaces, patch capture, hidden checks, and sandbox cleanup.

### Missing pieces

The repository does not yet provide:

- a task benchmark designed specifically for skill evaluation;
- task-specific isolated workspaces for the Claude Code skill adapter;
- a reliable patch boundary between agent execution and verification;
- a generic hidden-check verifier for skill tasks;
- an A/B experiment driver;
- paired statistical comparison grouped by skill variant.

## Goals

1. Measure the causal effect of a skill while holding the model, agent, task, fixture, sampling configuration, and verifier constant.
2. Run every coding rollout in a clean, isolated workspace.
3. Verify agent changes in a separate clean environment containing hidden checks.
4. Prefer executable checks over subjective judging.
5. Preserve enough evidence to reproduce and diagnose every score.
6. Support individual-skill evaluation and full-bundle evaluation.
7. Keep the infrastructure reusable across Claude Code, Codex, OpenCode, and other coding agents.
8. Use existing Gym configuration, rollout, sandbox, and reward conventions.

## Non-goals

- Teaching ML fundamentals or general environment-design theory.
- Replacing existing model or benchmark evaluation workflows.
- Building an automatic skill optimizer in the first version.
- Supporting training token IDs or RL on these coding trajectories initially.
- Adding every coding-agent runtime in the first vertical slice.
- Treating one composite LLM-judge score as the source of truth.

## Proposed Architecture

```text
Experiment manifest
        |
        v
Skill evaluation driver
        |
        +---------------------------+
        |                           |
        v                           v
Discovery-control arm          Treatment arm
        |                           |
        v                           v
Fresh agent sandbox           Fresh agent sandbox
No target skill               Target skill staged
        |                           |
        +-------------+-------------+
                      |
                      v
               Captured Git patch
                      |
                      v
             Fresh verifier sandbox
             + hidden check suite
                      |
                      v
          Scores, diagnostics, artifacts
                      |
                      v
              Paired comparison report
```

The central design decision is to use separate agent and verifier sandboxes. The agent never receives hidden tests, and the verifier never trusts the agent's modified workspace.

## Repository Structure

```text
benchmarks/agent_skills/
├── README.md
├── config.yaml
├── configs/
│   ├── claude_code.yaml
│   └── experiments/
│       └── create_environment_v1.yaml
├── data/
│   ├── create_environment_development.jsonl
│   └── create_environment_validation.jsonl
├── fixtures/
│   └── create-environment/
│       ├── existing-data/
│       └── sql-resources-server/
└── scripts/
    ├── run_experiment.py
    └── compare_variants.py

responses_api_agents/claude_code_agent/
├── app.py
├── sandbox_runner.py
├── configs/
│   └── claude_code_agent_sandbox.yaml
└── tests/
    └── test_sandbox_runner.py

resources_servers/agent_skills/
├── app.py
├── verifier.py
├── checks/
│   └── create_environment.py
└── tests/
    └── test_app.py
```

The benchmark, verifier, and driver are generic to agent-skill evaluation. The first check suite and dataset focus on `nemo-gym-create-environment`.

## Sandboxed Agent Execution

### Configuration

Sandbox execution should be an opt-in mode of the existing Claude Code agent. Host execution remains the default.

An agent configuration will identify:

- sandbox provider;
- immutable image reference;
- workspace path;
- CPU and memory limits;
- timeout or TTL;
- runtime-specific skill discovery path.

Example:

```yaml
sandbox_provider: sandbox
sandbox_spec:
  image: nemo-gym-agent-skills@sha256:<digest>
  workdir: /workspace/nemo-gym
  ttl_s: 1800
  resources:
    cpu: 4
    memory_mib: 8192
```

### Clean workspace

The sandbox image contains:

- a pinned NeMo Gym revision;
- Python and project dependencies;
- Node and Claude Code;
- Git;
- no hidden verifier files.

The workspace used for evaluation must not contain the repository's built-in skill directories. Otherwise, the baseline can discover the treatment skill from the checkout. Before running the agent, the fixture must remove or exclude:

```text
.agents/skills/
.claude/skills/
.codex/skills/
```

Any other auto-discovered instructions must be identical between discovery-control and treatment arms.

### Agent lifecycle

For each rollout:

1. Start a new `AsyncSandbox`.
2. Initialize a clean workspace from the pinned fixture.
3. Apply task-specific setup files or a setup patch.
4. Create a fresh runtime configuration directory.
5. Stage the selected skill bundle when the arm requires it.
6. Run the coding-agent CLI inside the sandbox.
7. Capture stdout, stderr, exit status, usage, timing, and trajectory.
8. Capture all repository changes, including new files:

   ```bash
   git add -N .
   git diff --binary
   ```

9. Return the patch as a rollout artifact.
10. Stop the sandbox in `finally`.

The host checkout must remain unchanged.

### Runner result

The sandbox runner should return a structured result such as:

```python
class SandboxRunResult(BaseModel):
    stdout: str
    stderr: str
    return_code: int
    patch: str
    elapsed_seconds: float
```

Patch size limits and clear errors are required for malformed or oversized submissions.

## Sandboxed Verification

### Verifier lifecycle

For each agent patch:

1. Start a fresh sandbox from the same pinned image.
2. Restore the same task fixture and setup revision.
3. Apply the agent patch.
4. Upload the hidden check suite.
5. Execute deterministic checks.
6. Run a blind rubric judge only for unresolved subjective criteria.
7. Return reward, component scores, logs, and failure diagnostics.
8. Stop the verifier sandbox in `finally`.

The verifier must not reuse the agent sandbox. This prevents the agent from:

- modifying visible tests to make them pass;
- altering the test runner;
- persisting state that changes verification;
- hiding changes outside the captured patch.

### Hidden-check storage

Dataset rows should contain identifiers, not hidden test contents:

```json
{
  "verifier_metadata": {
    "fixture_id": "create-environment-sql",
    "check_suite_id": "create-environment-sql-v1"
  }
}
```

The resources server resolves those identifiers to server-side fixtures and checks.

## Task Schema

Example dataset row:

```json
{
  "task_id": "create-env-sql-001",
  "category": "create-environment",
  "task_type": "implementation",
  "responses_create_params": {
    "input": [
      {
        "role": "user",
        "content": "Create a NeMo Gym resources server for SQL generation using the supplied test database."
      }
    ]
  },
  "verifier_metadata": {
    "fixture_id": "create-environment-sql",
    "check_suite_id": "create-environment-sql-v1"
  }
}
```

The prompt should not name the target skill in the primary benchmark. This tests automatic discovery and activation from realistic user language. Explicit skill invocation can be measured separately.

## Initial `nemo-gym-create-environment` Tasks

### First vertical-slice task

The first real task should be:

> Create a resources server for SQL generation that executes queries against a supplied test database.

The fixture provides:

- a small SQLite database;
- example input rows;
- a pinned clean Gym checkout;
- task requirements;
- no environment implementation.

Hidden checks validate:

- expected resources-server structure;
- valid task JSONL;
- valid YAML configuration;
- correct agent, resources-server, and dataset references;
- correct `verify()` behavior for successful SQL;
- correct failure behavior for wrong, malformed, or empty output;
- safe handling of read-only queries;
- resources-server unit tests;
- example dataset collation;
- environment smoke tests;
- absence of known Gym anti-patterns.

### Additional task families

After the vertical slice works, add:

1. Existing-data environment with deterministic verification.
2. Environment requiring synthetic task generation guidance.
3. LLM-as-judge verifier selection and implementation.
4. Broken environment requiring diagnosis and repair.
5. Ambiguous requirements where the agent should ask or make a documented assumption.
6. Negative-control coding tasks unrelated to environment creation.

Maintain separate development and held-out validation splits.

## Deterministic Checks

The create-environment verifier should support:

- file and directory presence;
- Python imports;
- Pydantic and Gym configuration validation;
- `gym dataset collate` example validation;
- resources-server tests;
- verifier pass, fail, empty-output, and malformed-output cases;
- optional environment startup and smoke rollout;
- AST or static anti-pattern checks;
- timeout and cleanup validation.

Examples of convention checks include:

- no `httpx` in asynchronous Gym server paths;
- no `ray.get()` in async code;
- no unsafe required-key metadata access;
- async `/run` and verification paths where required;
- errors return structured failures rather than crashing;
- configuration flows through Gym config instead of undocumented environment variables.

## Scoring

Report dimensions independently:

| Metric | Meaning |
|---|---|
| `task_success` | All required deterministic checks pass |
| `correctness` | Functional tests and expected behavior |
| `completeness` | Required artifacts and workflow coverage |
| `convention_compliance` | Gym conventions and absence of anti-patterns |
| `guidance_quality` | Quality of decisions or explanation where deterministic checks are insufficient |
| `negative_control_success` | Performance on tasks that should not activate the skill |
| `elapsed_seconds` | End-to-end rollout time |
| `token_usage` | Model usage reported by the agent |
| `turns_used` | Agent interaction count |

`task_success` is the primary metric. Correctness is a hard gate. A high rubric score cannot compensate for broken code.

A composite score may be included for convenience, but all component scores must remain visible.

## A/B Experiment Design

### Arms

Use three arms:

1. **Bare baseline**
   - native discovery disabled;
   - no target skill.

2. **Discovery control**
   - native discovery enabled;
   - clean workspace;
   - no target skill.

3. **Treatment**
   - native discovery enabled;
   - same clean workspace;
   - target skill staged.

The primary causal comparison is treatment minus discovery control. Bare baseline minus discovery control measures the effect of enabling the runtime's discovery mode.

### Skill bundles

`skills.path` points to a directory containing one or more skill directories. It does not point directly to a single `SKILL.md` directory.

The driver snapshots canonical skills into an experiment-local bundle:

```text
results/<experiment-id>/skill-bundles/treatment/
└── nemo-gym-create-environment/
    ├── SKILL.md
    └── references/
```

This prevents the evaluated bytes from changing during a run and gives the experiment an immutable content hash.

### Controlled variables

All arms use the same:

- dataset and task ordering;
- task fixture revision;
- sandbox image digest;
- model and exact model version;
- coding-agent version;
- system prompt;
- tool permissions;
- sampling parameters;
- output-token limit;
- rollout count;
- verifier and hidden checks;
- resource limits.

Arm order should be randomized. Sampling seeds should be recorded and reused where supported.

## Experiment Driver

The first implementation should be benchmark-local:

```text
benchmarks/agent_skills/scripts/run_experiment.py
```

Do not add a generic `gym eval compare` command until the benchmark-local design has been validated.

### Manifest

Example:

```yaml
name: create-environment-v1

dataset:
  path: benchmarks/agent_skills/data/create_environment_validation.jsonl
  split: validation

agent:
  name: agent_skills_claude_code
  config: benchmarks/agent_skills/configs/claude_code.yaml
  sandbox_image: nemo-gym-agent-skills@sha256:<digest>

model:
  name: claude-sonnet-4-6
  temperature: 0.2
  max_output_tokens: 16384

sampling:
  repeats: 3
  seed: 1234
  concurrency: 4

arms:
  discovery_control:
    bare: false
    skills: null

  treatment:
    bare: false
    skills:
      - .agents/skills/nemo-gym-create-environment

  bare_baseline:
    bare: true
    skills: null

metrics:
  primary: task_success
  secondary:
    - correctness
    - completeness
    - convention_compliance
    - guidance_quality
    - elapsed_seconds
    - token_usage
```

### Driver lifecycle

1. Validate the manifest.
2. Resolve and hash the dataset, fixture, image, model, agent, verifier, and skills.
3. Materialize immutable skill bundles.
4. Write an experiment lock file.
5. Start the configured Gym environment.
6. Execute each arm with identical rollout settings.
7. Preserve raw rollouts, materialized inputs, failures, logs, and aggregate metrics.
8. Align results by `(task_id, rollout_index)`.
9. Compute paired metrics and confidence intervals.
10. Write machine-readable and human-readable reports.

### Outputs

```text
results/create-environment-v1/
├── experiment.lock.json
├── skill-bundles/
├── bare_baseline/
│   ├── rollouts.jsonl
│   └── aggregate_metrics.json
├── discovery_control/
│   ├── rollouts.jsonl
│   └── aggregate_metrics.json
├── treatment/
│   ├── rollouts.jsonl
│   └── aggregate_metrics.json
├── comparison.json
└── report.md
```

## Statistical Comparison

Align paired outcomes by task and rollout index.

Report:

- success rate by arm;
- paired treatment delta;
- win, loss, and tie counts;
- per-category deltas;
- pass@k;
- negative-control regression rate;
- token and latency deltas;
- paired confidence intervals.

Bootstrap by task rather than by rollout. Multiple rollouts for the same task are not independent tasks.

For binary task success, paired methods such as McNemar's test can be included once the validation set is large enough. Effect size and confidence intervals are more important than a standalone p-value.

## Skill Acceptance Criteria

A skill is ready to ship when:

- treatment improves task success over discovery control;
- correctness does not regress;
- gains occur across multiple tasks rather than one outlier;
- convention compliance improves or remains stable;
- negative-control performance remains within an agreed regression threshold;
- latency and token costs remain acceptable;
- observed gains reproduce on the held-out validation split;
- failure analysis does not reveal verifier leakage or sandbox contamination.

Exact numerical thresholds should be set after the development benchmark establishes baseline variance.

## Security and Reliability

Sandboxed agents execute generated code and must be treated as untrusted.

Requirements:

- no host repository bind mount with write access;
- separate agent and verifier sandboxes;
- immutable base image references;
- bounded CPU, memory, process count, timeout, and patch size;
- cleanup in `finally`;
- no hidden tests in agent-visible files;
- secrets passed only where required;
- restricted network access where practical;
- dependencies preinstalled in the image;
- no reuse of workspaces across rollouts.

A CLI agent inside the sandbox needs access to its model endpoint. Initial Docker development may use bridge networking, but production evaluation should restrict egress to required endpoints or use an authenticated model proxy.

## Implementation Plan

### Phase 0: Canonical skill source

Status: complete.

- Store skills under `.agents/skills/`.
- Preserve agent-specific compatibility paths.
- Keep agent-specific metadata in canonical skill directories.

### Phase 1: Sandboxed Claude Code runner

- Add optional sandbox configuration to the Claude Code agent.
- Implement `sandbox_runner.py`.
- Stage skills into the sandbox runtime configuration.
- Run Claude Code inside the sandbox.
- Capture a complete Git patch.
- Guarantee cleanup on success, failure, and timeout.
- Add mocked lifecycle tests.
- Add one Docker smoke test.

Acceptance criteria:

- a trivial agent task creates a file in the patch;
- the host checkout remains unchanged;
- the patch applies to a fresh checkout;
- skills can be enabled and disabled;
- no sandbox survives completion.

### Phase 2: Generic skill verifier

- Add `resources_servers/agent_skills/`.
- Provision a fresh verifier sandbox.
- Apply the captured patch.
- Resolve hidden checks by identifier.
- Return component scores and diagnostics.
- Test pass, failure, malformed patch, timeout, and cleanup paths.

Acceptance criteria:

- verifier results depend only on the fixture, submitted patch, and hidden check version;
- agent-side test modification cannot affect verification;
- failures are actionable and serializable.

### Phase 3: First create-environment benchmark

- Add development and validation datasets.
- Add the SQL resources-server fixture.
- Implement deterministic create-environment checks.
- Author `nemo-gym-create-environment`.
- Run smoke evaluations for discovery control and treatment.

Acceptance criteria:

- both arms run over identical materialized tasks;
- scores include all required dimensions;
- treatment rollouts carry the expected `skills_ref`;
- raw artifacts are sufficient to reproduce every score.

### Phase 4: A/B driver and reporting

- Add the experiment manifest schema.
- Snapshot skill bundles.
- Lock all experiment inputs.
- Execute arms in randomized order.
- Compute paired comparisons.
- Produce `comparison.json` and `report.md`.

Acceptance criteria:

- rerunning a locked experiment uses identical inputs;
- reports identify task-level wins, losses, and ties;
- comparison does not mix skill hashes or agent configurations.

### Phase 5: Expand coverage

- Add more create-environment scenarios.
- Add negative controls.
- Add Codex or another coding-agent runtime.
- Add the remaining foundational skills.
- Promote reusable comparison functionality into the public CLI if warranted.

## Proposed Work Items

1. Sandboxed Claude Code execution and patch capture.
2. Fresh-sandbox skill verifier and hidden-check registry.
3. SQL create-environment fixture and checks.
4. `nemo-gym-create-environment` skill.
5. Experiment manifest and driver.
6. Paired comparison and report generation.
7. Held-out validation task set.
8. Additional coding-agent adapter.

## Open Questions

1. Should sandbox execution remain an option on `claude_code_agent`, or become a separate generic `sandboxed_cli_agent` after the first runtime?
2. Where should immutable sandbox images be built and published?
3. Should hidden fixtures live in this repository, a separate private artifact, or both?
4. Which model endpoint and network-isolation strategy should CI use?
5. What task-success improvement and negative-control regression thresholds should block or approve a skill?
6. Should the final comparison functionality remain benchmark-local or become `gym eval compare`?
7. How many held-out tasks and repeats are needed after measuring initial variance?

## Recommended Starting Point

Begin with the smallest complete infrastructure boundary:

1. add a sandbox runner to the Claude Code agent;
2. run a trivial file-creation task inside Docker;
3. capture its patch;
4. apply the patch in a fresh verifier sandbox;
5. return a deterministic reward;
6. repeat with a trivial staged skill and without it.

This proves isolation, skill staging, artifact capture, verification, and A/B execution before the project invests in complex environment-generation tasks.
