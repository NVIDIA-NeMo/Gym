# Current NeMo Gym Benchmark Patterns

Use these patterns after choosing an integration profile in the parent skill. They explain relationships that are easy
to get wrong; they are not a second schema or a substitute for generated files.

Before copying an example, regenerate a disposable baseline with the current CLI and compare it with:

- `nemo_gym/environment/manifest.py` for authored manifest fields and profile requirements;
- `nemo_gym/environment/scaffold.py` for generated layout and extension points;
- `nemo_gym/environment/validation.py` for static validation and mirror checks;
- `nemo_gym/task_data.py` for row-schema ownership and legacy placement;
- `nemo_gym/verifier_fixture.py` for executable scorer cases; and
- `fern/versions/latest/pages/contribute/environments/new-environment.mdx` for the maintained workflow.

```bash
gym env init --help
gym env init --benchmark sample_benchmark --profile custom-gym-verifier
```

## Manifest and config have different authority

The manifest owns catalog identity and behavioral declarations. The Gym config owns runtime composition. The manifest
also contains a read-only mirror of that composition so validation and catalog tools can inspect it without running the
workload.

For a standard benchmark, the generated relationship is:

```yaml
# benchmarks/sample_benchmark/manifest.yaml — authored metadata plus mirrors
name: sample_benchmark
version: 0.1.0
kind: benchmark
integration_profile: custom-gym-verifier
domain: other
description: "..."
modality: text
licensing: Apache-2.0
authors: ["..."]
reward:
  range: [0.0, 1.0]
  higher_is_better: true
determinism: unknown
resources_server: sample_benchmark
agent_server: simple_agent
model_server: policy_model
datasets:
  - name: sample_benchmark
    type: benchmark
    jsonl_fpath: benchmarks/sample_benchmark/data/example.jsonl
    prepare_script: benchmarks/sample_benchmark/prepare.py
    prompt_config: benchmarks/sample_benchmark/prompt.yaml
    num_repeats: 1
canonical_split: test
standard_prompt_config: benchmarks/sample_benchmark/prompt.yaml
```

```yaml
# benchmarks/sample_benchmark/config.yaml — runtime authority
config_paths:
  - resources_servers/sample_benchmark/configs/sample_benchmark.yaml

sample_benchmark_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: sample_benchmark_resources_server
      model_server:
        type: responses_api_models
        name: policy_model

sample_benchmark_resources_server:
  resources_servers:
    sample_benchmark:
      datasets:
        - name: sample_benchmark
          type: benchmark
          jsonl_fpath: benchmarks/sample_benchmark/data/example.jsonl
          prepare_script: benchmarks/sample_benchmark/prepare.py
          prompt_config: benchmarks/sample_benchmark/prompt.yaml
          num_repeats: 1
```

Edit runtime wiring in the config first. Then run `gym env validate --sync sample_benchmark --kind benchmark` to update
only the manifest's composition mirrors after all static checks pass. Do not hand-maintain two competing compositions.
Authored fields such as reward semantics, provenance, lifecycle, and benchmark protocol remain manifest-owned and are
not overwritten by sync.

## Raw rows, prompts, and Responses input are distinct stages

A benchmark source or prepared row may be flat task data. It does not need to contain
`responses_create_params.input` before prompt rendering:

```json
{"task_id":"example-0001","question":"What is 6 x 7?","expected_answer":"42"}
```

```yaml
# benchmarks/sample_benchmark/prompt.yaml
user: |-
  Answer the question. Return only the final answer.

  {question}
```

Static validation and data preparation render the prompt into the Responses input envelope. Keep scorer-only fields
such as `expected_answer` out of the rendered prompt. Preserve stable task IDs and canonical splits across conversion.

A preparation module exposes `prepare(...) -> Path`, validates source rows, writes deterministic JSONL, and raises on
malformed source data instead of silently dropping it. The generated module is the starting pattern:

```python
def prepare(source: Path = SOURCE_PATH, output: Path = OUTPUT_PATH) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            row = json.loads(line)
            if not isinstance(row.get("question"), str):
                raise ValueError(f"invalid source row {line_number}")
            dst.write(json.dumps(row) + "\n")
    return output
```

Adapt this to the upstream source rather than adding network downloads, caches, or credentials unless the benchmark
actually requires them. Preserve the upstream revision and license in the workload metadata and README.

## `TaskData` describes task-owned fields

The resources server normally owns the row schema in `resources_servers/<server>/task_data.py`. A self-contained agent
with datasets and no resources-server reference owns its own schema instead.

Write the schema in the flat end-state shape. Framework-owned fields such as `responses_create_params` and `agent_ref`
do not belong in it:

```python
from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(json_schema_extra={"consumed_by": ["provenance"]})
    question: str = Field(json_schema_extra={"consumed_by": ["prompt"]})
    expected_answer: str = Field(json_schema_extra={"consumed_by": ["verify"]})
```

Required fields must match the server's request model, even if `verify()` does not currently read every field. Use
`extra="allow"` by default so validation can report undeclared inputs; use `extra="forbid"` only for a server that is
already intentionally fail-closed. Never rely on Pydantic's default `extra="ignore"`, which silently loses fields.

Legacy rows may still carry task fields inside `verifier_metadata`. Keep the schema flat. Add
`json_schema_extra={"legacy_location": "verifier_metadata"}` to a field only when the current wire reads that field
exclusively from the legacy wrapper. Do not add the marker when the request model accepts both placements.

Inspect the effective schema before preparing a large dataset:

```bash
gym env schema --resources-server <server>
```

## Keep the verifier independently executable

Separate scoring logic from FastAPI startup so the same implementation can be exercised by the verifier fixture:

```python
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.verifier_fixture import VerifierFixture


class SampleResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class SampleVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    expected_answer: str


class SampleVerifier:
    async def verify(self, body: SampleVerifyRequest) -> BaseVerifyResponse:
        reward = float(body.response.output_text.strip() == body.expected_answer.strip())
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


class SampleResourcesServer(SampleVerifier, SimpleResourcesServer):
    config: SampleResourcesServerConfig


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=SampleVerifier,
    request_model=SampleVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)
```

Preserve the request and response fields with `body.model_dump()` unless the authoritative response contract says
otherwise.

Do not copy the equality scorer when the benchmark uses partial credit, execution, state, or an LLM judge. Implement
the manifest's declared reward range and direction. A valid but wrong model answer should reach the worse endpoint;
malformed task/request data or unavailable grading infrastructure should follow an explicit error contract rather than
being silently scored as an ordinary wrong answer.

## Verifier fixtures pin the scorer contract

The resources server owns `VERIFIER_FIXTURE` and its JSONL cases. Start from the generated full Responses object rather
than inventing a shortened payload that the request model may not accept.

Every fixture needs these case kinds:

- `full_reward`: reaches the better endpoint;
- `zero_reward`: reaches the worse endpoint; despite the historical name, that value need not be numeric zero; and
- `malformed`: fails with the declared error substring and has no expected reward.

When manifest determinism is `seeded`, also add a `determinism` case and a fixture `reseed` adapter. It must reproduce
the reward on fresh server instances after explicit reseeding.

The test passes the same reward contract declared by the manifest:

```python
def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE,
            reward_range=(0.0, 1.0),
            higher_is_better=True,
            determinism="unknown",
        )
    )
```

Add focused tests beyond the fixture for meaningful scoring boundaries, extraction behavior, tool/subprocess failures,
timeouts, state isolation, and any workload-specific `grading_mode` override.

## Reuse a scorer without copying it

Search before adding a verifier. A reused resources server must expose exactly one inspectable `VERIFIER_FIXTURE`; its
config may bundle at most one `simple_agent` and may not bundle a model server.

```bash
gym search resources-servers "exact answer scoring"
gym env init --benchmark sample_benchmark --profile custom-gym-verifier \
  --reuse-verifier existing_scorer --reward-range 0 1 --higher-is-better
```

The reward flags declare the reused scorer's contract. They are accepted only with `--reuse-verifier`. For a new scorer
with non-default semantics, generate it first and then update the manifest, fixture cases, and fixture test together.

Scorer reuse does not transfer benchmark provenance, prompt fidelity, canonical splits, or upstream parity evidence;
the new workload still owns those.

## Non-default profiles change ownership

Choose a profile because it reflects who owns the measured behavior, not because a nearby benchmark happens to use it:

| Profile | Episode owner | Pattern to implement |
| --- | --- | --- |
| `custom-gym-verifier` | Gym's standard agent loop | Custom or reused resources-server scorer |
| `custom-gym-agent-loop` | Authored Gym agent | Replace the generated agent `responses()` TODO |
| `external-agent-loop` | External framework behind a Gym agent | Implement agent `run()` delegation, then make `responses()` raise `NotImplementedError` |
| `external-rollout-driver` | Configured rollout driver above the agent | Replace the generated rollout-driver delegation |

The profile is descriptive metadata; runtime behavior still comes from the resolved Gym config. Do not add a custom
agent just to duplicate `simple_agent`, and do not move an external episode into a resources-server verifier.

For model calls, session cookies, downstream server calls, trace context, subprocesses, and concurrency, follow the
current root `AGENTS.md`. Those cross-cutting contracts deliberately are not copied here.

## Validation and handoff pattern

Use each layer for what it proves:

```bash
# Inspect resolved runtime wiring.
gym env resolve --config benchmarks/sample_benchmark/config.yaml

# Validate schema, mirrors, prompt rendering, paths, and inferred profile.
gym env validate sample_benchmark --kind benchmark

# Execute the resources-server-owned verifier fixture.
gym env test sample_benchmark --kind benchmark

# Only after authored placeholders and required evidence are complete.
gym env publish sample_benchmark --kind benchmark
```

Static validation does not import components, execute preparation code, start services, call a model, or prove grading
quality. Run preparation tests, server tests, representative real smoke rollouts, and inspect verifier behavior as
required by `AGENTS.md`. For benchmarks, also perform reward profiling and inspect task-level failures and variance.
For a port, reproduce upstream metrics first and compare the same models in Gym.

Stop and investigate rather than declaring the integration complete when any of these remains unexplained:

- manifest mirrors disagree with the resolved config;
- raw-row fields conflict across flat, `task_data`, or legacy `verifier_metadata` placement;
- the inferred profile disagrees with the declared profile;
- a fixture endpoint or failure case does not match the manifest reward contract;
- infrastructure failures are indistinguishable from valid wrong answers; or
- a port has a material upstream-versus-Gym task-level or aggregate delta.
