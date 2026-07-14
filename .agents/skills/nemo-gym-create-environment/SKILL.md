---
name: nemo-gym-create-environment
description: >-
  Creates native NeMo Gym environments, benchmarks, and resources servers from
  scratch. Use when asked to create an environment, build a resources server,
  add a new native benchmark, define Gym tasks, implement verify or reward
  logic, wire agents and datasets in YAML, validate an environment, or baseline
  its rewards. Covers existing-data and task-generation paths. Do not use for
  wrapping a third-party benchmark harness that already owns orchestration and
  scoring.
---

# Create a NeMo Gym Environment

Use this workflow when the user knows the capability or task they want but
needs Gym-specific implementation.

Do not teach general ML fundamentals. Make the environment runnable,
verifiable, and reviewable using current repository conventions.

## First decision: where are the tasks?

Resolve this before scaffolding.

### Existing data

If tasks already exist:

1. Inspect their schema, license, split, and expected answers.
2. Convert them to Gym JSONL without discarding verifier inputs.
3. Put conversion scripts in the source dataset repository when one exists.
4. Commit only a small `data/example.jsonl`; publish train and validation data
   through the configured dataset registry.
5. Ensure every row contains `responses_create_params.input` and task-specific
   verification metadata.

### Tasks need to be created

If no tasks exist:

1. Define the capability and observable success condition.
2. Design tasks that vary difficulty and failure modes rather than paraphrasing
   one template.
3. Specify how correctness will be established before generating examples.
4. Create a small, manually reviewed example set first.
5. Use an existing synthetic-data pipeline only when the task requires scale.
   The calendar environment README and notebooks are useful examples.

Ask a focused question only when the missing data decision materially changes
the implementation. Otherwise state a reasonable default and proceed.

## Determine the environment shape

Most new native environments need:

- one resources server under `resources_servers/<name>/`;
- an existing agent such as `simple_agent`, unless the task is genuinely
  multi-step or tool-driven;
- a model server reference;
- example tasks;
- deterministic or judge-based verification;
- tests and documentation.

Read these repository references as needed:

- `fern/versions/latest/pages/contribute/environments/new-environment.mdx`
- `fern/versions/latest/pages/environment-tutorials/single-step-environment.mdx`
- `nemo_gym/base_resources_server.py`
- a nearby resources server with similar verification behavior

Do not copy an unrelated server merely because its directory layout is
convenient.

## Scaffold

For a new resources server, prefer:

```bash
gym env init --resources-server <name>
```

The finished directory should include:

```text
resources_servers/<name>/
├── app.py
├── configs/<name>.yaml
├── data/example.jsonl
├── tests/test_app.py
├── requirements.txt
└── README.md
```

Add `data/example_rollouts.jsonl` when preparing the contribution for review.

Keep server names, config instance names, dataset paths, and agent references
consistent.

## Define the task contract

Before implementing `verify()`, write down:

- what the model receives;
- what output is expected;
- which metadata the verifier needs;
- which malformed outputs are expected;
- whether verification has side effects;
- what earns reward 0 and reward 1;
- whether partial credit is meaningful.

Use typed Pydantic request and response models extending Gym's base types.
Optional metadata must be accessed safely. Required metadata should fail with
an actionable validation error rather than an incidental `KeyError`.

## Choose verification deliberately

Prefer the strongest reliable verifier available:

1. **Exact deterministic check** for normalized strings, structured values, or
   known answers.
2. **Execution-based check** for code, SQL, tools, and stateful tasks.
3. **Reference comparison** when semantic equivalence can be computed.
4. **LLM judge** only when correctness cannot be determined mechanically.

For execution-based verification:

- isolate untrusted execution;
- enforce timeouts and concurrency limits;
- make destructive operations impossible or disposable;
- verify side effects, not only process exit codes;
- return reward 0 for empty, malformed, timed-out, or unsafe output.

For an LLM judge:

- define the rubric and output labels explicitly;
- handle missing or malformed judge output;
- reduce positional bias where relevant;
- test obvious positive and negative cases;
- record judge details needed to diagnose disagreement.

## Implement the resources server

Use `SimpleResourcesServer` unless the task requires a more specialized base.

The normal verification shape is:

```python
class MyVerifyRequest(BaseVerifyRequest):
    verifier_metadata: dict


class MyVerifyResponse(BaseVerifyResponse):
    status: str


class MyResourcesServer(SimpleResourcesServer):
    async def verify(self, body: MyVerifyRequest) -> MyVerifyResponse:
        ...
```

Requirements:

- endpoint logic is asynchronous;
- empty or malformed model output returns a scored failure instead of crashing;
- task execution failures are represented in the response;
- reward is numeric and has a documented meaning;
- diagnostics explain why a rollout failed;
- concurrency is bounded around scarce subprocess, database, or judge
  resources.

### Async and HTTP rules

- Use Gym's `nemo_gym.server_utils.request()` or `ServerClient`.
- Do not introduce `httpx` in asynchronous Gym paths.
- Await Ray futures directly; do not call `ray.get()` from async code.
- Do not block the event loop with synchronous network or long-running
  subprocess operations.
- Decode external process output with `errors="replace"`.

### Configuration rules

- Pass configuration through Gym YAML and typed config models.
- Do not add undocumented environment-variable configuration.
- Include `domain`, `description`, and `value` where repository conventions
  require them.
- Reference resources servers, model servers, agents, and datasets by their
  configured instance names.
- Keep `verified: false` until baselining is complete.

## Prepare example data

Create at least five representative rows. Cover:

- normal success;
- plausible wrong output;
- malformed or empty output;
- an edge case;
- a case that exercises important verifier metadata.

Example shape:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "user", "content": "Task instructions"}
    ]
  },
  "verifier_metadata": {
    "expected": "value"
  }
}
```

Do not put secrets, private endpoints, or large datasets in committed example
files.

## Test the verifier, not only helpers

Tests should instantiate the resources server and call its public verification
method.

At minimum cover:

- correct output;
- incorrect output;
- empty output;
- malformed output;
- timeout or execution failure when applicable;
- unsafe or destructive output when applicable.

For external optional tools, skip tests clearly when the tool is unavailable.
If startup auto-installs a tool, ensure installation happens before test
collection evaluates skip conditions.

Run:

```bash
gym env validate --config resources_servers/<name>/configs/<name>.yaml
gym env test --resources-server <name>
pytest tests/unit_tests/ -x
```

Validate example data:

```bash
gym dataset collate \
  --config resources_servers/<name>/configs/<name>.yaml \
  --output-dir /tmp/<name>-example \
  --mode example_validation
```

## Smoke test the full environment

Start the configured resources server, agent, and model:

```bash
gym env start \
  --config resources_servers/<name>/configs/<name>.yaml \
  --model-type openai_model
```

Collect a small rollout:

```bash
gym eval run --no-serve \
  --agent <configured-agent-name> \
  --input resources_servers/<name>/data/example.jsonl \
  --output resources_servers/<name>/data/example_rollouts.jsonl \
  --num-repeats 1
```

Inspect the actual trajectories and verifier diagnostics. A successful command
is not evidence that reward logic is correct.

## Baseline verifier quality

Run repeated rollouts on a representative validation subset and profile
rewards:

```bash
gym eval profile \
  --inputs <materialized-inputs.jsonl> \
  --rollouts <rollouts.jsonl>
```

The baseline should demonstrate:

- non-trivial pass rates;
- expected ordering across reference models where applicable;
- stable verifier behavior across repeats;
- understandable failure categories;
- no infrastructure failures counted as model failures.

For a public benchmark, reproduce published results before and after
integration. For a new training environment, validate that the reward signal is
usable before beginning training.

## Completion checklist

- [ ] Data source or task-generation approach is documented.
- [ ] Task rows use Gym Responses input format.
- [ ] Verifier design matches the task.
- [ ] Empty and malformed outputs fail safely.
- [ ] Config links resources server, agent, model, and datasets correctly.
- [ ] Example data contains at least five representative tasks.
- [ ] Public verifier behavior has unit tests.
- [ ] Dataset collation succeeds.
- [ ] Environment smoke test succeeds.
- [ ] Reward profiling produces meaningful results.
- [ ] README documents setup, data, verification, license, and commands.
- [ ] No known async, HTTP, metadata, or configuration anti-patterns remain.
