# Multi-Harness Training Routing Plan

## Goal

Add a first-class rollout-routing mode that matches the FineEnvs multi-harness
training recipe:

- configure a pool of compatible agent harnesses for a task source;
- choose exactly one harness for each task;
- keep that harness fixed for every repeated rollout of the task; and
- mix the resulting harness-specific rollout groups while training one shared
  policy.

The unit of assignment remains one task group, not one model call. A rollout
does not switch harnesses after it starts.

## Existing support to reuse

NeMo Gym already provides most of the required machinery:

- `agent_map` routes matching rows to one explicitly named agent;
- `fan_out` runs every matching row with every listed agent;
- `num_repeats` creates the repeated rollouts used by GRPO-style training;
- preprocessing preserves one `_ng_task_index` across a task's repeats;
- materialized inputs make routing stable across resume;
- dispatch validates agent names and `allowed_agents` pairings;
- aggregate metrics are already separated by agent; and
- training-token capture can opt in every configured agent with
  `token_id_capture.all_agents: true`.

The missing behavior is choosing one agent from a configured pool both in
batch collection and in the direct `run_examples()` path used by NeMo RL and
VeRL, before the trainer's repeated generations are dispatched.

## Proposed interface

Add an `agent_pool` field to `RolloutCollectionConfig`, the merged Gym global
config, and the public `preprocess_examples()` API:

```yaml
agent_pool:
  shared_resources_server:
    - opencode_agent
    - claude_code_agent
    - codex_agent
    - mini_swe_agent
num_repeats: 8
```

For the first version, list order defines deterministic round-robin assignment:

```text
selected_agent = agent_pool[routing_key][source_task_index % len(agent_pool[routing_key])]
```

This deliberately matches the FineEnvs task-index assignment. Weighted or
random sampling can be considered separately after the deterministic contract
is established.

## Routing contract

1. Determine the existing routing key from `agent_ref.name` or `task_source`.
2. Resolve or preserve `_ng_task_index` before selecting an agent.
   Prefer the stable `_ng_agent_pool_index` stamped during dataset collation;
   trainers may replace `_ng_task_index` with a run-local admission index.
3. If `agent_pool` matches the routing key, select one agent using the task
   index and stamp it into `agent_ref`.
4. Apply `num_repeats` only after selection, so every repeat has the same
   agent.
5. Continue using the selected agent first and the original routing key second
   when resolving dict-form `num_repeats`.
6. Materialize the selected `agent_ref`; resumed runs therefore replay the
   exact assignment instead of selecting again.
7. Reuse the existing pre-dispatch agent-name and verifier-compatibility
   validation.
8. When NeMo RL or VeRL calls `run_examples()` directly, read `agent_pool`
   from the merged Gym config and apply it without expanding the row count.

`agent_pool` and `fan_out` must not both match the same routing key: the former
chooses one agent, while the latter intentionally runs all agents. Reject that
configuration before dispatch. `agent_map` remains available for unrelated
keys and as the fallback for rows not matched by `agent_pool`.

## Implementation steps

### 1. Configuration and validation

- Add `agent_pool: Optional[Dict[str, List[str]]]` to
  `RolloutCollectionConfig`.
- Reserve the top-level `agent_pool` key in the global config so a routing
  config can be included through `config_paths`.
- Reject empty pools, duplicate agents, and keys that overlap `fan_out`.
- Include pool destinations in existing configuration rename/alias handling so
  agent composition cannot leave stale routing names.

Expected files:

- `nemo_gym/rollout_collection.py`
- `nemo_gym/global_config.py`

### 2. Task-level selection

- Move task-index resolution early enough that routing can use it.
- Select the target once per source task, before the repeat loop.
- Preserve the existing `agent_ref` structure when the selected target is
  already present; otherwise stamp `{"name": selected_agent}`.
- Extend `preprocess_examples()` with the same option for trainer integrations
  that dispatch in-memory rows directly.
- Stamp `_ng_agent_pool_index` during `gym dataset collate`, keeping repeated
  copies of one source row on the same ordinal.
- Apply a run-wide pool automatically inside `run_examples()`, which is the
  interface NeMo RL and VeRL call during policy training.
- Use a stable content digest only as a compatibility fallback for older
  collated data that lacks both pool and task indices.

Expected file:

- `nemo_gym/rollout_collection.py`
- `nemo_gym/train_data_utils.py`

### 3. Training invariants

- Confirm all repeats for `(task_index, selected_agent)` remain consecutive by
  default and share one task index.
- Confirm `interleave_repeats` changes scheduling only, not assignment.
- Preserve unique rollout IDs and rollout indices so token capture continues to
  correlate every external-harness model call correctly.
- Do not change response reconstruction or the trainer-facing trajectory
  contract.

### 4. Tests

Add focused unit tests for:

- deterministic round-robin selection;
- all repeats of one task using one agent;
- caller-provided global task indices producing stable choices across chunks;
- collated source-task indices surviving repeats, shuffle, and restart;
- direct `run_examples()` routing from the merged training config;
- composition with integer and dict-form `num_repeats`;
- disjoint `agent_map`, `agent_pool`, and `fan_out` routes;
- rejection of empty, duplicate, and pool/fan-out-conflicting entries;
- early rejection of unknown agents and unsupported verifier pairings;
- public `preprocess_examples()` behavior without input mutation; and
- materialized-input/resume stability.

Expected files:

- `tests/unit_tests/test_rollout_collection.py`
- `tests/unit_tests/test_global_config.py`
- `tests/unit_tests/test_train_data_utils.py`

### 5. Documentation and example

- Document that multi-harness training means one selected harness per rollout
  group and several harnesses per training run.
- Add a policy-training example combining `agent_pool`,
  `grpo.num_generations_per_prompt`, and `token_id_capture.all_agents`, plus an
  evaluation example using `num_repeats`.
- Contrast `agent_pool` with `agent_map` and `fan_out`.
- State that GRPO advantages should be computed within the repeated
  task-and-harness group, not across different harnesses.

Expected files:

- `fern/versions/latest/pages/training-tutorials/external-agent-harnesses.mdx`
- `fern/versions/latest/pages/training-tutorials/nemo-rl-grpo/gym-configuration.mdx`

### 6. Companion NeMo RL sharded routing

Gym can select across every configured harness without trainer changes when a
single Gym actor hosts the pool. A sharded NeMo RL job needs one additional
step: select the agent before choosing the Gym actor, then send the complete
GRPO prompt group to the shard that hosts that agent.

- Read and validate `env.nemo_gym.agent_pool` during sharded actor setup.
- Mirror Gym's task-index and stable-content fallback selection exactly.
- Preserve `_ng_agent_pool_assignment` as the authoritative resume marker.
- Permit only a pooled resources-server route to be repeated beside its target
  agents; keep agent entries unique across shards.
- Validate all pool targets and dataset source routes before training.
- Strip `agent_pool` from each individual shard config so a shard does not try
  to resolve agents hosted by another actor.
- Leave unsharded jobs unchanged: forward the pool to Gym and let Gym select.

Expected companion repository files:

- `nemo_rl/environments/nemo_gym.py`
- `nemo_rl/environments/nemo_gym_shards.py`
- `nemo_rl/experience/rollouts.py`
- their focused unit tests and `docs/design-docs/nemo-gym-integration.md`

## Validation

Run, in order:

```bash
pytest tests/unit_tests/test_rollout_collection.py -x
pytest tests/unit_tests/test_global_config.py -x
pytest tests/unit_tests/test_train_data_utils.py -x
pre-commit run --files \
  nemo_gym/rollout_collection.py \
  nemo_gym/global_config.py \
  nemo_gym/train_data_utils.py \
  tests/unit_tests/test_rollout_collection.py \
  tests/unit_tests/test_global_config.py \
  tests/unit_tests/test_train_data_utils.py \
  fern/versions/latest/pages/training-tutorials/external-agent-harnesses.mdx \
  fern/versions/latest/pages/training-tutorials/nemo-rl-grpo/gym-configuration.mdx
```

In the companion NeMo RL checkout, run the focused shard and rollout-router
tests plus pre-commit on the touched files. These tests do not need a model or
GPU; a full training smoke does.

Before marking the PR ready, run a representative smoke with two compatible
harnesses, at least two tasks, and multiple repeats. Verify that:

- adjacent tasks select different harnesses in round-robin order;
- every repeat of a task stays on its selected harness;
- every unmasked external-harness rollout contains captured token IDs and log
  probabilities; and
- aggregate metrics report both agents independently.

If model compute or external harness credentials are unavailable, open the PR
as a draft and explicitly record the missing rollout evidence.

## Estimated and implemented scope

- Core routing and validation: 80-150 lines.
- Unit tests: 150-300 lines.
- Documentation and example: 60-120 lines.
- Implemented Gym branch: about 760 changed product/test/documentation lines,
  plus this plan.
- Implemented companion NeMo RL branch: about 550 changed lines including
  tests and documentation.
- Implementation plus focused tests: about 1-2 hours.
- Full checks, smoke evidence, review, and PR preparation: about 3-5 hours,
  depending on dependency installation and harness/model availability.

## Non-goals

- Switching harnesses during a single rollout.
- Training on multiple independent call chains from one rollout.
- Replacing the existing `fan_out` comparison/evaluation behavior.
- Changing token-capture storage or response reconstruction.
- Adding weighted or adaptive harness selection in the first PR.
- Changing trainer loss or advantage computation inside NeMo RL, VeRL, or
  another downstream framework. NeMo RL changes only pre-dispatch routing for
  sharded Gym actors.
