# Multi-Harness Training Routing Plan

## Goal

Add a first-class rollout-routing mode that matches the FineEnvs multi-harness
training recipe:

- configure a pool of compatible agent harnesses for a task source;
- expand every source task once per configured harness;
- keep each expanded task-and-harness prompt group fixed across its repeated
  rollouts; and
- mix the resulting harness-specific rollout groups while training one shared
  policy.

For four source tasks and four harnesses, the training dataset therefore has 16
independent prompt groups. The unit of assignment remains one task-and-harness
group, not one model call. A rollout never switches harnesses after it starts.

NeMo Gym also supports `agent_pool` for the distinct choose-one-of-N behavior.
The two modes must remain explicit: `fan_out` creates a cross-product, while
`agent_pool` deterministically selects one destination.

## Existing support to reuse

NeMo Gym already provides most of the required machinery:

- `agent_map` routes matching rows to one explicitly named agent;
- `fan_out` runs every matching row with every listed agent in direct Gym
  collection;
- `num_repeats` creates the repeated rollouts used by GRPO-style training;
- preprocessing preserves one `_ng_task_index` across a task's repeats;
- materialized inputs make routing stable across resume;
- dispatch validates agent names and `allowed_agents` pairings;
- aggregate metrics are already separated by agent; and
- training-token capture can opt in every configured agent with
  `token_id_capture.all_agents: true`.

The missing training behavior is expanding before trainer batching and before
distributed Gym-shard selection. Otherwise NeMo RL can route only one copy of
a source task, or expand too late and split sibling generations across actors.
The integration must also preserve a neutral source route, validate all
destinations, and support the existing direct Gym collection path.

## Proposed interface

Use `fan_out` as the task-cross-product field in `RolloutCollectionConfig`, the
merged Gym global config, NeMo RL environment configuration, and the public
`preprocess_examples()` API:

```yaml
fan_out:
  anyterminal_multi_harness:
    - anyterminal_opencode
    - anyterminal_openclaw
    - anyterminal_pi
    - anyterminal_hermes
num_repeats: 8
```

List order defines deterministic expansion order:

```text
expanded_groups = [(source_task, harness) for harness in fan_out[routing_key]]
```

`agent_pool` remains available when a caller intentionally wants the earlier
round-robin choose-one behavior.

## Routing contract

1. Determine the existing routing key from `agent_ref.name` or `task_source`.
2. If `fan_out` matches the routing key, create one copy per configured agent,
   stamp that agent into `agent_ref`, and record a stable
   `_ng_fan_out_index`.
3. Apply `num_repeats` or GRPO sibling generation only after expansion, so
   every sibling in one prompt group uses the same harness.
4. Preserve the original source ordering and configured harness ordering.
5. Materialize the selected `agent_ref`; resumed runs therefore replay the
   exact assignment instead of expanding or selecting again.
6. Reuse the existing pre-dispatch agent-name and verifier-compatibility
   validation.
7. When NeMo RL owns batching, expand in `NemoGymDataset` before dataloader
   admission and shard routing. When Gym owns batching, expand in
   `preprocess_examples()`.

`agent_pool` and `fan_out` must not both match the same routing key: the former
chooses one agent, while the latter intentionally runs all agents. Reject that
configuration before dispatch. `agent_map` remains available for unrelated
keys and as the fallback for unmatched rows.

## Implementation steps

### 1. Configuration and validation

- Add `fan_out` and `agent_pool` routing fields to
  `RolloutCollectionConfig`.
- Reserve the routing keys in the global config so a routing
  config can be included through `config_paths`.
- Reject empty lists, duplicate agents, and routes configured in both modes.
- Include destinations in existing configuration rename/alias handling so
  agent composition cannot leave stale routing names.

Expected files:

- `nemo_gym/rollout_collection.py`
- `nemo_gym/global_config.py`

### 2. Task-level expansion and selection

- Move task-index resolution early enough that routing can use it.
- Expand every matched source task once per `fan_out` target before the repeat
  loop.
- Select one target once per source task for the separate `agent_pool` mode.
- Preserve the existing `agent_ref` structure when the selected target is
  already present; otherwise stamp `{"name": selected_agent}`.
- Extend `preprocess_examples()` with both options for trainer integrations
  that dispatch in-memory rows directly.
- Stamp `_ng_agent_pool_index` during `gym dataset collate`, keeping repeated
  copies of one source row on the same ordinal.
- Apply run-wide routing automatically inside `run_examples()`, which is the
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

- deterministic fan-out ordering and round-robin selection;
- one source row producing one independent group per configured harness;
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

- Document that fan-out means one fixed harness per expanded rollout group and
  every configured harness per source task.
- Add a policy-training example combining `fan_out`,
  `grpo.num_generations_per_prompt`, and `token_id_capture.all_agents`, plus an
  evaluation example using `num_repeats`.
- Document `agent_pool` separately as choose-one-of-N routing.
- Contrast `agent_pool` with `agent_map` and `fan_out`.
- State that GRPO advantages should be computed within the repeated
  task-and-harness group, not across different harnesses.

Expected files:

- `fern/versions/latest/pages/training-tutorials/external-agent-harnesses.mdx`
- `fern/versions/latest/pages/training-tutorials/nemo-rl-grpo/gym-configuration.mdx`

### 6. Companion NeMo RL sharded routing and pre-batch fan-out

Gym can expand or select across every configured harness without trainer
changes when it owns collection. A sharded NeMo RL job must perform the
cross-product before dataloader batching, then send each complete GRPO prompt
group to the shard that hosts its stamped agent.

- Read and validate `env.nemo_gym.fan_out` and `agent_pool` during actor setup.
- Expand `NemoGymDataset` rows before batching and stamp
  `_ng_fan_out_index` plus the destination `agent_ref`.
- Mirror Gym's task-index and stable-content fallback selection exactly.
- Preserve `_ng_fan_out_index`, stamped `agent_ref`, and
  `_ng_agent_pool_assignment` as authoritative routing metadata.
- Keep agent entries unique across shards while permitting the neutral source
  route to map to every hosted fan-out target.
- Validate all routing targets and dataset source routes before training.
- Consume trainer-side routing before Gym dispatch so individual shards receive
  stamped destinations and cannot expand the same row twice.
- Prevent double expansion by consuming trainer-side fan-out before Gym
  dispatch.

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

Before marking the PR ready, run a representative smoke with all four P0
harnesses, real Terminal-Bench tasks, and multiple GRPO sibling generations.
Verify that:

- every source task appears once under OpenCode, OpenClaw, Pi, and Hermes;
- every sibling generation stays on its task-and-harness assignment;
- every unmasked external-harness rollout contains captured token IDs and log
  probabilities; and
- aggregate metrics and TMPE are reported without masked or invalid groups.

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
- Random, weighted, or adaptive fan-out target sampling.
- Changing token-capture storage or response reconstruction.
- Adding weighted or adaptive harness selection in the first PR.
- Changing trainer loss or advantage computation inside NeMo RL, VeRL, or
  another downstream framework. NeMo RL changes only pre-dispatch routing for
  sharded Gym actors.
