# VisGym integration

This package connects [VisGym](https://github.com/visgym/VIsGym) visual,
interactive environments to NeMo-Gym. VisGym is distributed as a fork of
`gymnasium`; environments are constructed through the familiar
`gymnasium.make(env_id, **env_kwargs)` interface, but observations are images
and actions are environment-specific strings.

Unlike a static visual-question-answering dataset, a VisGym sample is a
stateful episode:

1. the environment is constructed and reset;
2. the reset produces the first visual observation;
3. the policy inspects that image and emits one action;
4. the environment applies the action and returns the next image and reward;
5. policy and environment turns repeat until termination or a configured cap;
6. NeMo-Gym verifies the episode and returns its accumulated training reward.

The initial validated integration targets `maze_2d/easy`. The resource server
is parameterized by `env_id`, so other VisGym environments can be added through
configuration and task data without creating another server implementation.

## Components

| Component | Location | Responsibility |
| --- | --- | --- |
| Resource server | `resources_servers/visgym` | Owns environment instances, renders observations, applies actions, and accumulates reward. |
| Rollout agent | `responses_api_agents/visgym_agent` | Alternates model calls with environment steps and extracts actions from model text. |
| Model server | `responses_api_models/vllm_model` | Generates policy responses and preserves token information needed for on-policy training. |
| Task rows | `resources_servers/visgym/data/*.jsonl` | Select the environment, seed, horizon, action grammar, and Responses API request. |

The data flow for one turn is:

```text
VisGym observation
  -> NeMo-Gym user message (text + image)
  -> Responses API policy call
  -> assistant text ending in \boxed{action}
  -> visgym_agent action extraction
  -> resources server /step
  -> next observation and reward
```

## Environment lifecycle

### Seed a session

`POST /seed_session` accepts either a `task_idx` loaded from configured JSONL
files or an inline `task_row`. The server:

- calls `gymnasium.make()` with `env_id` and `env_kwargs`;
- calls `env.reset(seed=..., init_state=...)` when an initial state is present;
- obtains the environment prompt with `env.get_prompt(**prompt_kwargs)` when
  available;
- converts the returned observation into a user message;
- allocates a server-side UUID used by subsequent requests.

The first image does not exist until reset, so task rows normally use an empty
Responses API input. The reset observation is returned separately as
`seed_obs`. Keeping `seed_obs` separate from generated output is important for
training: generated assistant messages carry prompt token IDs, generation token
IDs, and log probabilities, while the reset observation is environment state.

### Apply an action

`POST /step` accepts the session UUID and an `action_string`. The server calls
`env.step(action_string)`, records the turn count, accumulates reward, and
returns the next visual observation.

If an environment rejects an action, the server returns a recoverable user
observation with:

- reward `0.0`;
- `done=false`;
- a rendered image when available;
- a concise description of the invalid action.

This lets the model correct its action without losing the episode. A configured
`horizon_cap` can terminate an episode even when the underlying environment has
not terminated.

### Verify reward

`POST /verify` drains the accumulated reward for the session, attaches it to
the response metadata as `training_reward`, and returns it to NeMo-Gym. Reward
is accumulated across all turns rather than inferred from the final text.

### Close a session

`POST /close` releases the environment and removes server-side state. Closing
an already-closed session is treated as a successful no-op. The rollout agent
attempts to close sessions both on normal termination and after failures.

## Observation transport

Each environment observation becomes a user-role Responses API message. Its
content can contain:

- an `input_text` part with the environment prompt or step feedback;
- an `input_image` part encoded as a PNG or JPEG data URL;
- sanitized `env_info` metadata for inspection and debugging.

NumPy arrays, PIL images, and image-like observations are supported. If an
environment returns no image-like observation and `render_on_missing_image` is
enabled, the server calls `env.render()` as a fallback.

Server options control this conversion:

| Option | Default | Effect |
| --- | --- | --- |
| `image_format` | `PNG` | Data-URL image encoding. JPEG is also supported. |
| `image_jpeg_quality` | `90` | JPEG quality when JPEG encoding is selected. |
| `skip_images` | `false` | Omits image parts for text-only diagnostics. |
| `include_env_feedback` | `true` | Includes `info["env_feedback"]` in the next user message. |
| `render_on_missing_image` | `true` | Uses `env.render()` when reset or step does not return an image. |
| `enforce_horizon_cap` | `true` | Enforces the per-task `horizon_cap`. |

Metadata is recursively converted to JSON-safe values. Large image payloads
remain in message content rather than being copied into `env_info`.

## Action transport

The paired `visgym_agent` uses plain-text action transport. It takes the last
boxed item from the assistant response and sends it to `/step` as
`action_string`:

```text
<think>The open path is to the right.</think>
\boxed{('move', 1)}
```

For `maze_2d/easy`, legal actions are:

```text
('move', 0)
('move', 1)
('move', 2)
('move', 3)
('stop', 'stop')
```

The task's `act_grammar_regex` documents the legal grammar, while the agent's
`unboxed_action_regex` can accept an exact unboxed action for models that omit
the wrapper. Prose surrounding an unboxed action is rejected.

Important agent options include:

| Option | Purpose |
| --- | --- |
| `max_steps` | Global rollout-turn backstop in addition to the task horizon. |
| `done_if_no_boxed_answer` | Ends the rollout when no valid action can be extracted. |
| `max_no_boxed_truncation_retries` | Retries a truncated model response that omitted an action. |
| `no_boxed_truncation_retry_factor` | Increases the output-token budget on each truncation retry. |
| `re_emit_rules_each_turn` | Prepends a concise action-format reminder on later turns. |
| `rules_summary_template` | Per-environment reminder text. |
| `return_transitions` | Returns transition records instead of the compact training history. |

## Task-row format

A task row fully describes an episode. This is a shortened version of the
committed 5x5 maze fixture:

```json
{
  "agent_ref": {"type": "responses_api_agents", "name": "visgym_agent"},
  "env_id": "maze_2d/easy",
  "env_kwargs": {"maze_width": 5, "maze_height": 5},
  "seed": 1234,
  "task_id": "maze_2d_easy_seed1234_5x5",
  "act_grammar_regex": "^\\('(?:move|stop)',\\s*(?:[0-3]|'stop')\\)$",
  "horizon_cap": 8,
  "task_metadata": {
    "suite": "visgym",
    "difficulty": "easy",
    "maze_size": "5x5"
  },
  "responses_create_params": {
    "model": "policy_model",
    "input": [],
    "temperature": 0.7,
    "max_output_tokens": 1024,
    "tools": []
  }
}
```

Additional optional fields are:

- `init_state`: explicit state passed to `env.reset()`;
- `seed_key`: constructor keyword that should also receive the seed;
- `prompt_kwargs`: keyword arguments for `env.get_prompt()`.

Task rows may be preloaded with `task_jsonl_fpaths` or passed inline to
`/seed_session`.

## Maze-size curriculum dataset

The committed curriculum increases maze size in four ordered stages:

| Stage | Maze size | Rows | Seed range | Horizon cap |
| --- | --- | ---: | --- | ---: |
| 1 | 5x5 | 1280 | 1234-2513 | 8 |
| 2 | 7x7 | 1280 | 11234-12513 | 12 |
| 3 | 9x9 | 1280 | 21234-22513 | 25 |
| 4 | 11x11 | 1280 | 31234-32513 | 35 |

The 5x5 and 7x7 stages retain the original `2 * (maze_size - 1)` horizon. The
9x9 and 11x11 stages use tuned caps of 25 and 35 so larger mazes allow recovery
from navigation mistakes. Seed ranges are disjoint so the 5120 curriculum
tasks are unique.

The full curriculum is ~8 MB across five files, so it is generated rather than
committed. Build it (deterministically) before training:

```bash
resources_servers/visgym/scripts/create_maze_curriculum.py
```

Then use the combined manifest for ordered curriculum training:

```text
data/maze_2d_easy_curriculum_5x5_7x7_9x9_11x11_1280each_t1024.jsonl
```

The committed `maze_2d_easy_curriculum_5x5_7x7_9x9_*` fixtures (64 rows per
stage) and `maze_2d_easy_smoke.jsonl` are small enough to use directly for
smoke runs.

Rows 0-1279 are 5x5, rows 1280-2559 are 7x7, rows 2560-3839 are 9x9, and rows
3840-5119 are 11x11. NeMo-RL must use `data.shuffle=false`; shuffling the
combined file turns it into a mixed-size dataset rather than a curriculum. With
the recipe's default 64 prompts per step, each 1280-row stage supplies 20
prompt batches before the collector advances to the next size.

Separate stage manifests are included for experiments that need to hold,
repeat, or evaluate one stage independently. The manifest index records stage
paths, row counts, seed ranges, and horizon caps:

```text
data/maze_2d_easy_curriculum_5x5_7x7_9x9_11x11_manifest_index.json
```

Each row includes `curriculum_name`, `curriculum_stage`, `maze_size`, and
`curriculum_stage_index` in `task_metadata`, so metrics can be grouped by stage
without parsing task IDs.

Regenerate the dataset deterministically from the Gym repository root:

```bash
resources_servers/visgym/scripts/create_maze_curriculum.py
```

The generator accepts `--sizes`, `--samples-per-stage`, `--seed-base`,
`--seed-stride`, `--temperature`, and `--max-output-tokens`. Sizes must be
unique, odd, and strictly increasing. For example:

```bash
resources_servers/visgym/scripts/create_maze_curriculum.py \
  --sizes 5,7,9,11 \
  --samples-per-stage 1280
```

## Configuration

`configs/visgym_maze2d_thinking_agent.yaml` defines both the resource server and
the rollout agent. A model-server config must be loaded alongside it. For
NeMo-RL training, the effective NeMo-Gym configuration contains:

```yaml
config_paths:
  - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
  - resources_servers/visgym/configs/visgym_maze2d_thinking_agent.yaml
```

The maze config uses a short system prompt, re-emits compact rules on every
turn, caps the agent at 25 turns, and validates exact maze action strings.

## Dependency isolation and headless rendering

VisGym currently ships as a Gymnasium fork. NeMo-Gym installs
`requirements.txt` into this server's own venv, so VisGym's `gymnasium` does
not replace the version other servers use.

That install goes through `uv`, which **cannot parse VisGym's pyproject**: it
inherits Gymnasium's duplicate extras (`classic-control` and `classic_control`,
`mujoco-py` and `mujoco_py`, `toy-text` and `toy_text`), and PEP 685
normalization makes those collide:

```text
TOML parse error ... duplicate normalized extra name `classic-control`
```

pip still accepts them, so the pinned revision is built into a wheel once and
`requirements.txt` installs that wheel.

### Building the VisGym wheel

`vendor_wheels/` holds build output and is gitignored, so **a fresh clone has
no wheel** and the server cannot start until one is built. There is no wheel to
download: VisGym is not on PyPI, and the whole reason this script exists is
that the published source cannot be installed by `uv` at all. What is
distributed is the script plus the pinned revision, and the wheel is
reproduced from those.

Most users never run this by hand. The NeMo-RL launcher
(`examples/nemo_gym/nemotron-3-super-omni/visgym_launch.sh`) checks for the
wheel and builds it when it is missing, into the code snapshot the job runs
from. To build it explicitly:

```bash
resources_servers/visgym/scripts/build_visgym_wheel.sh
```

That clones `https://github.com/visgym/VIsGym.git` at the revision pinned in
the script and runs `pip wheel --no-deps`, leaving
`vendor_wheels/gymnasium-1.1.1-py3-none-any.whl`. The package is named
`gymnasium` because VisGym is a fork of it.

Two prerequisites are worth checking first, because both fail on machines that
otherwise look fine:

- **A `python3` that has pip.** `uv` cannot substitute here -- it is the tool
  that cannot parse the project in the first place. The script exits with
  status 2 and says so if pip is absent; point `PYTHON_BIN` at an interpreter
  that has it.
- **Outbound access to `github.com`**, unless you supply the source yourself.

On a cluster without network access, clone VisGym somewhere that has it, copy
the tree over, and build against it:

```bash
VISGYM_REPO_ROOT=/path/to/VIsGym \
  resources_servers/visgym/scripts/build_visgym_wheel.sh
```

`VISGYM_REV`, `VISGYM_URL`, `OUT_DIR` and `PYTHON_BIN` override the pin, the
remote, the output directory and the interpreter. Confirm the result with:

```bash
ls resources_servers/visgym/vendor_wheels/gymnasium-*.whl
```

Copying an already-built wheel between machines also works -- it is a pure
Python wheel with no compiled extensions -- but prefer rebuilding from the pin,
since a stray wheel carries no record of which revision produced it.

Once VisGym drops the duplicate extras, the wheel step can go away and
`requirements.txt` can name the git revision directly.

The robotics (`fetch_*`) and `refcoco_plus` tasks need two more forked
packages; they live in `requirements-robotics.txt` and are built by
`scripts/build_vendor_wheels.sh`, which takes a VisGym checkout as its first
argument because both sources live inside that tree:

```bash
resources_servers/visgym/scripts/build_vendor_wheels.sh /path/to/VIsGym
```

Every purely rendered task, including the maze curriculum and the blended
ten-environment manifest, runs without them.

For source-checkout development, set `VISGYM_REPO_ROOT` before importing the
server. The path is inserted before importing `gymnasium`.

The server defaults to headless rendering:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

`maze_2d/easy` does not require MuJoCo robotics assets or an external dataset.

## Exact token history during training

Multi-turn on-policy training must use the exact prompt token IDs sampled by
vLLM. Re-rendering previous assistant text can produce different whitespace or
multimodal placeholder tokens, making the rollout off-policy.

The training model proxy therefore promotes the latest recorded prompt and
generation token IDs to top-level `required_prefix_*` request fields. NeMo-RL
splices that exact prefix into the next rendered prompt and realigns historical
and current image placeholders independently. This behavior is training-only;
it does not change the resource-server API.

## Testing

From the Gym repository root:

```bash
pytest -q \
  resources_servers/visgym/tests \
  responses_api_agents/visgym_agent/tests/test_app.py
```

The unit tests cover task validation, observation conversion, lifecycle and
reward accumulation, invalid-action recovery, action extraction, truncation
retries, rule re-emission, and cleanup. The real-environment smoke test runs
when the pinned VisGym dependency is installed and otherwise reports a skip.

## NeMo-RL training

The NeMo-RL side is one recipe:

```text
examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-tp8ep16cp2-async-gym-visgym.v1.yaml
```

It inherits the Nemotron Super Omni async-Gym recipe and changes what the
multi-turn visual shape requires: one image per turn allowed through vLLM
(`limit_mm_per_prompt.image` at least the agent's `max_steps`), sequence
packing off so exact prompt/generation token replay survives, in-flight weight
updates off so a rollout stays on one weight version, `data.shuffle=false` for
the curriculum, and a 512-token per-turn generation budget inside a 16384-token
context. Reasoning is on via `chat_template_kwargs.enable_thinking: true`.

Submit it with the launcher, which generates the curriculum if it is missing:

```bash
DRY_RUN=true MODEL_PATH=... CONTAINER=... PERSISTENT_CACHE=... \
  SLURM_ACCOUNT=... examples/nemo_gym/nemotron-3-super-omni/visgym_launch.sh
```

Drop `DRY_RUN=true` to submit. The default 80 steps consume one ordered pass of
the curriculum: 20 steps per stage at 64 prompts per step. Beyond 80 steps the
finite async dataloader repeats the manifest and starts another
5x5 -> 7x7 -> 9x9 -> 11x11 cycle; use a single stage manifest instead when a
run should stay at one maze size.

## Adding another VisGym environment

1. Confirm the pinned VisGym package registers the desired `env_id`.
2. Add a small JSONL task fixture with deterministic seeds and a finite
   `horizon_cap`.
3. Define the legal `act_grammar_regex` and align the agent's
   `unboxed_action_regex` and prompt with that grammar.
4. Verify that reset and step observations are image-like or that
   `env.render()` provides an image.
5. Add a real reset/step smoke test and a mocked server lifecycle test.
6. Run a bounded NeMo-RL smoke before increasing rollout length or parallelism.

## Current scope and limitations
