# VisGym integration

> Qualification scope: see [the qualification benchmark](../../benchmarks/visgym_maze_smoke/README.md).
> Current Gym installs this component using `pyproject.toml` and `uv.lock`.
> Binary evaluation is opt-in with `reward_mode: task_success`; native accumulated
> training reward remains the default. Missing reward sessions are masked.


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
is parameterized by `env_id`; additional native-rendered families can be
qualified through task data without another server implementation. Maze3D and
Fetch are explicitly rejected pending separately qualified renderer support.
The experimental corridor/state-diagram renderers are not included. Arbitrary environments from the forked Gymnasium
registry are not automatically valid training tasks.

## Session and failure contract (current main)

The resource server accepts both the existing agent's native seed request and
Gym's typed `ResourcesSeedSessionRequest`. Typed seeds bind the requested
session ID to an episode/task; repeated seeds are idempotent and conflicting
owners are rejected. `/close_session` uses the framework lifecycle hooks.
The legacy `/step`, `/close`, and `/verify` routes require the owning cookie
when the session was seeded through the typed protocol.

The paired custom agent still uses the native seed/observation loop through
`legacy_agent`. Typed seed/close HTTP conformance is tested; this does not
claim a complete typed agent migration or checkpoint/resume qualification.

Normal completion calls `/close` with `discard_reward=false`: this releases
the native environment but preserves its score until `/verify`. Verification
before native close returns 409 without consuming the score. Verification then
retires episode state and retains a compact grade/request digest for the
framework's bounded retry window (`session_close_retry_window_seconds`). An
identical retry gets the same grade; a changed request is masked as
`verification_mismatch`. Expired/missing state is masked as `session_lost`.
Abort close (`discard_reward=true`) discards reward and invalidates retry evidence.

Provider-reported invalid actions retain their normal observation/reward/done
semantics, including the configured horizon. Unexpected step, rendering, or
success-contract errors, and nonfinite rewards, end the episode and return a
masked `simulator_failed` grade. They are not counted as incorrect model
answers. Failed native calls are never repeated within that session; framework
or caller recovery must start a new attempt. Cancellation still propagates.

## Components

| Component | Location | Responsibility |
| --- | --- | --- |
| Environment bundle | `environments/visgym` | Provides the canonical config and a committed example dataset. |
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

The generic `gymnasium` server and agent remain the right choice for string
observations and function-call actions. VisGym keeps a specialized pair because
it must transport multimodal image messages, parse boxed text actions, drain a
final accumulated training reward, and preserve exact prompt/generation token
metadata across every turn. Reusing the generic pair would require changing its
public request and response schemas for all existing consumers.

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

Provider-reported invalid actions are returned as ordinary feedback. Unexpected
exceptions instead follow the masked failure contract above. A configured
`horizon_cap` can terminate an episode before native termination.

### Verify reward

`POST /verify` grades the accumulated reward for the closed native session, attaches it to
the response metadata as `training_reward`, and returns it to NeMo-Gym. Reward
is accumulated across all turns rather than inferred from the final text.
It separately emits binary `task_success` only when an audited VisGym family
terminates with raw reward `1.0`; positive nonterminal rewards do not
count as success. Strict mode rejects registry IDs outside the 34 audited
`{easy,hard}` task IDs. Disabling it is for reward-only diagnostics and returns
`task_success=null` for those unsupported IDs.

### Close a session

`POST /close` releases the environment, retaining reward until verification unless
`discard_reward=true`. Closing
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
| `return_transitions` | Must remain `false`; complete accepted history is returned as flat output. |

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

## Configuration and compaction

The neutral example config is `environments/visgym/config.yaml`; it uses the
committed five-row Maze2D dataset and requires no data generation. Start it with
`gym env start --environment visgym --model-type vllm_model`. Set the model
endpoint and sampling settings for your chosen policy.

Merge `responses_api_agents/visgym_agent/configs/first_last_two.yaml` to enable
compaction through Gym #3368. It keeps the first and latest two reasoning/image
groups, retains all actions and feedback, and requires capture-owned model
routes and measured context admission. See [the agent contract](../../responses_api_agents/visgym_agent/README.md).

Task rows accept native environment kwargs, seeds, assets and initial states.
The server returns native accumulated reward. Requests containing
`task_metadata.reward_shaping` are rejected with HTTP 400, including permissive
reward-only mode. Curriculum generation, HF reset conversion, synthetic assets,
model-specific thinking prompts and custom reward shaping belong in separately
reviewed experiments, not this adapter.

## Dependency isolation and headless rendering

VisGym ships as a Gymnasium fork. Current Gym installs this component from
`pyproject.toml` and `uv.lock` into its own venv, so it does not replace the
Gymnasium version used by other servers. All four agent/resource components
inherit the shared resolver policy. VisGym explicitly enables OpenCV for native
image processing and gprof2dot for provider imports through the existing
`scripts/component-lock-policy.json` exception mechanism.

The dependency points to
[the packaging fork](https://github.com/aroshanghias-nvd/VisGym/commit/33f996cda320db5f127b1e2a858c5653bebc90fc),
commit `33f996cda320db5f127b1e2a858c5653bebc90fc`, based on upstream VisGym
`927271d107ad0196ad6aa597095ca57d01c6ddbb`. Its only change removes three
redundant extra aliases from `pyproject.toml`: `classic_control`, `mujoco_py`,
and `toy_text`. Their normalized names and dependency arrays already exist
under hyphenated keys. This fixes uv's duplicate-extra parser error without
changing provider Python code, assets, sampling or scoring.

No VisGym wheel is bundled in Gym. A clean installation fetches the pinned Git
commit and builds it through uv; it requires GitHub and package-index access or
populated caches. Commit pins, not a moving branch, identify the dependency.
The install regression checks the installed distribution's Git URL and exact
commit, because stock Gymnasium has the same distribution name/version but
lacks VisGym's environments.

The fork preserves upstream LICENSE, NOTICE and package license declarations
unchanged. The MIT metadata versus Apache-2.0 root-license discrepancy remains
an upstream clarification item; this packaging repair does not choose or
rewrite licensing terms. Gym's `ATTRIBUTIONS.md` records the external dependency,
and the original Gymnasium v1.1.1 MIT notice remains in
`vendor_licenses/gymnasium-v1.1.1-LICENSE` for attribution.

Once upstream accepts the packaging repair, a separately validated update can
point the pin back to upstream. Do not substitute the PyPI Gymnasium release.

The checked-in real-runtime smoke covers `maze_2d/easy`. Other native-rendered
families require their assets, provider dependencies and reset/step validation;
they are candidates, not qualified support claims. Optional robotics/counting
wheel builders are not shipped. Fetch and Maze3D are rejected before any native
session is created, rather than silently substituting different observations.

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

- Real VisGym execution has been validated for `maze_2d/easy`; the remaining
  rendered environments are configuration candidates, not runtime-validated
  support claims.
- Asset-backed tasks require their provider dependencies and data. No alternate
  renderer or shaping protocol is enabled implicitly.
- The manifest workload passes its verifier/publication checks. Legacy server CI
  still requests five saved rollout rows; CONTRIBUTING documents this routing
  mismatch and asks contributors not to duplicate artifacts just for that check.
- The combined NeMo-RL learner smoke still requires execution on the target GPU
  cluster before this integration is training-ready.
- VisGym is a pinned external source dependency, not a bundled wheel. It retains
  the Gymnasium distribution name/version; verify its Git origin when debugging
  installation. The upstream license-metadata discrepancy remains documented.
