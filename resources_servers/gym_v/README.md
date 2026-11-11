# Gym-V NeMo-Gym resources server

> Current upstream draft: see [the qualification benchmark](../../benchmarks/gym_v_smoke/README.md).
> Current Gym installs this component using `pyproject.toml` and `uv.lock`.
> Binary evaluation is opt-in with `reward_mode: task_success`; native accumulated
> training reward remains the default. Missing reward sessions are masked.


Env-id-parametric NeMo-Gym resources server wrapping any
[`gym_v.Env`](https://github.com/rohitrango/gym-v) behind `/seed_session`,
`/step`, `/close`, and `/verify`. Env selection is driven entirely by JSONL
task rows (`env_id`, `seed`, `env_kwargs`, ...). Adding an environment does not
require another server implementation, but training also requires an audited
binary-success predicate for that exact environment contract.

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

## Layout

```
resources_servers/gym_v/
  __init__.py
  app.py                  # GymVResourcesServer + DEFAULT_SYSTEM_PROMPT
  schemas.py              # Pydantic wire shapes
  _metadata.py            # JSON-safe Observation.metadata walker
  _observation.py         # observation_to_user_message + image_to_data_url
  configs/gym_v.yaml      # server defaults
  data/example.jsonl      # smoke-test task rows
  pyproject.toml / uv.lock # pinned Gym-V provider and managed dependencies
  tests/test_server.py
```

## Running the helper-level tests

The observation, metadata, schema, and task-data checks do not require `gym_v`
to be installed (they use an `ObservationLike` protocol). The server module
skips cleanly outside its isolated runtime. Run from a venv with `pytest`,
`pydantic`, `pillow`, and `nemo_gym` on `PYTHONPATH`:

```bash
cd <gym-checkout>
export PYTHONPATH="$PWD"
python -m pytest resources_servers/gym_v/tests -v --tb=short
```

## Per-server venv (full stack)

```bash
RAY_TMPDIR=/tmp gym env test --resources-server gym_v
```

First run installs `gym-v[games,spatial,reasoning-gym]` from the
rohitrango fork pinned at `7ccf14ecb5241f37372dd530accf77f7e7ef5fc6`.

## Paired agent

This server speaks `/seed_session` + `/step` + `/close` + `/verify` and
is paired with [`responses_api_agents/gymv_agent/`](../../responses_api_agents/gymv_agent/),
which extracts `\boxed{...}` from model output and drives the step loop.

## Reward and task success

Gym-V does not expose one universal success flag. Many environments award
partial credit, successful MiniGrid episodes can return less than `1.0`, and
termination/truncation alone does not distinguish success. The server therefore
keeps the accumulated float `reward` as the learner signal and emits a separate
`task_success` only from an explicit typed predicate.

Predicates can be supplied per task row with `success_criterion` or through the
server's exact `success_criteria_by_env_id` allowlist. The checked-in config
requires a criterion and contains predicates audited for its five example
environments against the Gym-V revision pinned in `pyproject.toml`. An
unlisted row is rejected instead of silently converting positive partial credit
into success. Legacy configs may leave `require_success_criterion=false`; those
rows return `task_success=null` and are not suitable for mixed-suite accuracy,
pass@k, or dynamic-sampling decisions.

## Additional provider prerequisites and action contracts

`Games/WordSearch-v0` and `Games/Wordle-v0` need NLTK's `words` corpus;
Wordle also needs `averaged_perceptron_tagger_eng`. Install these into an explicit
project-owned data directory before launching a worker, and set `NLTK_DATA` to
that directory:

```bash
python -m nltk.downloader -d <project-data>/nltk_data words averaged_perceptron_tagger_eng
export NLTK_DATA=<project-data>/nltk_data
```

Record the installed archive hashes with evaluation evidence. Python package
locks do not pin corpus contents. The pinned TextArena provider also attempts
NLTK setup during import/construction; installing its Python package alone is
not an offline-ready deployment. Missing assets are setup failures, never
model wrong answers. No corpus is redistributed in this contribution.

`Spatial/Empty-v0` and `Spatial/DoorKey-v0` are explicit exceptions to the
normal invalid-action return convention: their pinned MiniGrid wrappers raise
before stepping. The adapter checks these two action maps before dispatch,
returns corrective feedback without mutating the simulator, and counts the
attempt toward the task horizon. This is audited against Gym-V revision
`7ccf14ecb5241f37372dd530accf77f7e7ef5fc6`,
`gym_v/envs/multi_turn/spatial/2d/{empty,doorkey}.py`. Other exceptions remain
masked; matching an exception's text is never used to classify it.

The adapter uses the provider's native dataset construction. It does not monkeypatch
private dataset methods or share mutable datasets across sessions.
