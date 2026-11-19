# ARC-AGI-3 agent: the official ARC Prize harness (option)

Overview of all the ARC-AGI-3 environments and agents, with a smoke run: [`resources_servers/arc_agi_3/OVERVIEW.md`](../../resources_servers/arc_agi_3/OVERVIEW.md).

This agent plays ARC-AGI-3 games with the **official ARC Prize harness**,
[arc-agi-3-benchmarking](https://github.com/arcprize/arc-agi-3-benchmarking), as either of upstream's two harnesses for
chat-completions models (`harness_type`):

| `harness_type` | Upstream harness | History |
|---|---|---|
| `provider` (default) | Provider Adapter: `continuous_conversation` with the open-source Chat Completions adapter | Every earlier reply is resent with its reasoning; a summary compaction replaces the oldest turns at `trigger_tokens` |
| `standard` | Standard: `manual_rolling` | Earlier replies are resent as text only; the oldest turn pairs are dropped while the transcript exceeds `MAX_CONTEXT_LENGTH` characters; no summaries |

It is an option next to
[`arc_agi_3_agent`](../arc_agi_3_agent/README.md), which plays them with the Kaggle kit's harness and stays the default.
The same [`arc_agi_3` resources server](../../resources_servers/arc_agi_3/README.md) scores both: same rows and
pins, the official game server's scorecard as the authority (reward = score / 100) and RHAE@T from the server's own
action times.

## What runs

- **The harness:** the pinned upstream commit `27f3d03d` (2026-11-12; MIT, Copyright (c) 2025 ARC Prize), unmodified.
  `official_harness_manifest.json` pins the sha256 of every file.
  - Upstream pins openai 2.41.1 and anthropic 0.125.0, and Gym pins older versions, so the harness has its own venv:
    `uv sync --frozen --no-dev` from its `uv.lock`.
  - It runs in one worker process per game (`worker.py`), using upstream's own `BenchmarkingAgent`.
- **The game** is on the resources server's official `arc_agi` game server (the pinned files, RESET keeps the level).
  The worker plays exactly as upstream's `Swarm` does: an ONLINE `arc_agi.Arcade` makes the game on a scorecard that
  this agent opened for the rollout.
  - The worker's report (guid, levels, final state, and the actions the server answered) goes to the verifier, which
    checks it against the card.
  - The rollout's `stop_reason` says why the game ended: `won`, `level_action_budget` (the level the game was on used
    its own allowance, upstream's ceil(baseline x `MAX_ACTIONS_BASELINE_MULTIPLIER`)), `total_action_budget` (the
    whole game's cap, upstream's `MAX_ACTIONS_HARD_CAP` or total `MAX_ACTIONS`: not a verdict on the last level),
    `play_window` (the window or upstream's `MAX_RUNTIME_SECONDS`), `crashed` (a harness error, e.g. replies cut at
    the output cap), `cancelled` (the agent server shut down) or `harness_end` (upstream stopped for another reason).
    Upstream reports both budgets as `ACTION_BUDGET`; the worker tells them apart with upstream's own checks on the
    final state (`action_budget` in `result.json`: `level` or `total`). The agent state (`agent_state`) is `gave_up`
    for both budgets and `play_window`.
  - A game in which no model call got an answer (an unreachable or failing endpoint) is reported to the verifier as
    `model_error`: its result is invalid and the rollout masked, not scored as a 0-action game.
  - There is no `benchmark.json`: the verifier's rows (`rollouts.jsonl`) hold the scores, with RHAE@T on the server's
    clock.
- **Model calls:** the harness's own OpenAI client calls this agent's route `/official/<game>/v1/chat/completions`.
  The route sends the call to the Gym model server on the rollout's own path (`url_path_for_run`), so Gym's capture
  (`model_call_capture_dir`, token capture) attributes every call to its rollout.
  - The worker tags each call with request headers (`worker.CALL_HEADERS`): the game's step (actions answered + 1) and
    level (levels completed + 1) when it was sent, and its role: `gameplay` / `main`, or `helper` / `harness-summary`
    for upstream's compaction summary.
  - What the model sees of its progress: each observation starts with `State: ...` and `Levels completed: N` (the
    level on the board is N + 1), with `New Level:` before the first frame of a new level; no step number.
- **Game version:** the worker makes the game by its full id (`<game>-<version>`); arc_agi's ONLINE client otherwise
  asks the server for the game name only and plays whichever version it returns. The first RESET of a game (the
  engine imports the game then) waits `reset_timeout_s` (default 120 s; the client's own timeout is 10 s).
- **The play clock (`run_clock`):** `shared` (default, for benchmarking) is one run clock: the origin is taken before
  the first game's scorecard is opened, and every game's window ends at origin + `play_seconds`; a game queued past
  that end does not start. `per_game` (data generation, RL) gives each game `play_seconds` from its own start, whenever
  it starts; RHAE@T is then measured from the game's start.
  - The worker stops at the window's end: upstream's `MAX_RUNTIME_SECONDS` is capped, and a `SIGALRM` interrupts a
    model call in flight.
  - A worker still running at the end + `window_grace_s` gets SIGTERM. It writes its report and exits.
  - The server cannot refuse late steps, so an action after the window counts in the final score, not in RHAE@T.
- **Upstream's own rules stay:**
  - its prompts and action parsing;
  - a RESET after GAME_OVER;
  - per-level action budgets: a level's budget is ceil(human baseline × `MAX_ACTIONS_BASELINE_MULTIPLIER`) actions,
    and the game stops when one is spent;
  - its history handling: the provider harness's summary compaction, the standard harness's rolling window.

## Earlier reasoning stays in the history (provider)

The provider harness resends the model's reasoning of every earlier turn. Nothing between the harness and vLLM drops
it:

1. **The runtime is fixed per type.** For `provider` the agent accepts only upstream's continuous conversation with the
   open-source Chat Completions adapter (`runtime.state: continuous_conversation`), for `standard` only
   `manual_rolling`, and rejects other configs at start.
   - `reasoning_replay` can be `reasoning_content`, `reasoning` or `reasoning_aliases`; all three resend the full
     text.
   - `manual_rolling` (the standard harness) resends only the reply text, by design.
2. **The route.** It puts an earlier turn's `reasoning_content` in a `<think>` prefix of its content, because Gym's
   request model drops unknown fields. The Gym vLLM model server splits that prefix back into `reasoning_content` and
   `reasoning` for vLLM. The route returns the reply's reasoning in `reasoning_content`.
3. **The model server.** With `provider`, the agent refuses to start unless the Gym model server sets
   `uses_reasoning_parser: true` and `uses_interleaved_reasoning: true`. Without them, Gym drops earlier reasoning.
4. **The chat template.**
   - The Qwen kit template keeps earlier reasoning with `preserve_thinking: true`, which the config sends.
   - Nemotron's model server config sets `truncate_history_thinking: false`.
5. **Compaction.** When the history reaches `trigger_tokens`, upstream's summary compaction replaces the oldest turns
   with a summary that the model writes. The turns it keeps still have their reasoning.

The CPU stack test checks this on the vLLM side: every assistant turn of every request carries the reasoning of that
reply, also after compactions. For `standard` it checks the opposite: no earlier reasoning, and a rolled transcript.

## Settings

[`arc_agi_3_official.yaml`](../../resources_servers/arc_agi_3/configs/arc_agi_3_official.yaml) holds the resources
server and this agent. `harness_type` selects the harness, and `benchmark_configs` holds one upstream
`model_configs.yaml` entry per type.
- **Defaults:** settings for a model served with a 262,144-token window. Window sizes
  are model-specific; set them for your model.
  - Both: multiplier 5, `max_tokens` 12288, `enable_thinking` and `preserve_thinking`.
  - `provider`: context 262,144 tokens; compaction at 192,000 tokens (upstream's 96k / 131k ratio) with 8,192
    headroom.
  - `standard`: `MAX_CONTEXT_LENGTH` 110,592 characters: a 131,072-token window less the output cap and the headroom.
- **Sampling:** not set by default, so the engine's defaults apply, as with upstream. Set `request.temperature`,
  `request.top_p` and `request.seed`; `top_k` goes in `request.extra_body`.
- **Reasoning effort:** `reasoning_effort` (e.g. `low` / `high` / `max`; unset = the template's default) is sent as the
  chat-template kwarg `reasoning_effort_kwarg` (default `reasoning_effort`; set the kwarg the model's template reads)
  on every request that asks for thinking. An upstream
  `request.reasoning_effort` is sent as it is instead.
- **Set by the agent:** the type's runtime `sdk` / `api` / `adapter_id` / `state` and the client (its own route).

## Launch on one GB200 / GB300 node

Serve the model as for [`arc_agi_3_agent`](../arc_agi_3_agent/README.md#launch-on-one-gb200--gb300-node), with a
reasoning parser. Then:

```bash
A=arc_agi_3_official_agent.responses_api_agents.arc_agi_3_official_agent
COMMON=(--config resources_servers/arc_agi_3/configs/arc_agi_3_official.yaml
        --config responses_api_models/vllm_model/configs/nemotron_3_5_super_vl.yaml
        --model-url http://127.0.0.1:8000/v1 --model-api-key dummy
        ++arc_agi_3_resources_server.resources_servers.arc_agi_3.expected_games=25
        ++$A.harness_type=provider ++$A.run_name=official_seed1)   # or standard; and vllm serve --seed 1
gym env start "${COMMON[@]}" &   # wait for "servers ready!"
gym eval run --no-serve "${COMMON[@]}" --agent arc_agi_3_official_agent \
  --input resources_servers/arc_agi_3/data/public25.jsonl --concurrency 25 --output results/arc3_official_rollouts.jsonl
```

- **First start:** the agent installs the harness from `harness_source`:
  - `git` (the default) clones the pinned commit;
  - a directory or a `.tar.gz` of the repository at that commit also works, for nodes without GitHub (for example
    `git archive 27f3d03d`).
- **Requirements:** building the venv needs `uv` and access to PyPI, or a filled uv cache.
- **A short smoke:** the two committed example games (cn04, ls20), once per type:
  `--input resources_servers/arc_agi_3/data/example.jsonl --limit 2 --concurrency 2` (the first two rows: one per game), `expected_games=2`, a shorter
  `play_seconds`, and `++$A.harness_type=provider` or `standard`.

## Outputs

`<output_dir>/<run_name>/` (default `results/arc_agi_3_official/`) holds:
- `run_config.json`: the effective config, upstream provenance, the game server and the model server config;
- `games/<index>-<game>/`, one directory per game:
  - `result.json`: the worker's report of the game;
  - `worker.log`: the harness's log;
  - `recordings/`: upstream's own per-step records;
  - `settings.json`;
  - `requests.jsonl`, with `save_request_logs`.

Each rollout row also carries `harness: official`, `harness_type`, `harness_revision` (upstream's repository and
pinned commit), `sampling` (the type's upstream `request` settings and the effort the route adds), and the game's
call log, as `arc_agi_3_agent` reports it: `ng_agent_observations` (the accepted gameplay calls in order, one
`context_compaction` per summary with its `harness-summary` helper calls) and `agent_call_log` (every call sent:
role, history id, status, response id, error, step, level, times). Repeats of one row get their own route key,
game directory and scorecard (`<index>-<game>-p<n>`).

The scores are in the verifier's rows: `summarize_rhae.py --expected <rows> <rollouts.jsonl> --hours 1 2`. The
game server keeps each closed card and its per-action recording (see the resources server README).

## Tests

```bash
python responses_api_agents/arc_agi_3_official_agent/setup_official_harness.py    # once: the pinned harness + venv
pytest responses_api_agents/arc_agi_3_official_agent/tests   # plays the committed example games
```

- **Unit tests:** config validation per harness type (for `provider`, the modes that would drop reasoning are
  rejected), the shipped configs of both types, the model-server checks, the
  route's mapping in both directions, the context-length 400, end states, the run clock, the pins and the install
  check.
- **Lifecycle tests:** small stand-in workers check the report, a harness error, a worker that dies without a report
  (this fails loudly), the window end, cancellation and shutdown.
- **Stack tests, one per type:** the real upstream worker plays the two example games through the real official game
  server, and its model calls go through the route and the real Gym vLLM model server to a stub vLLM. They check:
  - `provider` (with forced compactions): the reasoning of every earlier turn on the vLLM side;
  - `standard`: no earlier reasoning resent, and a rolled transcript;
  - that the verifier accepts both games against their scorecards, with RHAE@T on the server's clock;
  - upstream's per-level budget stop.
- The stack test skips without the game files or the harness install.
