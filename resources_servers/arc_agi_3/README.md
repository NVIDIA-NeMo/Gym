# ARC-AGI-3 (public25)

Overview of all the ARC-AGI-3 environments and agents, with a smoke run: [`resources_servers/arc_agi_3/OVERVIEW.md`](OVERVIEW.md).

[ARC-AGI-3](https://arcprize.org/arc-agi/3/) is a set of interactive, turn-based games. The agent sees a 64×64 board, sends actions (keys, clicks, RESET) and must discover the rules and the goal of each level by itself. A game has several levels. The score of a game is RHAE (relative human action efficiency):

- a cleared level `i` (1-based) scores `min(115, (human_actions_i / agent_actions_i)^2 * 100)`;
- the game score is the mean of the level scores weighted by `i`, capped at the weighted fraction of the levels that scored, times 100;
- RHAE of a run is the mean game score (0-100) over the expected games. `rhae_<h>h` counts only the levels cleared in the first `h` hours of the run (one run clock: the first game's start, as the kit's benchmark start).

This environment plays the **25 public games** (public25) with the **ARC-AGI-3 Kaggle kit's harness** (a public Kaggle dataset, pinned file by file). The agent is [`responses_api_agents/arc_agi_3_agent`](../../responses_api_agents/arc_agi_3_agent/README.md).

The **official ARC Prize harness** ([arc-agi-3-benchmarking](https://github.com/arcprize/arc-agi-3-benchmarking)) is an option, with either of its chat-completions harnesses (`harness_type: provider | standard`): [`responses_api_agents/arc_agi_3_official_agent`](../../responses_api_agents/arc_agi_3_official_agent/README.md), config [`arc_agi_3_official.yaml`](configs/arc_agi_3_official.yaml). Same rows, pins, game server and scoring.

This server is the authority and the verifier for both:
- **It runs the official game server.** This is `arc_agi`'s own server on the pinned game files (`game_server.py`), and the harness plays every game through it, one scorecard per rollout.
- **The reward is the card's score.** At `/verify` it closes the rollout's scorecard (if the harness has not), and sets the reward to the card's score / 100.
- **It fails loudly on a mismatch.** The agent's own report (game run guid, levels, won or not, action count) must equal the card, or verify raises.
- **Timed scores use the server's clock.** RHAE@1h / @2h come from the server's own per-action timestamps.
- **The pins are checked.** It compares the row's pins (full game id, version, sha256 of the game files) with the games it loaded, and checks that the agent's report is for that exact game.

## Data

Two example games are committed (cn04, ls20, under `data/environment_files`); the other 23 are not. `create_dataset.py` builds the rows from the competition's `environment_files` (`<game>/<version>/{<game>.py, metadata.json}`, arc_agi 0.9.8, 3.9 MB):

```bash
# Option A: download the files from the Kaggle competition (needs a Kaggle token and the competition rules accepted)
python resources_servers/arc_agi_3/create_dataset.py \
    --download-kaggle-competition resources_servers/arc_agi_3/data \
    --output resources_servers/arc_agi_3/data/public25.jsonl
# Option B: use a copy you already have
python resources_servers/arc_agi_3/create_dataset.py \
    --environments-dir /path/to/environment_files --output resources_servers/arc_agi_3/data/public25.jsonl
# Check a copy against a dataset (sha256 of every game file)
python resources_servers/arc_agi_3/create_dataset.py \
    --environments-dir resources_servers/arc_agi_3/data/environment_files \
    --check resources_servers/arc_agi_3/data/public25.jsonl
```

The server and the agent read the games from `data/environment_files` (config `environments_dir`). The 25 game versions are pinned in `create_dataset.py`.

- `data/example.jsonl` (committed): Gym's five example rows over the 2 committed games, cn04 (keyboard and click) and ls20 (keyboard), alternating (each row its own attempt). It runs with no download.
- `data/public25.jsonl` (generated): all 25 games, one row per game.

Each row's `verifier_metadata` has the game id, version, baselines, number of levels, the sha256 of the game files and the harness dispatch index (rows are in sorted game-id order). Its `agent_context` gives the agent session the same pinned game (game id, full id, version, file sha256, dispatch index).

**Game names.** A game is `<game>-<version>` (`full_game_id`, e.g. `ls20-9607627b`), its files are in `<game>/<version>/`, and the server, the agents and `summarize_rhae.py` key games by that full id. One `environments_dir` can hold several versions of a game name: the synthetic game sets have them (e.g. `sg0005-aee9edbb` and `sg0005-2b3d7198`), and each version is its own game with its own pins. The public25 games have one version each. Asked by full id (`/api/games/<game>-<version>`, RESET with that id), the game server serves exactly that version.

**Game runtimes.** Some synthetic games import a runtime that is not part of their game file: the param25 games import `arcgames` from their upstream tree. arc_agi loads a game without a module path, so the game cannot find a tree beside it. The config `python_path` lists directories the server puts first on the game server's `PYTHONPATH` (relative paths are relative to the Gym root), e.g. `++arc_agi_3_resources_server.resources_servers.arc_agi_3.python_path='[/data/arc3-games/python/param25/src]'`. Games that do not import it are unaffected.

## Example data

`data/example.jsonl`, `data/example_metrics.json` and `data/example_rollouts.jsonl` are Gym's example-data contract
(`ng_test +entrypoint=resources_servers/arc_agi_3 +should_validate_data=true`). The rows are `create_dataset.py
--environments-dir resources_servers/arc_agi_3/data/environment_files --example-output resources_servers/arc_agi_3/data/example.jsonl`;
the metrics are `gym dataset collate "+config_paths=[resources_servers/arc_agi_3/configs/arc_agi_3.yaml]"
+output_dirpath=<tmp> +mode=example_validation`. The rollouts are a CPU run of the real servers (this server, its game
server, `arc_agi_3_agent` with the pinned harness, the Gym vLLM model server) against the scripted model of
`responses_api_agents/arc_agi_3_agent/tests/stub_model.py` (one action per call, no learning), so every game ends on an
8-action budget with no level cleared:

```bash
python responses_api_agents/arc_agi_3_agent/tests/stub_model.py --port 8765 --log /tmp/stub.jsonl &
A=arc_agi_3_agent.responses_api_agents.arc_agi_3_agent
COMMON=(--config resources_servers/arc_agi_3/configs/arc_agi_3.yaml
        --config responses_api_models/vllm_model/configs/qwen_vl_arc3_kit.yaml
        --model-url http://127.0.0.1:8765/v1 --model-api-key dummy --model stub
        ++arc_agi_3_resources_server.resources_servers.arc_agi_3.expected_games=2
        ++$A.run_name=example_rollouts ++$A.play_seconds=120 ++$A.max_actions_per_game=8
        ++$A.concurrency=5 ++$A.num_attempts=5 ++$A.seed=1)
gym env start "${COMMON[@]}" &   # wait for "servers ready!"
gym eval run --no-serve "${COMMON[@]}" --agent arc_agi_3_agent \
    --input resources_servers/arc_agi_3/data/example.jsonl --concurrency 5 \
    --output resources_servers/arc_agi_3/data/example_rollouts.jsonl
```

The committed rollouts have the run's absolute path prefix removed from `run_dir` (repository-relative).

## The game server (the authority)

At start, this server launches `python -m resources_servers.arc_agi_3.game_server`, a child process, and stops it at shutdown. That process runs `arc_agi`'s own Flask server (`Arcade(OFFLINE).listen_and_serve`, arc_agi 0.9.8 / arcengine 0.9.3) on `environments_dir`. RESET keeps the level, as in the kit (`ONLY_RESET_LEVELS=true`).
- **Where agents find it:** `GET /game_server` returns its URL. The Kaggle kit agent plays through it with taaf's own ONLINE `ArcadeSpec`; the official agent plays upstream's own ONLINE `Arcade`.
- **What it writes** to `<game_server_dir>/<start time>/`:
  - `cards/<card_id>.json`: every scorecard as it was closed. A closed card is gone from the server, and the kit's taaf closes its card at the end of its game.
  - `recordings/<card_id>/<game>-<guid>.jsonl`: one line per response, with the server's UTC `timestamp`, the action, `levels_completed` and `state` (no frames).
  - `game_server.log`.
- **Human baselines:** the server's game-info routes (`/api/games`, `/api/games/<id>`) also give each game's `baseline_actions`. arc_agi's own server leaves them out (the competition setting), but both harnesses need them:
  - the kit's taaf records them, and the kit's `rhae.py` scores `benchmark.json` with them;
  - the official harness sets its per-level action budgets from them; without them it falls back to 10 actions per game.

  Scoring and game logic stay arc_agi's.
- **Idle cards:** a card idle for `scorecard_timeout_min` (default 240) is closed by the server. The server's own default, 15 min, can be shorter than one model turn.
- **Cost per action** (measured on one client): 2-7 ms through the server against 0.3 ms in process, and 4-10 ms with the recordings. Most of it is serializing the frame JSON.

## Verify

Request: the standard verify request plus:
- `game_record`: the agent's own report of the attempt, or null when the game could not start. The Kaggle kit agent sends the taaf `GameRun.to_json_dict()`; the official agent sends the same fields with an `actions` count instead of a `history`.
- `scorecard_id`, `game_guid`, `scorecard_api_key`: the rollout's card, its game run and the key that opened the card. The key lets the verifier close a card that is still open.
- `benchmark_start_time`: the run clock's origin as a UTC time, taken before the first game started; the agents always send it. Without it there is no timed score: `rhae_<h>h` are null and `clock_error` says why.
- `agent_error` / `persistence_error`: the agent's own problems.
- `model_error`: set by the agents when no model call of the game got an answer (an unreachable or failing endpoint).
  The game was played without the model: the result is invalid and the rollout masked (`mask_sample`, failure kind
  `model_request_error`).

The scorecard is the source of truth. Verify raises (HTTP error, so the rollout fails visibly) when any of these holds:
- the card does not hold exactly this game;
- a run's guid differs from the agent's;
- the last run's levels or won state differ from the agent's;
- the card's action count differs from the agent's;
- the server's recording disagrees with its own card;
- the request has no card.

Measured: the card counts exactly the actions the harness sent, and the RESET of `make()` is not counted.

A row without pins, or with pins that differ from the loaded game, is rejected (HTTP error, so the rollout fails visibly).

Response fields:

| Field | Meaning |
|---|---|
| `reward` | the scorecard's score / 100 |
| `rhae` | the scorecard's game score (0-100; the SDK's best run) |
| `rhae_1h`, `rhae_2h` | game score at 1 h and 2 h of the run clock (`score_hours`), from the server's action times; null when the timing is invalid or the request has no `benchmark_start_time` |
| `levels_completed`, `number_of_levels`, `actions` | the attempt, from the scorecard |
| `state` | terminal state from the scorecard: `won` / `not_won` (`not_started` without a game) |
| `agent_state` | the agent's own end state: `gave_up` (window over or budget spent), `won`, `crashed`, `cancelled`, `not_started` |
| `scorecard_id`, `scorecard_runs` | the rollout's card and its runs of the game (a full reset starts another run) |
| `timing_valid`, `timing_error`, `clock_offset_seconds`, `clock_error` | the timing check on the server's recording; the game's first server event on the run clock (null without the origin, `clock_error` says so) |
| `result_valid` | the attempt counts as a complete, verified result: not cancelled, timing valid, no `agent_error` / `model_error` / `persistence_error` |
| `mask_sample`, `failure_kind`, `failure_reason` | set when `model_error` is: the score does not measure the model |
| `actions_per_level`, `level_clear_steps` | from the server's recording: the actions taken on each level (the clearing action counts on the level it clears), and the game-wide action number (1-based) at which each cleared level was first cleared; null when the recording cannot be read |

`aggregate_metrics` reports the counts (`received_games`: distinct pinned game ids, `received_rows`, `valid_results`, `not_started`, `crashed`, `cancelled`, `timing_invalid`, `persistence_errors`, `model_errors`). `rhae` / `rhae_<h>h` are the mean over games of each game's mean over its rollouts (repeats never weigh a game more). Rollouts that failed in Gym never reach it, so the received games can be fewer than the evaluation's:
- with `expected_games` set in the server config (e.g. `++arc_agi_3_resources_server.resources_servers.arc_agi_3.expected_games=25` for public25), `rhae` / `rhae_<h>h` are reported only when that many distinct games are there with valid results (`complete: true`; a duplicated game does not stand in for a missing one);
- without it, they are reported when every received row is valid, and they are a statistic of the received games only (`score_over: received_games`, `complete: null`).

The benchmark table comes from `summarize_rhae.py --expected`.

## Run

See the [agent README](../../responses_api_agents/arc_agi_3_agent/README.md) for the full launch (model server, agent, GB200/GB300). Summary of one or more runs over the expected game set:

```bash
python resources_servers/arc_agi_3/summarize_rhae.py --expected resources_servers/arc_agi_3/data/public25.jsonl \
    results/arc_agi_3/<run_name> [<run dir> | <benchmark.json> | <rollouts.jsonl> ...] --hours 1 2 [--json out.json]
```

What it reads and reports:
- **Run directory:** scores come from its `benchmark.json` (the run clock, as the kit's `rhae.py`; readable during play), with the verifier's flags from `rollouts.jsonl` and Gym's `rollouts_failures.jsonl`.
- **Completeness:** a column gets a score only when every expected game has a valid value; otherwise it reads `INCOMPLETE n/N`. Nothing is zero-filled.
- **Final or not:** a value is final when the game's attempt has ended, or (for `@T`) when the snapshot was saved at or after run start + T. A run directory with a `rollouts.jsonl` also needs the verifier's row for each game (`not_verified` otherwise). A column with all values present but not all final reads `<value> not final`: the score so far of a live or aborted run. `complete` is true only when every column is final.
- **Rollouts JSONL:** a row needs a real `clock_offset_seconds` (finite, non-negative) for its timed scores; otherwise it is flagged `no run clock`.
- **Counts:** missing, not started, cancelled, crashed, invalid timing, agent and persistence errors, and failed rollouts (a game whose scorecard disagrees with its agent fails its rollout).

## Results

Measured on 2026-11-04 and 2026-11-05 with an earlier revision of this environment and the Kaggle kit's harness (source commit 30d2cc8b, see the agent README).
- **Setup:** public25, all 25 games at once, 7200 s of play, 2 seeds per config (engine seed = run seed). One GB300 node (4 GPUs) per run, interactive Slurm; vLLM v0.32.0, TP4, MTP 3.
- **Score:** RHAE (0-100) from the kit's `rhae.py` on each run's `benchmark.json` (one run clock). Every cell has 25/25 games and 120 min of play. Δ is against the B0 mean.
- **B0** = the kit harness's deployed settings: temperature 0.7, top_p 0.95, top_k 20, 24576 tokens of kept history, thinking on, 12288 output tokens.

### Nemotron 3.5 Super VL (NVIDIA-Nemotron-3.5-Super-VL-120B-A12B, BF16 unless noted)

| config | change vs B0 | s1 @1h | s2 @1h | mean @1h | Δ @1h | s1 @2h | s2 @2h | mean @2h | Δ @2h | levels in 2 h (s1 / s2) |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | — | 2.01 | 1.80 | 1.90 | | 2.03 | 1.98 | 2.00 | | 22 / 20 |
| S1 | temperature 1.0, top_p 1.0, no top_k | 1.77 | 2.07 | 1.92 | +0.02 | 1.80 | 2.24 | 2.02 | +0.02 | 32 / 30 |
| S2 | temperature 0.6, top_p 0.95, no top_k | 1.60 | 2.11 | 1.86 | −0.05 | 1.64 | 2.11 | 1.87 | −0.13 | 22 / 23 |
| S3 | temperature 1.0 | 2.96 | 1.89 | 2.42 | +0.52 | 3.12 | 1.89 | 2.50 | +0.50 | 34 / 21 |
| C1 | keep 12288 | 3.03 | 2.17 | 2.60 | +0.69 | 3.94 | 3.60 | **3.77** | **+1.77** | 37 / 34 |
| C2 | keep 49152 | 1.51 | 1.36 | 1.44 | −0.47 | 1.69 | 1.37 | 1.53 | −0.47 | 19 / 20 |
| C3 | keep 98304 | 1.45 | 1.72 | 1.58 | −0.32 | 1.45 | 1.75 | 1.60 | −0.40 | 20 / 22 |
| R1 | thinking off | 0.79 | 0.24 | 0.51 | −1.39 | 0.79 | 0.24 | 0.51 | −1.49 | 18 / 20 |
| R2 | thinking token budget 8192 | 2.29 | 3.06 | 2.68 | +0.78 | 2.32 | 3.07 | 2.69 | +0.69 | 25 / 23 |
| P1 | NVFP4 weights | 1.90 | 2.17 | 2.04 | +0.14 | 1.90 | 3.28 | 2.59 | +0.59 | 19 / 27 |
| TC0 | no turn compaction (`harness_env.ARC3_TURN_COMPACTION=0`) | 1.74 | 3.20 | 2.47 | +0.57 | 1.88 | 3.22 | 2.55 | +0.55 | 21 / 23 |
| TH | drop past reasoning (`truncate_history_thinking`) | 1.19 | 1.07 | 1.13 | −0.77 | 1.22 | 1.07 | 1.14 | −0.86 | 19 / 23 |

- C1 is the best config: both seeds are above both B0 seeds, and it still gains from 1 h to 2 h. R2 is the only other config with both seeds above both B0 seeds.
- Thinking off (R1) and dropping past reasoning (TH) are below B0. Sampling (S1-S3), NVFP4 (P1) and TC0 are within seed noise (the seed spread inside a config is up to 1.4).
- The absolute level is low: most games clear 0-1 levels. Nemotron's transcripts show many turns without an action and looping reasoning, with no integration errors.

### Drain summaries

When the harness drops old history (a drain), it asks the model for a summary of the dropped turns. Usable share (written summaries in the request logs against `drain summary unusable` warnings; 17 of the 24 Nemotron runs counted):
- Nemotron with thinking on (B0, S1-S3, C1-C3, R2, P1): 61-82 %;
- Nemotron R1 (thinking off): 99 %; TH: 100 %; TC0: no summary requests.

### Validity

- All 24 runs: 25/25 games, 120 min of play, one run clock, 0 replay mismatches. These runs used the earlier replay verifier, which replayed every recorded action in a fresh engine; the scorecard authority replaced it.
- That revision had a race in the snapshot save, fixed since. In 11 of the 24 runs, one game's save raced. For all 600 games, the `benchmark.json` entry agrees with the game's event log in levels and actions, so the kit scores above are intact. Gym's own rollout rows read 0 levels for some raced games, which made Gym's own aggregate low by up to 0.22. The current code saves snapshots under a lock, and `summarize_rhae.py` reports such a run as `INCOMPLETE` instead of scoring it.

## Tests

```bash
ARC3_ENVIRONMENTS_DIR=/path/to/environment_files pytest resources_servers/arc_agi_3/tests
```

The tests on the real game server skip when the game files are absent.

## Licensing

- Code: Apache 2.0.
- Game files: ARC Prize Foundation. The 2 committed example games (cn04, ls20) are MIT; the other public25 games come from the ARC Prize 2026 ARC-AGI-3 Kaggle competition and are not redistributed.
