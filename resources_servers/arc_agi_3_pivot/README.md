# ARC-AGI-3 pivot windows (PivotRL)

Overview of all the ARC-AGI-3 environments and agents, with a smoke run: [`resources_servers/arc_agi_3/OVERVIEW.md`](../arc_agi_3/OVERVIEW.md).

RL from recorded decision points of ARC-AGI-3 games. A row is a **pivot**: one recorded model call of a game played
with the Kaggle kit's harness. A rollout restores the game at that call, lets the policy continue it for a short window, and is
rewarded by how much closer to completing the level it got. The agent is
[`responses_api_agents/arc_agi_3_pivot_agent`](../../responses_api_agents/arc_agi_3_pivot_agent/README.md).

## How a window runs

Each window runs the recorded harness itself, unmodified, in a worker process (`worker.py`). The harness's solver plays
the game (a local copy from the pool's `games.parquet`), and only its model endpoint is replaced: every model request it
sends comes to the window.

- **Restore (replay).** The worker starts the game from the beginning and answers each of the harness's requests up to the
  pivot with the recorded reply. Before answering, it checks that the request is the recorded one (the sha256 of the POST
  body; object addresses a cell printed may differ), and that the board and the action count are the recorded ones. The
  harness therefore re-runs every recorded Python cell and game action itself, and the state at the pivot (game, host
  state, live Python sandbox) is its own. A pivot whose replay leaves the recording is masked
  (`environment_prepare_failed`), never scored.
- **The window.** From the pivot on, each request the harness sends is the policy's turn: `/seed_session` returns the
  pivot request, and each `/step` hands the policy's reply to the harness and returns its next request.

What a replay needs from the recording, applied as configuration (the harness's code is not changed):

| What | Why |
|---|---|
| The sandbox seed (hash seed and RNGs) the recording ran with | cells that use randomness print the same |
| A virtual run clock, set from the clock values the recorded tool results show | the harness prints the run clock in every action result; live calls advance it by `clock_call_seconds` + `clock_token_seconds` per generated token |
| The sandbox memory limit and retry-guard cell counter the recorded tool results show | the harness reports both to the model |
| The recorded names of the sandbox's temporary directories | a path can reach the model |
| Each reply's `usage` and `finish_reason` when the recording kept them (`record.py` keeps them); else its token count estimated from the text, clamped to the recorded turn ends | the harness ends a turn after `LOCAL_ANALYZER_YIELD_TOKENS` generated tokens, and calibrates its context estimate from `prompt_tokens` |
| A long sandbox timeout (`replay_timeout_s`) for cells that finished in the recording | the recorded timeouts were wall clock on Kaggle's CPUs |
| The Gym engine settings (`GYM_ENGINE_ENV` of the `arc_agi_3` agent: no GDN state ownership, no CPU retention) | a recording made on the Kaggle kit's engine carries engine-only request fields (`gdn_ownership`), which the request check leaves out |

The harness's helper requests are answered from where the recording shows their replies: a reply cut off inside its
reasoning (`finish_reason` length) and its force-act continuation (the tool calls of the closed reply), a level summary
(the text later openers show under the level's summary heading), and a history compaction (the text the next request
shows under "[Summary of earlier history ...]").

**Known limit.** Where the harness compacts or trims its context depends on its token estimate, which it calibrates from
each reply's `prompt_tokens`. A recording without `usage` does not have it, so a pivot after the attempt's first
compaction may not restore (masked). Recordings that keep each reply's `usage` (as `record.py` writes them) restore
past it.

When a replayed cell printed a clock value that an earlier call read, the restore runs again with that read pinned to the
recorded value (up to `clock_repasses` times).

## Endpoints

| Endpoint | What |
|---|---|
| `/seed_session` | Restore the row's pivot and return its request: `messages`, `tools`, `chat_template_kwargs`. A pivot that cannot be restored returns `restore_error` (the rollout is masked). |
| `/step` | One model reply (an assistant chat message) -> the harness's next request (`request_messages`, and `messages`: the part after the reply), `done`, and the call's record (actions, level completed, game over, tool error, oracle distance before and after, why the window ended). |
| `/verify` | Reward and components, then the window is released. |

## Window and reward

The window ends, at a call boundary, at the first of: the level completed, `max_actions` game actions (default 12),
`max_calls` model calls (default 24), `dollar_cap` (when set), or the game's end. Game over does not end it: the harness
resets, as in a full run.

A window is one chain of calls: every request extends the previous one plus its reply, as the policy generated it. When
the harness's next request does not (it compacted or trimmed its context, sent a helper request such as a summary,
continued a reply cut off at the output limit, or holds the reply changed), the window ends and is masked
(`model_context_limit`), not scored.

The oracle distance: the pool's games are written as pure functions with a `policy(h, z, level)` that plays a shortest
completion of the level (`oracle.py`). The distance is how many actions that policy needs from the current state.

| Field | Meaning |
|---|---|
| `reward` = `progress` | `d0 - d_end`: the oracle distance at the pivot minus at the window's end (`d0` when the level completes) |
| `level_completed`, `actions`, `calls`, `tool_errors`, `gen_tokens` | the window's measures (`reward_components` holds progress only) |
| `dollars`, `uncached_tokens`, `cached_tokens`, `dollar_cap_reached` | the window's cost |
| `d0`, `d_end` | the oracle distances |
| `reasoning_effort` | the effort the live requests carried (null: the template's default) |

Reasoning effort: the replay gives each request its recorded effort (a recording whose `chat_template_kwargs`
carry `reasoning_effort` / `thinking_effort` set outside the harness restores too). The window's live thinking requests
carry the row's `verifier_metadata.reasoning_effort`, else the server's `reasoning_effort`, else the pivot's recorded
effort, under the recording's effort kwarg (default `reasoning_effort`). The recorded effort is the recorded requests'
own, else the pool row's `thinking_effort` column (a collector that set the effort past the harness's request, e.g. a
proxy in front of the model), else none: the template's default. The
agent sends it with the request's chat-template kwargs, and `/verify` reports it (`reasoning_effort`). To train on several
efforts, write a dataset per effort (`create_dataset.py --reasoning-effort low`) and mix the rows.

Cost: each call costs `(uncached * in + cached * cached + decoded * out) / 1e6` USD at `prices` (USD per 1M tokens,
defaults 0.20 / 0.016 / 0.47). `cached` is the part of the prompt the previous call of the window already holds (its
prompt + reply). The agent measures it on the token ids when the model server returns them; otherwise the server takes
it as the previous call's prompt + reply tokens. The window's first call counts as all cached.

Masked: a restore error (`environment_prepare_failed`), a window whose next request left the chain
(`model_context_limit`), a window cut short by a failed model call (`model_request_error`), and a window the oracle
cannot measure (`verifier_error`: no distance at the pivot or at the window's end, the oracle's search cap or a loop;
distance zero counts only when the game completed the level).

The worker: each window's harness runs in a child process under a minimal environment (`PATH`, `HOME`, `LANG`,
`LC_ALL`, `TMPDIR`, `TZ`, then the recorded harness settings; no credentials of this server), in its own process
group. Every answer it owes is bounded by `worker_timeout_s` (default 3600 s); a worker that does not answer is killed
with its sandbox processes and the window is masked (`environment_prepare_failed` at the restore,
`environment_server_failed` during the window). The pool's paths are checked before use: a row's recording and
harness env file must resolve inside `pool_dir`, a `games.parquet` entry must name `<game>/<version>/<game>.py` of its
own game and a sha256 `source_sha256`, and a row that pins a game source sha256 restores only that revision.

## Setup

`setup_worker.py` installs what the windows run on into `.runtime/` here on the first start:

| What | Source (config) | Notes |
|---|---|---|
| The harness | `harness_source`, default `kaggle:eladsarafian/arc3-c237-j30vfh-20261102` (the Kaggle kit's harness, a public Kaggle dataset: no token needed). A directory or a tarball also works. | installed and checked file by file by the `arc_agi_3` agent's `setup_harness.py`: the harness that agent plays |
| An existing install | `harness_root` | a directory with `harness/taaf/src` and `config/` |
| The worker interpreter | `worker_python: "3.12"` (default) | a uv venv with the harness's runtime (the `arc_agi_3` agent's dependencies); a path to an interpreter, or `null` for this server's |

The example pool was recorded with the Kaggle kit's harness, whose source commit `30d2cc8b` is the server's
`harness_revision` (rows recorded under another revision are refused). Every restore checks each request against the
recording, so a difference would mask the pivot, never score it.

**Python 3.12.** Pivots recorded under 3.12 (the Kaggle kit) restore exactly only under 3.12 when the harness's retry
guard fired before the pivot: the guard identifies code by a hash of `ast.dump`, whose output changed in 3.13. The
harness's Python sandbox runs under the worker's interpreter (`sandbox_python` to use another). Linux only.

**Pool** (`pool_dir`): `games.parquet` (each game's code with its oracle), `env/` (the harness env files the attempts
were recorded under, checked by sha256) and `snap-raw/` (the recordings: every model request of an attempt). Rows name
these files relative to the pool directory.

- The default is the committed example pool, [`data/example_pool`](data/example_pool/README.md): 2 games (cn04, ls20),
  5 pivots, rows in `data/example.jsonl`.
- **Recording**: `record.py` plays a game of the pool from its start in the window's worker, against any
  OpenAI-compatible endpoint, and writes every request the harness sends (`snap-raw/<attempt>.snap.jsonl`, with the
  board, action count and each reply's `usage`). It then restores each chosen pivot by replay in a fresh worker, as the
  server does, and adds its row to a pool split (`po_d0`: the oracle distance there; `restore_ok`: the restored request
  is the recorded one). Run it from the Gym root with this server's venv (Linux):

```bash
python -m resources_servers.arc_agi_3_pivot.record <pool_dir> --game cn04-2fe56bfb \
    --harness-env env/kit-harness_env.toml --model-url http://<host>:8000/v1 --model <served name> \
    --calls 24 --pivots 4,8,12 --split train.parquet
```

- **Dataset**: one Gym row per pivot from a pool split; rows that restored (`restore_mode` snapshot, `restore_ok`) are
  kept:

```bash
python resources_servers/arc_agi_3_pivot/create_dataset.py <pool_dir>/eval.parquet \
    --output resources_servers/arc_agi_3_pivot/data/eval.jsonl --per-game 4
```

## Run

```bash
R=arc_agi_3_pivot_resources_server.resources_servers.arc_agi_3_pivot
COMMON=(--config resources_servers/arc_agi_3_pivot/configs/arc_agi_3_pivot.yaml
        --config responses_api_models/vllm_model/configs/vllm_model.yaml
        ++$R.pool_dir=<pool_dir>)
gym env start "${COMMON[@]}" &
gym eval run --no-serve "${COMMON[@]}" --agent arc_agi_3_pivot_agent \
    --input resources_servers/arc_agi_3_pivot/data/eval.jsonl --num-repeats 4 --concurrency 32 \
    --output results/arc_agi_3_pivot_rollouts.jsonl
```

To run the example, drop the `pool_dir` override and use `--input resources_servers/arc_agi_3_pivot/data/example.jsonl`.

## Example data

`data/example.jsonl`, `data/example_metrics.json` and `data/example_rollouts.jsonl` are Gym's example-data contract
(`ng_test +entrypoint=resources_servers/arc_agi_3_pivot +should_validate_data=true`). The rows are the five pivots of
the example pool (`create_dataset.py data/example_pool/example.parquet --output data/example.jsonl`, see the
[pool's README](data/example_pool/README.md)); the metrics are `gym dataset collate
"+config_paths=[resources_servers/arc_agi_3_pivot/configs/arc_agi_3_pivot.yaml]" +output_dirpath=<tmp>
+mode=example_validation`. The rollouts are a CPU run of the real servers (this server and its harness workers,
`arc_agi_3_pivot_agent`, the Gym vLLM model server) against the scripted model of
`responses_api_agents/arc_agi_3_agent/tests/stub_model.py` (one action per call), so their rewards are a stub's:

```bash
python responses_api_agents/arc_agi_3_agent/tests/stub_model.py --port 8765 --log /tmp/stub.jsonl &
COMMON=(--config resources_servers/arc_agi_3_pivot/configs/arc_agi_3_pivot.yaml
        --config responses_api_models/vllm_model/configs/qwen_vl_arc3_kit.yaml
        --model-url http://127.0.0.1:8765/v1 --model-api-key dummy --model stub)
gym env start "${COMMON[@]}" &   # wait for "servers ready!"
gym eval run --no-serve "${COMMON[@]}" --agent arc_agi_3_pivot_agent \
    --input resources_servers/arc_agi_3_pivot/data/example.jsonl --concurrency 5 \
    --output resources_servers/arc_agi_3_pivot/data/example_rollouts.jsonl
```

## Training tokens

The agent's model calls already use the rollout's capture route; nothing in the environment changes per trainer. What
needs care is the window's first call. Its request is the pivot's recorded request, which already holds assistant turns
this rollout did not generate. That history must become the chain's prompt context (loss mask 0), and the window's own
calls chain to it.

**NeMo RL (the default).** Turn on NeMo RL's token capture (`token_capture.enabled: true`, with vLLM or Megatron
generation). The policy's worker then stages the tokens, and Gym roots a first call with recorded history: a `text`
admission on the rollout's empty capture ledger. Both of Gym's parent modes do this:

- `token_capture.forest_training: false` uses `parent_resolution: fingerprint`;
- `token_capture.forest_training: true` uses `parent_resolution: source_history`, which roots it with
  `root_reason: first_call` and joins later calls by their source prefix.

**Gym's own capture** (another trainer, or `gym eval run` with capture on). By default the first call is unresolved and
the rollout is masked. Gym's `token_id_capture.root_seeded_first_call: true` (a Gym change proposed upstream) roots it:

```yaml
token_id_capture:
  enabled: true
  root_seeded_first_call: true
```

Either way, the chain needs each request to hold the previous reply exactly as generated. The server checks that the
harness's next request carries the policy's reply unchanged (content, reasoning, tool calls); otherwise the window ends
and is masked.

**NeMo RL recipe settings.**

- `token_capture.enabled: true` (the agent's output carries no token ids).
- `grpo.masked_reward_policy: exclude`: masked windows (a restore error, a window that left the chain) must not enter
  the group baseline. NeMo RL's default, `include`, counts their placeholder reward 0.
- The agent's response `output` is every call's served output items, in order (Gym's `completion_items` of each
  completion), so `selection_policy: returned_history` matches the captured calls.
- `reward` is `reward_components.progress` (NeMo RL checks that the reward is the sum of its components).
- Train on rows that restore: with `forest_training: true`, NeMo RL runs call selection on every row, and a window
  that never reached the model (a restore error) has no captured call.

## Tests

```bash
pytest resources_servers/arc_agi_3_pivot/tests
```

`test_app.py` runs the server against a stand-in worker, plus the recording, oracle, clock and setup pieces.
`test_worker_live.py` runs the example pivots through the server with the real harness (skipped unless
`ARC3_PIVOT_HARNESS_ROOT` and `ARC3_PIVOT_WORKER_PYTHON` are set): each pivot request must be the recorded one, and
continuing with the recording's own replies must reproduce its next requests.
