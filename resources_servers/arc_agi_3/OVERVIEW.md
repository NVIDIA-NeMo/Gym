# ARC-AGI-3 in NeMo Gym: overview

[ARC-AGI-3](https://arcprize.org/arc-agi/3/) is a set of interactive, turn-based games: the agent sees a 64×64 board,
sends actions (keys, clicks, RESET) and has to discover each level's rules and goal by itself. NeMo Gym has two
environments for it, one for whole games and one for short windows from recorded decision points.

| Environment | What a rollout is | Reward | Use | Resources server | Agent(s) | Config |
|---|---|---|---|---|---|---|
| **Whole games** | one full game, played for a fixed time with a real harness | the official scorecard's score / 100; RHAE@1h / @2h on the server's clock | evaluation (public25), RL | [`resources_servers/arc_agi_3`](README.md) | [`arc_agi_3_agent`](../../responses_api_agents/arc_agi_3_agent/README.md) (the Kaggle kit's harness, default) | [`arc_agi_3.yaml`](configs/arc_agi_3.yaml) |
| | | | | | [`arc_agi_3_official_agent`](../../responses_api_agents/arc_agi_3_official_agent/README.md) (the official ARC Prize harness, `provider` or `standard`) | [`arc_agi_3_official.yaml`](configs/arc_agi_3_official.yaml) |
| **Pivot windows** (PivotRL) | a recorded game restored exactly at one model call (game, harness state, live Python sandbox), then a few calls of the policy | oracle progress: the distance to completing the level at the pivot minus at the window's end | RL | [`resources_servers/arc_agi_3_pivot`](../arc_agi_3_pivot/README.md) | [`arc_agi_3_pivot_agent`](../../responses_api_agents/arc_agi_3_pivot_agent/README.md) | [`arc_agi_3_pivot.yaml`](../arc_agi_3_pivot/configs/arc_agi_3_pivot.yaml) |

## Whole games

- **The game server is the authority.** The resources server runs `arc_agi`'s own game server on the pinned game files.
  Every harness plays through it with one scorecard per rollout, and `/verify` scores the card and checks the agent's
  own report against it.
- **Two harness families**, the same rows, game server and scoring:
  - **Kaggle kit harness** (`arc_agi_3_agent`): the harness of the ARC-AGI-3 Kaggle kit, installed from its public
    Kaggle datasets (`kaggle:eladsarafian/arc3-c237-j30vfh-20261102`) and checked file by file. It runs unmodified
    inside the agent; only its model calls are routed to the Gym model server.
  - **Official ARC Prize harness** (`arc_agi_3_official_agent`): [arc-agi-3-benchmarking](https://github.com/arcprize/arc-agi-3-benchmarking)
    at a pinned commit, unmodified, in its own venv. `harness_type: provider` keeps every earlier reply with its
    reasoning and compacts with a summary; `standard` keeps a rolling text window.
- **Data:** `data/example.jsonl` (five rows over the 2 committed games, cn04 and ls20) runs with no download; `create_dataset.py` builds
  public25 from the Kaggle competition's game files.

## Pivot windows

- **A row** is a pivot: one model call of a recorded attempt, from a pivot pool (`games.parquet`, the harness env files
  and the recordings: every model request the harness sent). `record.py` records attempts and adds pivots to a pool.
  `data/example.jsonl` holds 5 pivots of 2 games from the committed example pool (recorded against a scripted stub
  model).
- **Restore:** each window runs the recorded harness, unmodified, in a worker process (Python 3.12). Up to the pivot,
  the worker answers the harness's requests with the recorded replies, after checking every request, board and action
  count against the recording; the restored state is the harness's own. A pivot that does not restore exactly is
  masked, never scored.
- **The window:** from the pivot on, every request the harness sends is the policy's turn. It ends at the level's
  completion, 12 actions or 24 calls (configurable), or a cost cap. A window is one chain of calls (each request
  extends the previous one plus its reply), so training-token capture sees one chain per rollout.
- **Reward:** the games of the pool carry a `policy()` that plays a shortest completion; the oracle distance is how
  many actions it needs from the current state.

## Run a smoke

Any OpenAI-compatible endpoint works if it parses tool calls (`qwen3_coder`) and splits reasoning. For example,
Qwen3.8-Flash-Next (NVFP4) on SGLang:

```bash
python3 -m sglang.launch_server --model-path <Qwen3.8-Flash-Next-NVFP4> --served-model-name qwen3.8-flash-next \
    --tp 2 --context-length 131072 --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --moe-runner-backend flashinfer_cutlass --trust-remote-code --port 30010
```

Pivot windows (the 5 example pivots, twice each):

```bash
COMMON=(--config resources_servers/arc_agi_3_pivot/configs/arc_agi_3_pivot.yaml
        --config responses_api_models/vllm_model/configs/vllm_model.yaml
        ++policy_base_url=http://<host>:30010/v1 ++policy_api_key=EMPTY ++policy_model_name=qwen3.8-flash-next)
gym env start "${COMMON[@]}" &   # wait for "servers ready"
gym eval run --no-serve "${COMMON[@]}" --agent arc_agi_3_pivot_agent \
    --input resources_servers/arc_agi_3_pivot/data/example.jsonl --num-repeats 2 --concurrency 8 \
    --output results/pivot_rollouts.jsonl
```

Whole games (the 2 example games for 10 minutes; for the official harness use `arc_agi_3_official.yaml`, agent
`arc_agi_3_official_agent` and `++<agent>.harness_type=provider|standard`):

```bash
A=arc_agi_3_agent.responses_api_agents.arc_agi_3_agent
COMMON=(--config resources_servers/arc_agi_3/configs/arc_agi_3.yaml
        --config responses_api_models/vllm_model/configs/qwen_vl_arc3_kit.yaml
        --model-url http://<host>:30010/v1 --model-api-key EMPTY ++policy_model_name=qwen3.8-flash-next
        ++arc_agi_3_resources_server.resources_servers.arc_agi_3.expected_games=2
        ++$A.play_seconds=600 ++$A.num_attempts=2 ++$A.concurrency=2 ++$A.run_name=smoke)
gym env start "${COMMON[@]}" &   # wait for "servers ready"
gym eval run --no-serve "${COMMON[@]}" --agent arc_agi_3_agent \
    --input resources_servers/arc_agi_3/data/example.jsonl --concurrency 2 --output results/arc3_rollouts.jsonl
```

First start: each environment installs what it runs (the harness, a Python 3.12 worker venv for pivot windows, the
official harness's venv) with `uv`. No Kaggle account is needed: the Kaggle kit's harness is a public Kaggle dataset,
downloaded without a token and checked file by file (a token in `KAGGLE_API_TOKEN` or `~/.kaggle/access_token` is used
when present). Only the full public25 game set needs one, to download the competition's files. Linux only.

## Status (2026-11-19)

Smoke on the final tree with Qwen3.8-Flash-Next (NVFP4, upstream SGLang, 2× RTX PRO 6000), Gym on another node:

| Run | Result |
|---|---|
| Pivot windows, 5 example pivots × 2 | 10/10 windows valid, none masked; mean reward 9.0; 2/10 completed the level |
| Whole games, Kaggle harness, 2 games × 10 min | 2/2 valid; each game completed level 1 (RHAE 4.8 and 3.3 at 10 min) |
| Whole games, official harness (`provider`), 2 games × 10 min | 2/2 valid, but no actions: 3 model calls per game in 10 min; this model reasons until its 12,288-token output cap and the harness rejects the empty reply (the Kaggle harness closes such a reply and makes it act) |
| Whole games, Kaggle harness, a dead model endpoint (HTTP 404 on every call) | 2/2 masked (`model_error`, failure kind `model_request_error`), not scored as 0-action games |

Earlier full runs (public25, 2 h, GB300, Nemotron 3.5 Super VL): see [the whole-games results](README.md#results).

Pivots after the harness compacted its context restore only from recordings that keep each reply's token usage, which
decides where it compacts (see the pivot README's known limit).

Tests: `pytest resources_servers/arc_agi_3/tests responses_api_agents/arc_agi_3_agent/tests
responses_api_agents/arc_agi_3_official_agent/tests resources_servers/arc_agi_3_pivot/tests
responses_api_agents/arc_agi_3_pivot_agent/tests`. The tests that need the harness, the game files or a GPU skip
without them; `resources_servers/arc_agi_3_pivot/tests/test_worker_live.py` restores the example pivots with the real
harness and checks that continuing with the recording's own replies reproduces its next requests.

## Training with NeMo RL

Both environments run under NeMo RL's Gym integration. For pivot windows, set:

- `token_capture.enabled: true` (the window's first call carries recorded history; capture roots it as prompt context);
- `grpo.masked_reward_policy: exclude` (masked windows must not enter the group baseline);
- train on rows that restore (with `forest_training: true`, NeMo RL selects captured calls on every row, and a window
  that never reached the model has none).

The pivot agent returns every call's served output items, so `selection_policy: returned_history` matches the captured
calls, and `reward` equals its single component (progress). Details: [pivot windows, training tokens](../arc_agi_3_pivot/README.md#training-tokens).

## Reasoning effort

Every environment can set the policy's thinking effort (e.g. `low` / `high` / `max`), sent as a
chat-template kwarg on thinking requests:

- whole games: `reasoning_effort` on the agent (`arc_agi_3_agent` or `arc_agi_3_official_agent`), one effort per agent
  server;
- pivot windows: per row (`create_dataset.py --reasoning-effort`) or on the server (`reasoning_effort`); unset, a
  window keeps the effort its pivot was recorded with (from the recorded requests, or the pool's `thinking_effort`
  column when the collector set it outside the harness). The replay before the
  pivot always uses the recorded effort, so recordings made at any effort restore.

The kwarg is `reasoning_effort` by default; `reasoning_effort_kwarg: thinking_effort` targets a template that reads
`thinking_effort` instead.

## Known limits

- **One configuration per agent process:** the Kaggle harness reads its settings at import.
- **Throughput:** the pivot server keeps its windows in one process (`num_workers: 1`), with one worker process per
  window (a restore takes about 2-3 s for early pivots).
- **Context:** set the model's window in the configs. The defaults are 131,072 tokens for the Kaggle harness, and
  262,144 for the official harness.
- **SGLang:** works as served above. Gym's agents send tool calls with their OpenAI fields only, so SGLang's extra
  `index` field is not echoed back.

## Licensing

- Code: Apache 2.0.
- The 2 committed example games (cn04, ls20): ARC Prize Foundation, MIT. The other public25 games come from the ARC
  Prize 2026 ARC-AGI-3 Kaggle competition and are not redistributed.
- The official harness (arc-agi-3-benchmarking): MIT, ARC Prize; installed at a pinned commit, not vendored.
