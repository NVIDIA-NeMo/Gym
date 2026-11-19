# ARC-AGI-3 in NeMo Gym: overview

[ARC-AGI-3](https://arcprize.org/arc-agi/3/) is a set of interactive, turn-based games: the agent sees a 64×64 board,
sends actions (keys, clicks, RESET) and has to discover each level's rules and goal by itself. NeMo Gym plays whole
games of it, with two harness families.

| Environment | What a rollout is | Reward | Use | Resources server | Agent(s) | Config |
|---|---|---|---|---|---|---|
| **Whole games** | one full game, played for a fixed time with a real harness | the official scorecard's score / 100; RHAE@1h / @2h on the server's clock | evaluation (public25), RL | [`resources_servers/arc_agi_3`](README.md) | [`arc_agi_3_agent`](../../responses_api_agents/arc_agi_3_agent/README.md) (the Kaggle kit's harness, default) | [`arc_agi_3.yaml`](configs/arc_agi_3.yaml) |
| | | | | | [`arc_agi_3_official_agent`](../../responses_api_agents/arc_agi_3_official_agent/README.md) (the official ARC Prize harness, `provider` or `standard`) | [`arc_agi_3_official.yaml`](configs/arc_agi_3_official.yaml) |

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

## Run a smoke

Any OpenAI-compatible endpoint works if it parses tool calls (`qwen3_coder`) and splits reasoning. For example,
Qwen3.8-Flash-Next (NVFP4) on SGLang:

```bash
python3 -m sglang.launch_server --model-path <Qwen3.8-Flash-Next-NVFP4> --served-model-name qwen3.8-flash-next \
    --tp 2 --context-length 131072 --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --moe-runner-backend flashinfer_cutlass --trust-remote-code --port 30010
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

First start: each environment installs what it runs (the harness, the official
harness's venv) with `uv`. No Kaggle account is needed: the Kaggle kit's harness is a public Kaggle dataset,
downloaded without a token and checked file by file (a token in `KAGGLE_API_TOKEN` or `~/.kaggle/access_token` is used
when present). Only the full public25 game set needs one, to download the competition's files. Linux only.

## Status (2026-11-19)

Smoke on the final tree with Qwen3.8-Flash-Next (NVFP4, upstream SGLang, 2× RTX PRO 6000), Gym on another node:

| Run | Result |
|---|---|
| Whole games, Kaggle harness, 2 games × 10 min | 2/2 valid; each game completed level 1 (RHAE 4.8 and 3.3 at 10 min) |
| Whole games, official harness (`provider`), 2 games × 10 min | 2/2 valid, but no actions: 3 model calls per game in 10 min; this model reasons until its 12,288-token output cap and the harness rejects the empty reply (the Kaggle harness closes such a reply and makes it act) |
| Whole games, Kaggle harness, a dead model endpoint (HTTP 404 on every call) | 2/2 masked (`model_error`, failure kind `model_request_error`), not scored as 0-action games |

Earlier full runs (public25, 2 h, GB300, Nemotron 3.5 Super VL): see [the whole-games results](README.md#results).

Tests: `pytest resources_servers/arc_agi_3/tests responses_api_agents/arc_agi_3_agent/tests
responses_api_agents/arc_agi_3_official_agent/tests`. The tests that need the harness or the game files skip without
them.

## Reasoning effort

Every environment can set the policy's thinking effort (e.g. `low` / `high` / `max`), sent as a
chat-template kwarg on thinking requests:

- whole games: `reasoning_effort` on the agent (`arc_agi_3_agent` or `arc_agi_3_official_agent`), one effort per agent
  server;

The kwarg is `reasoning_effort` by default; `reasoning_effort_kwarg: thinking_effort` targets a template that reads
`thinking_effort` instead.

## Known limits

- **One configuration per agent process:** the Kaggle harness reads its settings at import.
- **Context:** set the model's window in the configs. The defaults are 131,072 tokens for the Kaggle harness, and
  262,144 for the official harness.
- **SGLang:** works as served above. Gym's agents send tool calls with their OpenAI fields only, so SGLang's extra
  `index` field is not echoed back.

## Licensing

- Code: Apache 2.0.
- The 2 committed example games (cn04, ls20): ARC Prize Foundation, MIT. The other public25 games come from the ARC
  Prize 2026 ARC-AGI-3 Kaggle competition and are not redistributed.
- The official harness (arc-agi-3-benchmarking): MIT, ARC Prize; installed at a pinned commit, not vendored.
