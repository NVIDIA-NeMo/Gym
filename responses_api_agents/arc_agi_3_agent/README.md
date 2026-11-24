# ARC-AGI-3 agent (Kaggle kit harness)

Overview of all the ARC-AGI-3 environments and agents, with a smoke run: [`resources_servers/arc_agi_3/OVERVIEW.md`](../../resources_servers/arc_agi_3/OVERVIEW.md).

This agent plays ARC-AGI-3 games with the harness of the ARC-AGI-3 Kaggle kit (public Kaggle datasets, 2026-11-02), against any model behind a NeMo Gym model server. Each agent session plays one game (one row of [`resources_servers/arc_agi_3`](../../resources_servers/arc_agi_3/README.md)) for `play_seconds`. The configs pair the agent with the `single_agent_turn_legacy` environment server, which runs each row as one episode: it seeds the agent session with the row's pinned game (the row's `agent_context`), activates it (the whole game), closes it, then sends the activation's reply, which carries the game record and its scorecard, to the `arc_agi_3` resources server, which scores it (RHAE). A repeated seed starts nothing; the close stops a game still in play.

## What runs

| Part | Source |
|---|---|
| Harness (game loop, prompts, Python sandbox, history trimming, priority gate, ...) | the public Kaggle dataset `eladsarafian/arc3-c237-j30vfh-20261102-harness` v1 (source commit 30d2cc8b, `tool_agent.py` sha256 22a2ae7b). Pinned file by file in `harness_manifest.json`; not vendored (third-party code). |
| Harness settings | the kit's `harness_env.toml` (the `...-config` dataset, sha256 a49ab988): keep 24K, opener trim, force-act, timer mask, reset mode first, ... |
| Game engine | the resources server's official `arc_agi` game server (0.9.8 / `arcengine` 0.9.3, the pinned files): the harness plays through it with taaf's own ONLINE `ArcadeSpec`, one scorecard per game. The scorecard is the authority for the score. |
| Model calls | `gym_transport.py` (below) to the Gym model server |

The harness files are used byte-for-byte; `setup_harness.py` refuses any difference. The agent changes these things outside them:

1. **Model calls.** The harness posts OpenAI chat completions with `requests`. The agent replaces the harness module's `requests` with a proxy that sends the call to the Gym model server's `/v1/chat/completions` through Gym's aiohttp client:
   - non-OpenAI fields (`top_k`, `continue_final_message` + `add_generation_prompt: false`) go in `metadata.extra_body`;
   - `chat_template_kwargs` go in `metadata.chat_template_kwargs`;
   - past reasoning becomes a `<think>` prefix, which the Gym vLLM server (`uses_reasoning_parser: true`) splits back out;
   - the reply's reasoning goes back to `reasoning_content`;
   - a continuation of an already-closed reasoning block (turn compaction, force-act) gets its reply as content. vLLM's `nemotron_v3` / `qwen3` parsers start every reply in the reasoning state.
   - The model server must return the engine's context-length 400 (`vllm_model` `propagate_context_overflow_errors: true`, set in `nemotron_3_5_super_vl.yaml` and `qwen_vl_arc3_kit.yaml`) so that the harness trims its history and retries. The agent refuses to start otherwise (`require_context_error_passthrough`).
2. **Engine-specific switches off:**
   - `ARC3_GDN_OWNERSHIP` and `ARC3_GDN_CPU_RETENTION`: the kit's SGLang engine's state protocol;
   - `ARC3_WARMUP_ACTION_GAMES`: a Kaggle keepalive that adds one RESET action.
3. **One run clock, as the kit.** The origin is taken just before the first game starts. Every game's play ends at origin + `play_seconds` (the solver's soft end), and RHAE@T is measured from the origin:
   - `benchmark.json` `start_time` holds it in local time, as the kit;
   - the verifier gets it in UTC and measures the game server's own action times from it.

   Run with `--concurrency` ≥ the number of rows, or later games get less time.
4. **Play through the game server.** taaf's `GameAPI` gets `ArcadeSpec(operation_mode=ONLINE, arc_base_url=<the resources server's game server>)` instead of the offline files, which is taaf's own server path (its `competition_sim` plays this way). The harness files do not change.
   - taaf opens one scorecard per game and closes it at the end of the game.
   - The agent sends the card id, the game run's guid and its key to the verifier, which checks the record against the card.
   - Each action costs 2-10 ms more than in process.
5. **Attempt lifecycle.**
   - A game that cannot start (pin mismatch, engine refusal) is scored as `not_started` with `agent_error`.
   - Once a game starts, its record is always scored. A failed snapshot save is reported as `persistence_error`, and an exception outside the harness as `agent_error`.
   - Closing a session whose game is still in play (its activation was cancelled or failed) stops only that game: it ends as `cancelled` and its pending model call is abandoned.
   - The harness's sampling is the agent's configuration: an activation that sets `temperature`, `top_p` or `max_output_tokens` (e.g. the CLI's `--temperature`) is refused with a 422.
   - A game in which no model call got an answer (an unreachable or failing endpoint) is reported to the verifier as `model_error`: its result is invalid and the rollout masked, not scored as a 0-action game.
   - Server shutdown stops every game, drains them within 60 s, finishes them as `cancelled`, writes the last snapshot and closes the solver.
   - The harness start belongs to the agent, not to the request that triggered it: a request cancelled during the start leaves it running, the next request waits for the same start, and shutdown waits for it and then closes it. A start that fails tears down what it built; the next request starts again. The start has no time limit of its own: shutdown waits for it however long it takes.

The harness is Linux only (its Python sandbox uses `prlimit`).

## Hyperparameters

The harness reads its settings once per process, so one agent server is one configuration. Set them in the agent config (`++arc_agi_3_agent.responses_api_agents.arc_agi_3_agent.<field>=...`). The config is the only source: at start the agent removes harness settings inherited from the shell that the config does not set (`ARC3_*`, `LOCAL_ANALYZER_*`, `TAAF_*`, ...) and lists their names in `run_config.json` (`harness_env_cleared`). The `gym eval run` flags `--temperature`, `--top-p` and `--max-output-tokens` only change the rows' `responses_create_params`, which this agent does not use.

| Field | Harness variable | Deployed value (null = keep) |
|---|---|---|
| `temperature`, `top_p`, `top_k` | `LOCAL_ANALYZER_TEMPERATURE/TOP_P/TOP_K` | 0.7, 0.95, 20 |
| `seed` | the harness's Python `random`; seed the engine with vLLM `--seed` | unset |
| `request_seed` | `LOCAL_ANALYZER_SEED`: the same sampling seed on every request | unset (keep it unset: a rolled-back turn is re-sent after a failure, and a fixed per-request seed gives it the same reply) |
| `enable_thinking` | `LOCAL_ANALYZER_ENABLE_THINKING` | true |
| `max_output_tokens` | `LOCAL_ANALYZER_MAX_OUTPUT` | 12288 |
| `context_window_tokens` | `LOCAL_ANALYZER_CONTEXT_WINDOW` | 131072 |
| `context_keep_tokens` | `ARC3_CONTEXT_KEEP_TOKENS` (history size after a trim) | 24576 |
| `play_seconds` | per-game runtime limit | 7200 |
| `run_clock` | `shared`: one run clock (the kit), every game ends at the first game's start + `play_seconds`, RHAE@T from that origin. `per_game`: each game has `play_seconds` from its own start, RHAE@T from it (training) | `shared` |
| `max_actions_per_game` | the solver's action budget per game | unset (no limit) |
| `reset_timeout_s` | how long the arc_agi client waits for a game's RESET (the engine imports the game on its first RESET; the client's own timeout is 10 s). Games are made by full id (`<game>-<version>`), so the pinned version is played | 120 |
| `harness_env` | any other harness variable, applied last | `{}` |
| `chat_template_kwargs` | sent with every request; the harness's own values win. Model-specific defaults are in the model server config | `{}` |
| `extra_body` | extra vLLM request fields | `{}` |
| `reasoning_effort` | the thinking effort (e.g. `low` / `high` / `max`), sent as the chat-template kwarg `reasoning_effort_kwarg` on every thinking request (`enable_thinking` not false) that sets no effort of its own: the harness's effort ladder (`ARC3_REASONING_EFFORT_LADDER`, after a truncated reply) wins. Unset: the template's default | unset |
| `reasoning_effort_kwarg` | the kwarg the model's template reads: `reasoning_effort` or, e.g., `thinking_effort` | `reasoning_effort` |

Notes:

- **`max_output_tokens` also caps the reasoning.** When a reply hits the cap inside its reasoning, the harness (`ARC3_FORCE_ACT`) closes the reasoning and continues the reply to a tool call.
- **Past reasoning stays in the prompt.** The harness sends `preserve_thinking: true` (the kit's Qwen template). Nemotron 3.5's template drops past reasoning by default; `nemotron_3_5_super_vl.yaml` sets `truncate_history_thinking: false` to keep it.
- **Medium reasoning effort** on Nemotron: `chat_template_kwargs: {medium_effort: true}` (the model config keeps `truncate_history_thinking: false`).

## Training (end-to-end RL)

The same agent serves training rollouts: one rollout plays one game from level 0 with the deployed harness, history
drains included. Set `run_clock: per_game` (rollouts start whenever the trainer sends them) and a budget
(`play_seconds`, `max_actions_per_game`); repeats of one game (a GRPO group) play side by side, each with its own
files (`<game>_p<n>`).

- **Model calls.** Every harness call goes to the model server on the activation's rollout path (`url_path_for_request`), so
  Gym token capture applies to all of them. Enable it with `token_id_capture: true` on the agent and the run-wide
  `token_id_capture` settings ([external agent harnesses](../../fern/versions/latest/pages/training-tutorials/external-agent-harnesses.mdx)).
- **Several context chains per rollout.** A history drain drops old turns (and, with `ARC3_TURN_COMPACTION_MODE=drainsum`,
  inserts a summary the model wrote in a helper call), so the next gameplay request starts a new chain. The harness
  also makes helper calls of its own (`level-summary`, `drain-summary`, ...). Gym's local capture path delivers one
  chain per rollout; train on all of them with framework-owned reconstruction (`parent_resolution: source_history`,
  `external_staging: true`, `rebuild_response: false`, see
  [context compaction](../../fern/versions/latest/pages/model-server/model-call-capture.mdx)).
- **What the rollout reports** (`ng_agent_observations`, from the session close; `call_log.py`):
  - one `agent_invocation` whose `model_calls` are the gameplay calls in order: the accepted calls, as references
    (model server + response id) to the captured calls;
  - one `context_compaction` per drain: the gameplay call before and after it, the helper calls that wrote the
    summary, the summary text, and the prompt tokens before and after (`usage.prompt_tokens`);
  - gaps: `model_call_failed` for a failed call, `history_rewritten` for a gameplay request that rewrote earlier
    history without dropping any (also a new chain, not a compaction).
- **`agent_call_log`**: every model call sent, in order: `index`, `role` (`gameplay` / `helper`), `history_id`
  (`main`, `level-summary`, `drain-summary`, `turn-compaction`, `rolling-summary`), `status` (HTTP; 0 when no answer
  came), `response_id` (joins the captured call; none for a failed call), `error` (timeout, connection error,
  cancelled), `step` and `level` (the game when the call was sent: actions taken + 1, levels completed + 1),
  `started_at` / `completed_at` and `prompt_tokens`.
- **`harness_log`**: the warnings the harness logged while it played the game (`lines`: `logged_at`, `step`, `level`,
  `severity`, `logger`, `message`; the first 2,000, each cut at 1,000 characters; `dropped`: the lines over that).
  Some failures show only here, e.g. a drain summary the harness rejected as unusable (`plain drain kept`).
- The harness marks its calls itself (`_chat_completion`'s `gdn_role` and `gdn_history_id`); the agent wraps that one
  method to read them. The harness files stay unchanged.

## Launch on one GB200 / GB300 node

1. **Inputs** (small; copy them to the node):
   - the harness: by default `harness_source: kaggle:eladsarafian/arc3-c237-j30vfh-20261102` (the Kaggle kit's harness and config datasets, public: no token needed; a token in `KAGGLE_API_TOKEN` or `~/.kaggle/access_token` is used when present), or a directory or `.tar.gz` of an install (`harness/` and `config/`);
   - the 25 game folders (see the resources server README).

   ```bash
   python responses_api_agents/arc_agi_3_agent/setup_harness.py kaggle:eladsarafian/arc3-c237-j30vfh-20261102
   python resources_servers/arc_agi_3/create_dataset.py --environments-dir resources_servers/arc_agi_3/data/environment_files \
       --output resources_servers/arc_agi_3/data/public25.jsonl
   ```
2. **Model** (vLLM v0.32.0, TP4 on the node's 4 GPUs). Each model config has its serve line; Nemotron 3.5 Super VL per the model card:
   ```bash
   vllm serve nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 --served-model-name nemotron-3.5-super-vl \
     --tensor-parallel-size 4 --trust-remote-code --async-scheduling --enable-mamba-shared-prefix-checkpoint \
     --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 \
     --speculative_config.method mtp --speculative_config.num_speculative_tokens 3 \
     --max-model-len 262144 --max-num-seqs 32 --limit-mm-per-prompt '{"image": 512, "video": 0}' --port 8000
   ```
   The harness keeps many board images in the history (82 in one request of a 60 s test). The model card's `--limit-mm-per-prompt` (1 image) is too low. `--max-model-len` must be at least `context_window_tokens`: the harness keeps the reply's tokens inside that window (the kit served 139264 for 131072). For the NVFP4 checkpoint, serve `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` the same way.

   A Qwen reasoning VL model as the kit serves its checkpoints (`qwen3` reasoning parser, `qwen3_coder` tool parser, the model's `chat_template.jinja` with `preserve_thinking`, engine context 139264): use `responses_api_models/vllm_model/configs/qwen_vl_arc3_kit.yaml` and its serve line, and set `++policy_model_name=<served name>`.
3. **Gym** (on the same node; work and temporary directories on scratch, not `/home`):
   ```bash
   A=arc_agi_3_agent.responses_api_agents.arc_agi_3_agent
   COMMON=(--config resources_servers/arc_agi_3/configs/arc_agi_3.yaml
           --config responses_api_models/vllm_model/configs/nemotron_3_5_super_vl.yaml
           --model-url http://127.0.0.1:8000/v1 --model-api-key dummy
           ++arc_agi_3_resources_server.resources_servers.arc_agi_3.expected_games=25
           ++$A.run_name=b0_seed1 ++$A.seed=1)   # and vllm serve --seed 1
   gym env start "${COMMON[@]}" &   # wait for "servers ready!"
   gym eval run --no-serve "${COMMON[@]}" --agent arc_agi_3_agent \
     --input resources_servers/arc_agi_3/data/public25.jsonl --concurrency 25 --output results/arc3_public25_rollouts.jsonl
   ```
   `gym eval run` without `--no-serve` reads the config's datasets (`--split`), not `--input`.
   `expected_games=25` makes Gym's aggregate RHAE a public25 score: it is reported only when all 25 games have a valid result.
   A smoke on the 2 committed example games (no download): `--input resources_servers/arc_agi_3/data/example.jsonl --limit 2 --concurrency 2` (the first two rows: one per game), `num_attempts=2`, `concurrency=2`, `expected_games=2`, and a shorter `play_seconds`.
4. **Results:** complete runs only, one clock:
   ```bash
   python resources_servers/arc_agi_3/summarize_rhae.py --expected resources_servers/arc_agi_3/data/public25.jsonl \
       results/arc_agi_3/<run_name> --hours 1 2
   ```

## Outputs

- **Rollouts JSONL:** one row per game, with the verify fields (`reward`, `rhae`, `rhae_1h`, `rhae_2h`, `levels_completed`, `actions`, `scorecard_id`, `result_valid`, ...), `run_dir`, `run_config_sha256`, `harness` (`arc3-kaggle-kit`), `harness_revision` (the pinned harness's Kaggle datasets and source commit), `sampling` (the effective temperature, top_p, top_k, output cap, thinking switch, request seed, context window, and the effort the transport adds), `ng_agent_observations`, `agent_call_log` and `harness_log` (see Training).
- **`<output_dir>/<run_name>/`:**
  - `benchmark.json`: taaf format, rewritten every `save_every_s` and at every game end. The kit's `rhae.py` and `summarize_rhae.py` read it, also during play.
  - `transcripts/`, `artifacts/`, `solver_analysis/`;
  - `requests.jsonl` (with `save_request_logs`);
  - `telemetry/events.jsonl`;
  - `harness_env.json`: the exact harness environment;
  - `run_config.json`: the effective configuration, credentials redacted. It holds the agent fields, seed semantics, transport extras and template kwargs, harness provenance, the model server config, the engine's own `/v1/models` report (served model, `max_model_len`) and the game versions and sha256. Every rollout row and `benchmark.json` carry its sha256.

## Model-specific behavior (Nemotron 3.5 Super VL)

- **Some drain summaries fail.** When the harness drops old history (a drain), it asks the model for a summary of the dropped turns. Nemotron sometimes answers that request with a tool call instead of a summary; the harness then keeps a plain drain and logs `drain summary unusable (0 chars), plain drain kept`.
  - One 2 h public25 run (25 games, thinking on): 272 summaries written, 102 unusable.
  - The harness logs a written summary at INFO level, which the agent's log does not show; count them in the request logs (`<run>/*_requests.jsonl`, user messages that start with `[Summary of earlier history`).
  - `harness_env.ARC3_TURN_COMPACTION=0` turns the summary requests off (they cost one prefix-cached request each).
- **Force-act works.** Replies cut off inside their reasoning are closed and continued to a tool call.

## Model-specific behavior (Qwen reasoning VL, as the kit serves it)

Checked on a Qwen-type reasoning VL checkpoint (2 seeds, 25 games, 2 h each), served with the `qwen3` / `qwen3_coder` parsers and the model's chat template. The runs used `nemotron_3_5_super_vl.yaml` with the model name swapped; its model settings are the same as in `qwen_vl_arc3_kit.yaml`, which was added after these runs. What the transcripts and request logs show:
- every reply has its reasoning split out and a parsed tool call; no tool-call markup or `</think>` in the content;
- drain summaries are written in every game, 0 unusable;
- force-act continues to a tool call.

## Tests

```bash
pytest responses_api_agents/arc_agi_3_agent/tests
```

`test_smoke_cpu.py` plays 2 real games for `ARC3_SMOKE_PLAY_SECONDS` (default 45 s) through a real game server against a scripted model, and checks the scores against the scorecards. It needs Linux, the harness install and the game files (`ARC3_ENVIRONMENTS_DIR`); otherwise it skips. `tests/stub_model.py --port N --log FILE` serves the scripted model as an OpenAI endpoint, for a full `gym eval run` without GPUs.
