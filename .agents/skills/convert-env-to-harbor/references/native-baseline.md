# The native arm

The native arm is the reference every Harbor number is compared with, so it must run the same harness and model as
the Harbor arm and nothing from the machine it runs on. Each rule below was a real failure in a pilot. Run 3-5 rows
natively before writing the adapter (SKILL.md Step 2), and again with the final configs for G4.

## Starting the native arm

Which command works depends on the native config:

| Native config | Command |
| --- | --- |
| Rows file passed on the command line (flat rows, `legacy_agent`) | `gym eval run --input` is refused when the command also starts the servers. Start the stack, run the client against it, then stop the stack (below). |
| A taskset or a dataset path in the config | `gym eval run --config <native config> <model flags> --split <split> --output <abs path>/rollouts.jsonl`. `--split` and `--output` are required for a config target. |
| An Environment Server pairing (`single_agent_turn`) whose dataset is `type: example` | Neither route above works: `--split example` is refused ("not runnable end-to-end"), and `/run` answers 422 to flat rows (it takes only materialized taskset rows). Add a run-local overlay that redefines the dataset as `type: train` on your rows, then use the config target with `--split train` (below). |

Rows file (math pilot):

```bash
gym env start --config <native config> --config native_arm.yaml <model flags> \
    ++observability_enabled=true ++model_call_capture_dir=<abs dir>
# in a second shell, once the stack logs "servers ready":
gym eval run --no-serve --config <native config> --config native_arm.yaml --agent <native agent instance> \
    --input <native rows.jsonl> --output <run dir>/rollouts.jsonl <model flags> --concurrency 4 \
    ++observability_enabled=true ++model_call_capture_dir=<abs dir>
# then stop the stack: SIGINT, and SIGTERM if it is still up after a minute
```

The capture flags go on `gym env start`: the servers it starts write the captures. On the client only, the run
scores but captures nothing (`model_calls_unavailable`; reasoning_gym lost a native oracle capture this way).
From G2 on, `<native rows.jsonl>` is the adapter's `--native-rows-out` of the dataset you are gating, run whole (no
`--limit`).

Config target (scale_swe pilot):

```bash
gym eval run --config <native config> <model flags> --split train --output <abs path>/rollouts.jsonl \
    --concurrency 3 ++use_absolute_ip=true ++observability_enabled=true ++model_call_capture_dir=<abs dir>
```

- Dataset overlay for an `example` pairing, passed as one more `--config` (swe_rebench ships it in
  `harbor/configs/native_arm.yaml`, filled with `++swe_rebench_native_rows=<abs path>`); keep the server's own
  `taskset` and `license`, and point `jsonl_fpath` at the adapter's `--native-rows-out`:
  ```yaml
  <resources server instance>:
    resources_servers:
      <server type>:
        datasets:
        - name: <dataset name>
          type: train
          taskset: <the config's taskset>
          jsonl_fpath: <abs path of native rows>
          num_repeats: 1
          license: <the config's license>
  ```
- A server that starts sandboxes itself (SWE) needs the provider config as one more `--config`
  (`nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml`) unless its YAML already names one.
- A config's dataset path resolves from the working directory first. To run selected rows, write them to the
  config's relative dataset path under the directory you run from (workplace_assistant), or point a copy of the
  config at your rows file (scale_swe).
- A server started in the background from a non-interactive shell ignores SIGINT; stop it with SIGTERM. `pkill -f`
  and `pgrep -f` match the shell that runs them when the pattern is in its own command line: find the pid, then
  `kill` it.
- Stop a held stack as soon as its client run ends; it holds ports and memory.

## Isolation

- **Remote sandboxes need `++use_absolute_ip=true`.** Without it the policy model listens on `127.0.0.1`, OpenCode in
  a remote sandbox cannot reach it, and every episode hangs silently until the episode timeout (21600 s by default;
  no error in any log). `gym eval run <task folder>` sets it on the Harbor arm by itself. Compare both arms'
  `rollouts_resolved_config.yaml` before trusting a native smoke.
- **A local CLI harness inherits the host.** A workspace inside a git checkout pulls that checkout's `AGENTS.md`
  into the system prompt; a real HOME pulls personal instructions (`~/.claude/CLAUDE.md`) and personal skills into
  its `<available_skills>` list. The tag itself is not contamination: OpenCode lists its built-in skill there even
  with an empty HOME, on both arms; what matters is extra entries on the native side. The math pilot's contaminated system prompt was 48,240 characters against 10,445 isolated.
  Fix it in `native_arm.yaml`:
  ```yaml
  <native agent instance>:
    responses_api_agents:
      opencode_agent:
        model: ${policy_model_name}
        model_server: {type: responses_api_models, name: policy_model}
        workspace_root: ???   # a dir outside any git repository, no AGENTS.md or CLAUDE.md up to /
        env:
          HOME: ???           # a fresh empty dir for every run: OpenCode writes state into it
          XDG_CONFIG_HOME: ${.HOME}/.config
          XDG_CACHE_HOME: ${.HOME}/.cache
          XDG_STATE_HOME: ${.HOME}/.local/state
  ```
  Supply both on `gym env start` and on the `gym eval run --no-serve` client as
  `++<native agent instance>.responses_api_agents.opencode_agent.workspace_root=<dir>` and
  `...opencode_agent.env.HOME=<fresh empty dir>`. A fixed HOME is not empty after the first run (reasoning_gym reused
  one across runs). The `env` block reaches only the OpenCode subprocess. A local CLI with shell access runs as your user and can read
  any file, the native rows (which hold the answers) included; this cannot be closed in config. Keep the rows outside
  the workspace and HOME, record the exposure as a numbered difference, and grep the native transcripts for the rows
  file's name.
- **Never set `OPENCODE_DISABLE_PROJECT_CONFIG` on the local path.** The local path writes its provider into the
  workspace's `opencode.json`; with that flag OpenCode ignores the file and exits before any model call.
- **Pin the local binary.** The local path installs Node and OpenCode into the repository's agent folder unless
  `opencode` is on PATH. Put the release the sandbox installs (`opencode_version` in the agent config), built for the
  host's architecture, first on PATH:
  ```bash
  arch=$(uname -m | sed 's/x86_64/x64-baseline/; s/aarch64/arm64/')   # the asset the sandbox installer uses
  curl -fsSLO https://github.com/anomalyco/opencode/releases/download/v<version>/opencode-linux-$arch.tar.gz
  sha256sum opencode-linux-$arch.tar.gz   # must equal the pinned digest before you unpack it
  tar xzf opencode-linux-$arch.tar.gz
  ```
  The pinned digests are `OPENCODE_RELEASE_SHA256` in `responses_api_agents/opencode_agent/runtime.py` (the ones
  the sandbox installer checks). For a version without an entry there (1.17.11 has none, and its release has no
  `SHA256SUMS`), compare with the asset digest GitHub records, and write it in the log:
  ```bash
  curl -fsS https://api.github.com/repos/anomalyco/opencode/releases/tags/v<version> \
      | python3 -c 'import json, sys; print(*[a["digest"] for a in json.load(sys.stdin)["assets"]
          if a["name"] == sys.argv[1]])' opencode-linux-$arch.tar.gz
  ```
  When you run the binary by hand, give it `</dev/null`: `opencode run` with an argument blocks on an open stdin.
- **Model routing.** Shipped configs may name a provider directly (math's config hard-coded one model on a direct
  provider). Route both arms through Gym's `policy_model` (above) so the same `<model flags>` select the model.
- **Training-only settings.** Set `token_id_capture` the same on both arms (scale_swe ships it `true` for
  training; the smoke set it `false` on the native arm to match).

## Model-free native runs

The native oracle and negative controls, and a capture of the native first request, need no model. Point
`policy_model` at the fake upstream from `scripts/harbor_conversion_checks`:

```bash
python -m scripts.harbor_conversion_checks fake-model --port <port> --replies <replies.jsonl>
# replies.jsonl: one {"match": "<text unique to the task's user message>", "text": "<reply>"} per task
```

Write each reply in the answer format the native prompt asks for and the native grader extracts: math's
`\boxed{<expected>}`, an `<answer>...</answer>` tag for reasoning_gym. It serves chat completions only (OpenCode); an
agent that calls the Responses API (`/v1/responses`, such as `simple_agent`) needs its own fake upstream.
reasoning_gym's native oracle and controls ran on it (50 rows: reference 1.0, empty and wrong 0, partial credit equal
to Harbor's). Check that every reference row scores 1.0 before you trust a control run on it.

Then run the native arm as above with `--model-type openai_model --model <the parity model's name> --model-url
http://127.0.0.1:<port>/v1` (the real name, so OpenCode picks the same model-family prompt), once with the reference
answers as replies (oracle: every row 1.0) and once per negative control (a wrong answer, no answer: 0). The same
`--model-url` on the Harbor arm captures the Harbor first request. Record requests with
`++observability_enabled=true ++model_call_capture_dir=<abs dir>`; the fake model keeps no log. A run that logs
`model_calls_unavailable` was started without those two flags and captured nothing.

The fake model answers text only. Tool-calling families use their own native control: the golden-patch mode of a
SWE server ([families.md](families.md#swe-per-row-image-patch-capture-hidden-tests-scale_swe-swe_rebench)), or `/seed_session`
then `/verify` against the Resources Server app with the reference, a wrong and an empty call list
(workplace_assistant). Write each probe case as its own `rollouts.jsonl`, one JSON line per row with `reward`,
`_ng_task_index` (0-based row), `_ng_rollout_index` 0 and the native key field; with no `_materialized_inputs.jsonl`
beside it, `harbor_parity` takes every row that left a record as expected.

## Known gaps on main

Each gap below made a native number wrong or impossible in a pilot. Until it is fixed on main, run the native arm
on a demonstrator of the fix and label every number with it. No issue is filed yet; link one here when it is.

| Gap | Symptom | Removed when |
| --- | --- | --- |
| A Resources Server whose `seed_session` override handles only the cookie seed | Every Environment Server episode fails at seed: `validation errors for ResourcesSeedSessionResponse` | The server implements `has_session_state`, `initialize_session_state`, `release_session_state` instead of the override |
| MCP tool names: Gym strips only `mcp__<server>__`; OpenCode (and Pi) record `<server>_<tool>` | A name-based `verify()` silently scores 0 on every row with a non-empty ground truth; the server log shows `Error executing tool: '<server>_<tool>'` | `mcp_auto_exposure` normalizes `<server>_<tool>` for the server's own tools |
| OpenCode's sandbox session refuses rows that carry `tools` or `parallel_tool_calls` | 422 | Strip both from native MCP rows (workplace_assistant's `--opencode-rows-out` does; `--native-rows-out` keeps raw lines) |
| Collect-hook output used as a mask signal | An agent turns its 0 into a mask | Task design rule: map hook output to a completed 0 ([reward-safety.md](reward-safety.md)) |

Grep the native server log for the second symptom after every native run; it must print nothing.
