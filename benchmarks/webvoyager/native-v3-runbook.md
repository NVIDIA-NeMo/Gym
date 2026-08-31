# Native WebVoyager end-to-end runbook

This is the shortest supported path from a clean Gym checkout to one native
WebVoyager rollout and then the maintained 552-task population. It uses the
reference-aligned browser and judge contracts in
`benchmarks/webvoyager/configs/native_v3.yaml`.

For protocol details, provenance hashes, distributed execution, and failure
classification, see [native-v3.md](native-v3.md). The BrowserGym-compatible
643-task profile is a different runtime and is documented in
[README.md](README.md).

## 1. Required inputs

Prepare these before starting:

| Input | Required value or source |
| --- | --- |
| Gym source | This checkout and its committed `uv.lock` |
| Task data | `jayl940712/webarena_benchmarks`, commit `6a2977939b157b0ab9de7799bb089c721f1ac115`, file `webvoyager.jsonl` |
| Policy | An OpenAI-compatible, vision-capable endpoint that emits the native WebVoyager tool calls |
| Browser egress | A working US HTTP proxy in `WA_BROWSER_PROXY_SERVER` |
| CAPTCHA | A funded CapSolver key in `CAPSOLVER_API_KEY` |
| Judge | A vision-capable judge endpoint, API key, and model name |
| Linux display | Xvfb at 1920x1080, Chromium, and `xclip` |

The benchmark, browser, and judge never load the policy tokenizer. Tokenizer,
chat-template, multimodal processor, and output parsers belong to the policy
server. The pinned public Nano Omni serving assets are described in section 4.

## 2. Install the locked Gym environment

Run from the Gym repository root:

```bash
uv lock --check
uv sync --frozen --extra dev
./.venv/bin/gym --help >/dev/null
```

On the Linux execution host, verify the non-Python runtime dependencies and
install the Playwright Chromium build once:

```bash
command -v Xvfb
command -v xvfb-run
command -v xclip
uv run --project resources_servers/webvoyager_browser playwright install chromium
```

Do not run the native profile on macOS. PyAutoGUI captures and controls the
shared X display, and one resource-server process therefore supports exactly
one live browser session.

## 3. Fetch and verify the 552-task dataset

```bash
export WEBVOYAGER_DATA_ROOT="$PWD/.cache/webvoyager-dataset"
git clone https://github.com/jayl940712/webarena_benchmarks.git "$WEBVOYAGER_DATA_ROOT"
git -C "$WEBVOYAGER_DATA_ROOT" checkout --detach 6a2977939b157b0ab9de7799bb089c721f1ac115
export WEBVOYAGER_SOURCE_JSONL="$WEBVOYAGER_DATA_ROOT/webvoyager.jsonl"

test "$(sha256sum "$WEBVOYAGER_SOURCE_JSONL" | awk '{print $1}')" = \
  "f635a9b27fa1980a63b39bbf64ae8e9e766159cb70fa765451d3d3c0b948ff98"
test "$(wc -l < "$WEBVOYAGER_SOURCE_JSONL" | tr -d ' ')" = "552"
```

The same lock is machine-readable in `native_v3_source_lock.json`.
`prepare.py` repeats both the hash and task-count checks before writing Gym
records.

## 4. Configure the policy endpoint

### Existing endpoint

Set the endpoint that owns model loading and serving:

```bash
export POLICY_BASE_URL="https://policy-host.example/v1"
export POLICY_MODEL_NAME="served-model-name"
read -rsp "Policy API key: " POLICY_API_KEY
export POLICY_API_KEY
printf '\n'

curl -fsS \
  -H "Authorization: Bearer $POLICY_API_KEY" \
  "$POLICY_BASE_URL/models" >/dev/null
```

The endpoint must accept image content and the five native WebVoyager tools,
and return standard `tool_calls` with JSON arguments. The one-task smoke run in
section 7 is the authoritative compatibility check.

### Public Nano Omni v3 example

The public profile pins the model and tokenizer to the same immutable Hugging
Face revision. It uses that revision's standalone `chat_template.jinja`, the
public `nano_v3` reasoning-parser plugin, and vLLM's built-in `qwen3_coder`
tool-call parser:

```bash
export NANO_ASSET_ROOT="$PWD/.cache/nano-omni-v3-assets"
export NANO_MODEL_REPO="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
export NANO_MODEL_REVISION="24e67ea000b7c2837fc8f9488aa2008524fac8ba"
export NANO_PARSER_REPO="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
export NANO_PARSER_REVISION="f6aca92089793f4bc9ece522ffbb5365d38b5113"

hf download "$NANO_MODEL_REPO" \
  --revision "$NANO_MODEL_REVISION" \
  --include tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja \
  --local-dir "$NANO_ASSET_ROOT/model"
hf download "$NANO_PARSER_REPO" nano_v3_reasoning_parser.py \
  --revision "$NANO_PARSER_REVISION" \
  --local-dir "$NANO_ASSET_ROOT/parser"

export NANO_V3_REASONING_PARSER_PLUGIN="$NANO_ASSET_ROOT/parser/nano_v3_reasoning_parser.py"
```

The checked-in Gym-managed serving profile is:

```text
responses_api_models/local_vllm_model/configs/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16-alignment.yaml
```

It pins TP8, 128K context, temperature 0.1, top-p 0.95,
`max_output_tokens=16384`, `truncate_history_thinking=false`, the tokenizer
revision, and both parsers. Operators using an external vLLM process must apply
the same model-specific serving contract themselves. Do not borrow tokenizer
or template assets from another checkpoint.

## 5. Configure proxy, CAPTCHA, and judge

Keep credentials in the process environment, not in a tracked YAML file:

```bash
export WA_BROWSER_PROXY_SERVER="proxy-host.example:19407"
export WA_CAPTCHA_PROVIDER="capsolver"
read -rsp "CapSolver API key: " CAPSOLVER_API_KEY
export CAPSOLVER_API_KEY
printf '\n'

export WEBARENA_JUDGE_BASE_URL="https://inference-api.nvidia.com/v1"
export WEBARENA_JUDGE_MODEL="gcp/google/gemini-3-flash-preview"
read -rsp "Judge API key: " WEBARENA_JUDGE_API_KEY
export WEBARENA_JUDGE_API_KEY
printf '\n'
```

Verify that the browser proxy exits through the intended region and that the
CapSolver account is usable:

```bash
curl -fsS -x "http://$WA_BROWSER_PROXY_SERVER" https://ifconfig.me/ip
mkdir -p results/webvoyager/preflight
./.venv/bin/python benchmarks/webvoyager/smoke_capsolver_account.py \
  --output results/webvoyager/preflight/capsolver-account.json
```

If the browser reaches Squid through a node-local SSH tunnel, also export
`WA_CAPTCHA_PROXY_SERVER` as the public Squid endpoint. CapSolver cannot use a
worker-local address such as `127.0.0.1`.

## 6. Prepare the Gym composition

From the Gym repository root:

```bash
./.venv/bin/python benchmarks/webvoyager/prepare.py \
  --profile native_v3 \
  --source "$WEBVOYAGER_SOURCE_JSONL" \
  --rollout-output "$PWD/results/webvoyager/full/rollouts.jsonl" \
  --force-env

(cd benchmarks/webvoyager && ../../.venv/bin/gym env prefetch)
```

This writes:

- `benchmarks/webvoyager/data/webvoyager_native_v3.jsonl`, after validating the
  source hash and 552-task population;
- private, mode-0600, gitignored `benchmarks/webvoyager/env.yaml`, which
  references the policy and judge credentials through environment variables.

`prepare.py` also prints the equivalent `gym env prefetch`, `gym env start`,
and `gym eval run --no-serve` commands.

## 7. Run one end-to-end smoke task

Use a fresh run directory. The one-shot command starts the policy proxy,
browser, judge, and agent, then shuts them down after collection:

```bash
export RUN_ROOT="$PWD/results/webvoyager/smoke-1"
mkdir -p "$RUN_ROOT/logs/components"

cd benchmarks/webvoyager
xvfb-run --auto-servernum --server-args="-screen 0 1920x1080x24" \
  ../../.venv/bin/gym eval run -v \
  --limit 1 \
  --concurrency 1 \
  --output "$RUN_ROOT/rollouts.jsonl" \
  ++nemo_gym_log_dir="$RUN_ROOT/logs/components"
cd ../..
```

The smoke gate passes only when:

- exactly one rollout row is present;
- the row has a terminal reward rather than a masked infrastructure failure;
- component logs contain browser session, model turn, native action, and judge
  completion events;
- the policy actually returns a valid native tool call.

Check the retained evidence:

```bash
test "$(wc -l < "$RUN_ROOT/rollouts.jsonl" | tr -d ' ')" = "1"
jq '{task_id: .responses_create_params.metadata.task_id, reward, task_success, mask_sample, failure_kind, error}' \
  "$RUN_ROOT/rollouts.jsonl"
rg "web_rollout_start|web_model_turn_complete|native_browser_tool_complete|webvoyager_judge_complete" \
  "$RUN_ROOT/logs/components"
```

## 8. Run all 552 tasks

The portable single-display command is intentionally sequential:

```bash
export RUN_ROOT="$PWD/results/webvoyager/full-552"
mkdir -p "$RUN_ROOT/logs/components"

cd benchmarks/webvoyager
xvfb-run --auto-servernum --server-args="-screen 0 1920x1080x24" \
  ../../.venv/bin/gym eval run -v \
  --concurrency 1 \
  --output "$RUN_ROOT/rollouts.jsonl" \
  ++nemo_gym_log_dir="$RUN_ROOT/logs/components"
cd ../..
```

For practical throughput, use multiple isolated Gym processes. Every worker
must own a distinct X display, HOME, temporary directory, artifact directory,
and rollout file. Do not raise concurrency against one native resource server.
The two-split, 16-browser-per-split reference topology is documented in
[native-v3.md](native-v3.md#debug-logging-contract).

## 9. Reconcile the denominator

```bash
mkdir -p "$RUN_ROOT/aggregate" "$RUN_ROOT/cleanup"
./.venv/bin/python benchmarks/webvoyager/summarize_native_v3.py \
  "$RUN_ROOT/rollouts.jsonl" \
  --dataset benchmarks/webvoyager/data/webvoyager_native_v3.jsonl \
  --output "$RUN_ROOT/aggregate/summary.json" \
  --missing-output "$RUN_ROOT/cleanup/retry.jsonl"

jq '{expected, completed_unique, success, strict_sr, missing, duplicate_task_ids, unexpected_task_ids, invalid_or_infrastructure, comparable}' \
  "$RUN_ROOT/aggregate/summary.json"
```

A reportable full result requires all 552 task IDs exactly once and zero
missing, duplicate, unexpected, invalid, and unresolved infrastructure rows.
Rerun only the generated cleanup input, then reconcile the first wave and
cleanup wave as described in `native-v3.md`.

## 10. Server lifecycle alternative

For repeated runs, keep the servers in the foreground:

```bash
cd benchmarks/webvoyager
xvfb-run --auto-servernum --server-args="-screen 0 1920x1080x24" \
  ../../.venv/bin/gym env start
```

In a second terminal, from the same directory:

```bash
../../.venv/bin/gym eval run --no-serve --concurrency 1
```

Stop `gym env start` with Ctrl-C after the final run. Gym currently has no
separate `gym env stop` command. It also does not own or stop an external vLLM
server, Squid proxy, or judge service.
