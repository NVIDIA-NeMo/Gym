# Native Nano Omni v3 WebVoyager

For copyable setup, preparation, one-task smoke, full-population execution,
and reconciliation commands, use the
[native WebVoyager end-to-end runbook](native-v3-runbook.md). This document is
the detailed protocol and reproducibility reference.

This profile reproduces the maintained screenshot/tool-call WebVoyager route
inside NeMo Gym. It is separate from `benchmarks/webvoyager/config.yaml`, which
continues to provide the public BrowserGym-compatible baseline. The original
Selenium runner is not included.

## Code path

```text
552-task pinned dataset
  -> WebAgent native profile
     -> Gym Responses boundary
        -> vLLM model proxy
           -> public Nano Omni v3 /v1/chat/completions endpoint (TP8)
     -> webvoyager_browser resource server
        -> one Xvfb display
        -> headed Chromium managed by Playwright
        -> visible coordinate actions executed by PyAutoGUI
     -> native_v3 WebVoyager Gemini judge
```

The dataset, agent, browser and judge communicate through the common Gym web
models. The agent never imports Playwright or PyAutoGUI, and the browser server
does not own model prompting or scoring. `webvoyager_browser` owns only the
public-site proxy, CAPTCHA, and WebVoyager navigation policy; reusable headed
Chromium/PyAutoGUI mechanics remain shared with `native_web`, whose resource
boundary is restricted to WebArena and VisualWebArena.

## Reproducibility contract

- Native recipe source: `osworld_internal` branch `nemotron-v3`, commit
  `3b775dc538931ead0cb6b4922349da9c6d493dab`.
- Dataset: `jayl940712/webarena_benchmarks` commit
  `6a2977939b157b0ab9de7799bb089c721f1ac115`, `webvoyager.jsonl`, 552 tasks,
  SHA-256 `f635a9b27fa1980a63b39bbf64ae8e9e766159cb70fa765451d3d3c0b948ff98`.
- Public policy example: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`
  revision `24e67ea000b7c2837fc8f9488aa2008524fac8ba`. The benchmark itself consumes
  an OpenAI-compatible multimodal policy endpoint and is not tied to this
  checkpoint.
- Generation: temperature 0.1, top-p 0.95, 100 browser steps, three recent
  browser screenshots, structured native browser tools.
- Serving contract used for the public profile: TP8, 128K context, the
  tokenizer and chat template published with the pinned model revision, and
  `chat_template_kwargs={"truncate_history_thinking": false}` on every vLLM
  chat-completions request. Layer
  `benchmarks/webvoyager/configs/native_v3_policy.yaml` after the generic
  `vllm_model` config.
- Judge: `gcp/google/gemini-3-flash-preview`, all trajectory screenshots,
  JSON `SUCCESS`/`FAILURE` verdict.
- Browser timing: the start URL settles on `domcontentloaded`, every
  policy-driven navigation settles on `load`, `page.goto` retries transport
  faults after 4/4/4/8 s, and one 45 s context deadline bounds every Playwright
  operation.
- Policy transport: up to 20 attempts at 5 s for a single turn, so an unstable
  endpoint does not mask a task the reference would have completed.

The profile opts into bounded recovery for decoding and executing an action the
policy already chose: inner-JSON and one-missing-bracket repair, unambiguous
public-v3 tool aliases, and a failed UI action left visible for correction. Its
three total action-parse attempts match the pinned reference: every attempt uses
the same request and temperature 0.1, with no harness-authored feedback and a
one-second delay after a parse failure. It also does not enable the repeated-
action hint, which would place harness-authored strategy in the policy's context
that the pinned reference never sends.

A browser that exhausts its CAPTCHA budget reports
`native_status=captcha_budget_exhausted`; the agent masks that rollout as
`failure_kind=captcha_budget_exhausted` rather than judging a forced stop, and
the summarizer routes it to the cleanup wave.

## Public policy-serving assets

The benchmark, agent, browser and judge do not load a tokenizer. They call an
OpenAI-compatible multimodal policy endpoint. Tokenizer, chat-template and
output-parser choices matter only when an operator starts a local model server.

The public Nano Omni profile uses the tokenizer and standalone
`chat_template.jinja` published at the
[same immutable revision as the model](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/tree/24e67ea000b7c2837fc8f9488aa2008524fac8ba). <!-- pragma: allowlist secret -->
The [`nano_v3` reasoning parser](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/f6aca92089793f4bc9ece522ffbb5365d38b5113/nano_v3_reasoning_parser.py) <!-- pragma: allowlist secret -->
is also public, but it is hosted in the NVIDIA Nemotron Nano model repository
and vLLM requires it as a local file. Download the small serving assets as
follows:

```bash
ASSET_DIR=/path/to/nano-omni-v3-assets
MODEL_REPO=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
MODEL_REVISION=24e67ea000b7c2837fc8f9488aa2008524fac8ba  # pragma: allowlist secret
PARSER_REPO=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
PARSER_REVISION=f6aca92089793f4bc9ece522ffbb5365d38b5113  # pragma: allowlist secret

hf download "$MODEL_REPO" \
  --revision "$MODEL_REVISION" \
  --include tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja \
  --local-dir "$ASSET_DIR/model"
hf download "$PARSER_REPO" nano_v3_reasoning_parser.py \
  --revision "$PARSER_REVISION" \
  --local-dir "$ASSET_DIR/parser"

export NANO_V3_REASONING_PARSER_PLUGIN="$ASSET_DIR/parser/nano_v3_reasoning_parser.py"
```

The checked-in local-vLLM profile pins the public tokenizer revision. The
locked Transformers/vLLM stack selects the standalone `chat_template.jinja`
from that snapshot instead of the older template embedded in
`tokenizer_config.json`. The profile also selects vLLM's built-in
`qwen3_coder` tool-call parser and fails closed when
`NANO_V3_REASONING_PARSER_PLUGIN` is unset. For an offline deployment, override
`tokenizer` and `chat_template` with the downloaded model directory and
`chat_template.jinja` path.

These public assets provide a fully public serving configuration. They are not
claimed to be byte-identical to every recipe-specific tokenizer or template
used by earlier reference-score experiments. A result produced with such
overrides must record their paths, revisions or hashes separately instead of
presenting them as requirements of the public benchmark profile. A tokenizer
warning observed with an older serving stack is evidence of version or asset
drift in that run, not a general WebVoyager compatibility rule.

## Using other policy models

Do not substitute a tokenizer from another model merely because both servers
use the OpenAI API or the same tokenizer class. A local model should use the
tokenizer, multimodal processor, chat template, reasoning parser and tool-call
parser published or documented for that checkpoint. Token IDs must agree with
the model embeddings and special-token configuration; multimodal placeholders
must render correctly; and the selected parser must recognize the template's
tool-call syntax.

For an externally managed endpoint, those serving choices remain outside Gym.
Compatibility with this benchmark is verified at the API boundary: send one
image and the native browser tool schemas, confirm that the endpoint returns a
standard `tool_calls` entry with an allowed function name and JSON arguments,
then complete a one-task rollout before scaling out. This serving contract is
independent of the browser backend. The native Playwright/PyAutoGUI route and
the BrowserGym-compatible route can call the same policy endpoint.

## Required run inputs

The operational form of this checklist, including exact commands and pass
criteria, is maintained in [native-v3-runbook.md](native-v3-runbook.md).

1. Set `WEBVOYAGER_SOURCE_JSONL` and run the native prepare script.
2. Point `POLICY_BASE_URL` at a compatible external endpoint, or prepare the
   public local-serving assets above and set
   `NANO_V3_REASONING_PARSER_PLUGIN`. Record explicit tokenizer/template
   overrides only when reproducing a separately defined serving recipe.
3. Start every native resource-server instance under a unique 1920x1080 Xvfb
   display. `max_sessions=1` is intentional: PyAutoGUI is display-global.
   Install `xclip` in the browser image. PyAutoGUI cannot emit non-ASCII
   characters, so the runtime pastes them through the X clipboard instead; a
   missing `xclip` turns every non-ASCII `type` action into a step failure.
4. Provide `WA_BROWSER_PROXY_SERVER` for the selected US-proxy domains. The
   reference list includes `html.duckduckgo.com`: Google Search tasks are
   rewritten to that fallback, and direct access can time out outside NRT.
5. Provide `CAPSOLVER_API_KEY` and set `WA_CAPTCHA_PROVIDER=capsolver`. The
   built-in Turnstile/reCAPTCHA v2 integration is used unless an approved
   `WA_CAPTCHA_SOLVER=module.path:factory` override is supplied.
   If the browser reaches Squid through a node-local SSH tunnel, also set
   `WA_CAPTCHA_PROXY_SERVER` to the public endpoint for that same Squid
   instance. CapSolver runs outside the worker namespace and cannot use a
   browser proxy such as `127.0.0.1:19407`.
6. Configure the judge endpoint/key through the standard Gym secret/config
   channel. Episodes call the judge resources server through Gym's canonical
   `/verify` route. Judge transport/schema failures use the shared
   `judge_failed` sidecar and retained response evidence supports judge-only
   reverification.

The native tool contract limits a single scroll action to 50 wheel clicks, and
the browser clamps the value again before calling PyAutoGUI. This matches the
reference worker-safety guard and prevents extreme model-generated values from
stalling a worker.

Prepare the local, gitignored benchmark composition with the same locked CLI
used by the root repository:

```bash
cd /path/to/Gym
uv lock --check
uv sync --frozen --extra dev
export WEBVOYAGER_SOURCE_JSONL=/path/to/pinned/webvoyager.jsonl
./.venv/bin/python benchmarks/webvoyager/prepare.py \
  --profile native_v3 \
  --force-env
```

Then run one-shot with `../../.venv/bin/gym eval run` from
`benchmarks/webvoyager`, or keep `gym env start` foregrounded and collect with
`gym eval run --no-serve` from a second terminal. Native concurrency requires
one isolated Gym process/X display per rollout; `prepare.py` rejects
`--concurrency` greater than one for a single native resource server.

The agent-facing route remains Gym's Responses API so the common rollout
contract is unchanged. The `vllm_model` proxy converts the request to OpenAI
Chat Completions and sends it to the policy endpoint. The captured outbound
payload—not the agent-facing URL—is the parity boundary: it must contain
`messages`, the five native WebVoyager tools and
`chat_template_kwargs.truncate_history_thinking=false`.

## Debug logging contract

Run with `verbose=true` and set `nemo_gym_log_dir` to a run-scoped directory.
Gym then retains one log per subprocess instead of relying only on the parent
terminal. The native route emits redacted `key=value` lifecycle events at the
following boundaries:

| Component | Representative events |
| --- | --- |
| Agent | `web_rollout_start`, `web_model_turn_complete`, `web_action_parsed`, `web_rollout_complete` |
| Browser/session | `web_session_seed_complete`, `native_browser_tool_complete`, `native_browser_screenshot`, `web_session_close` |
| CapSolver | `captcha_detected`, `captcha_task_created`, `captcha_solved`, `captcha_solver_failed` |
| Judge | `webvoyager_judge_model_complete`, `webvoyager_judge_unparseable`, `webvoyager_judge_complete` |

Logs include benchmark/task/session/step identifiers, safe URL origins,
operation names, status, latency, evidence sizes/hashes, token counts and error
classes. They intentionally exclude full URLs with query strings, browser
action arguments, screenshots, answers, judge text, API keys, proxy
credentials, provider task IDs and CAPTCHA solution tokens. Provider and
evidence identifiers appear only as short SHA-256 fingerprints.

For an experiment rooted at `${RUN_ROOT}` use overrides equivalent to:

```bash
++verbose=true \
++nemo_gym_log_dir=${RUN_ROOT}/logs/components
```

The CapSolver account gate is separate from the real browser-challenge gate:

```bash
python benchmarks/webvoyager/smoke_capsolver_account.py \
  --output "${RUN_ROOT}/status/capsolver-account-preflight.json"
```

It calls `getBalance` directly, never through the browser proxy, and persists
only balance/status metadata and a key fingerprint. A comparable run also
requires a real challenge log containing `captcha_detected` followed by
`captcha_solved` and evidence that the target page accepted the injected token.

For the reference topology, run two dataset splits concurrently. Each split
uses one TP8 policy replica and 16 one-session browser resource replicas. A
cleanup/resume pass dispatches only tasks without a terminal result.

Each browser replica must have an independent X display, HOME, temporary
directory, artifact directory and rollout output. Do not set
`gym eval --concurrency=16` against a single native resource server: native
coordinate actions are display-global and that server intentionally permits
only one session. Start 16 isolated, single-concurrency Gym processes per
split instead. The policy endpoints and read-only component environments may
be shared within a split after one serialized prefetch has completed.

Merge worker outputs by task ID and generate the cleanup input from the exact
pinned dataset, rather than from worker exit codes. The summarizer accepts
multiple files or run directories and writes the missing-task JSONL
atomically:

```bash
python benchmarks/webvoyager/summarize_native_v3.py \
  split-0/workers split-1/workers \
  --dataset webvoyager.jsonl \
  --output aggregate/summary.json \
  --missing-output cleanup/missing.jsonl
```

A result is comparable only when all expected task IDs are present exactly
once, no unexpected task IDs are present, and no task is masked as an
infrastructure/configuration failure.

When a cleanup wave reruns masked rows, pass its input with
`--superseded-ids-jsonl` and list the first-wave paths before cleanup paths.
The last result for only those explicitly declared task IDs then supersedes
the preserved first-wave result; undeclared duplicates remain an audit error.

## Validation order

Run one task first, then one task from every retained domain, then a 32-task
coverage subset, and only then all 552 tasks. Every report must distinguish
policy failure from browser, proxy/captcha, model-server and judge failures;
only a fully accounted denominator is comparable to a reference SR.

## Full-population validation

On 2026-08-26, a hash-sealed PR 2295 review candidate completed the maintained
552-task population at **421/552 = 76.27% SR**. Reconciliation reported zero
missing, duplicate, unexpected, invalid, or unresolved infrastructure rows and
an empty retry set. The result was 7 successes below the previous 428/552 Gym
control and 8 below the 429/552 maintained reference golden.

The corrected-wave transport audit observed `max_tokens=16384`, temperature
0.1, top-p 0.95, and `truncate_history_thinking=false` on every recorded
outbound policy request. The final source generation also passed production
agent/browser imports, 5 focused agent regressions, and all 14 CAPTCHA
regressions in the target Linux/Enroot runtime. This evidence applies only to
the maintained native 552-task profile; it does not claim equivalent coverage
for the BrowserGym-compatible 643-task route, WebArena, or VisualWebArena.
