# Agentic-MME (atomic tools)

Native Gym implementation of the public Agentic-MME atomic interface:
14 image tools plus 3 optional retrieval tools. The verifier reports **Track B,
task-only accuracy**, not the paper's full S/V process scores. No reward is
given for using a tool. No benchmark tasks are configured as training data.

Sources:

- [Paper](https://arxiv.org/abs/2604.03016), especially appendices A, B, E.
- [Reference interface](https://github.com/ChoS3nE11ven/Agentic-MME/tree/5e5dfbdc7b40b53dea17d5ce754c14034acb185b).
- [Official data](https://huggingface.co/datasets/Crystal1047/Agentic-MME/tree/b9ea9d3f68ff896d83fec666b6143da594c054a8).

## Tools

Image tools: crop, rotate, flip, resize, enhance, grayscale, autocontrast,
blur, sharpen, denoise, edge_detect, invert, equalize, threshold.

Retrieval: google_search, google_lens_search (SerpAPI's Google Lens, image uploaded to SerpAPI; providers below),
fetch_webpage (Jina Reader, or Exa /contents). The paper describes download_image, but the released
atomic harness disables it; this integration likewise does not expose it.
Free-form Python/Gen mode is not implemented.

Images use zero-based indices. All original images are indexed first; each
successful visual operation appends one image. Coordinates are normalized to
0–1000. Positive rotations are counterclockwise. New images are actually included
in the next model input, not just returned as paths. Original images must be
PNG/JPEG/WebP data URLs. Local paths and arbitrary remote URLs are not opened.

Tools enforce per-image/episode pixel limits, argument validation and bounded
rollout concurrency. OpenCV is required for consistent denoise/Canny/Sobel
semantics; there is no silent Pillow approximation.

## Scoring and traces

Task metadata uses the released shape:

    verifier_metadata:
      task_id: ...
      golden_answer:
        value: "44.6 million"
        match_type: contains
      process_evaluation:
        efficiency:
          reference_tool_calls: 2

Accepted match modes are exact (case-sensitive), contains (case-insensitive,
with the released short-number boundary rule), and numeric (absolute tolerance).
The default is contains, matching the released evaluator, although the paper
describes normalized exact match. A task with an empty, missing or nonfinite target is not graded: its row is
masked (see Sessions).
This preserves the released contains behavior, including its limitations; it is
not a hardened training verifier. Final answers use <answer>...</answer> (the
last tag when there are several), with plain terminal text as a fallback;
reasoning blocks are excluded.

With the Agentic-MME agent, the result includes tool_trace, counts, overthink and process_scores_available=false,
on /run and under an agent session alike. The agent reports the trace in response.metrics, which the model never
sees, and verify moves it into the result.
Overthink counts successful observable interactions relative to reference_tool_calls;
failed requests are counted separately. It never changes the reward.
Full S/V, V-tool and V-true checkpoint judging and published-score reproduction
remain unimplemented. Tool traces retain arguments and outputs for future process
evaluation; each new image is in response.output, linked by new_image_index. Do not report these runs as official Track A.

The default budget is 15 interaction rounds / 15 attempted tools, plus one more
model call to answer. There is no forced final-answer call: every model call
keeps the same tools and tool_choice, so each call's prompt extends the previous
one (NeMo RL token capture links a rollout's calls by that prefix). The episode
ends unanswered (reward 0) when the rounds run out or the model calls a tool
beyond the budget. Invalid tool calls consume the budget.
All prior messages and model token metadata are preserved between turns.

## Data

The five checked-in examples are small **synthetic smoke tasks**, not official
benchmark examples or a baseline. The official dataset is converted by the
benchmark's preparation script, which downloads the pinned release (or reads a
local snapshot from AGENTIC_MME_DATASET_ROOT):

    gym eval prepare --benchmark agentic_mme

The converter retains task IDs, gold answers, process checkpoints and difficulty
in verifier_metadata, never in the model prompt. It embeds input images and
does not expose reference crops or search answers. The released HF split is
called train, but it is an evaluation benchmark; do not train on it by default.
Full converted datasets are ignored by git. See
[benchmarks/agentic_mme](../../benchmarks/agentic_mme/README.md) for the measurement contract.

## Sessions

Every pairing runs under the `single_agent_turn_legacy` Environment Server, which reads the flat rows unchanged. For
each episode it seeds this server with a typed seed, opens an agent session, runs one turn, calls `/verify`, and then
calls `/close_session`.

- **What a session holds.** With `serve_image_tools`, an image workspace that the tools change. With
  `sandbox_provider`, the sandbox lent to the harness, with or without the task images. With neither, the server keeps
  nothing, and the base class answers the seed and the close.
- **Seed.** A row whose `image_files` is missing, malformed, outside `image_root` or unreadable is refused with 422.
  A seed that names an open session with another episode or task, or a closed session, is refused with 409. A repeated
  seed returns the same sandbox.
- **Close.** Close stops the lent sandbox and drops the workspace by `resources_session_id`. It is safe to repeat. A failed stop raises without fencing, so a retried close stops the sandbox. A
  closed ID is fenced for `session_close_retry_window_seconds`.
- **After close.** An image tool call answers 404 (`session_lost`), and a `/verify` returns a masked sample.
- **Shutdown and expiry.** Shutdown stops every sandbox still lent. A session that is never closed keeps its sandbox
  until shutdown, or until the provider reclaims it after `sandbox_config.ttl_s`. Set `ttl_s` above the agent's
  timeout plus grading: `opencode.yaml` sets 7200 s against OpenCode's 3600 s timeout.
- **Grading.** A wrong or missing answer is reward 0 with `mask_sample: false`. A judge failure, or a judge reply
  without a verdict, masks the row and sends it to the failures sidecar (`judge_failed`). A task without a usable gold
  answer is masked with `verifier_error` and `evaluation_completed: false`.

| Config in `benchmarks/agentic_mme/` | Harness | Image tools | Sandbox |
| --- | --- | --- | --- |
| `config.yaml` | `agentic_mme_agent` | in the agent | none |
| `server_tools.yaml` | `simple_agent` | on this server, over direct HTTP | none |
| `opencode.yaml` | `opencode_agent` | its own shell, Python and read tool | lent, with the task images |
| `opencode_server_tools.yaml` | `opencode_agent` | on this server, over MCP | lent, empty |

The seed that an agent's `/run` sends through a `legacy_agent` relay still works, for compatibility.

## Run

From the Gym checkout, install the agent's requirements in its isolated environment.
Provide policy_base_url, policy_api_key and policy_model_name in private Gym config.

    gym env start \
      --config resources_servers/agentic_mme/configs/agentic_mme.yaml \
      --model-type openai_model

    gym eval run --no-serve --agent agentic_mme_agent \
      --input resources_servers/agentic_mme/data/example.jsonl \
      --output /tmp/agentic-mme-smoke.jsonl --num-repeats 1 \
      --max-output-tokens 4096 --temperature 0

### Image tools on the resources server

With `serve_image_tools: true`, this server also serves the 14 image tools as routes, so a general harness can run
the task instead of the Agentic-MME agent. `benchmarks/agentic_mme/server_tools.yaml` runs it with `simple_agent`
over direct HTTP. Each episode's seed loads the images that the row's `image_files` names (under `image_root`) into
a workspace on this server, as images 0, 1, ... in prompt order. A tool reply carries its new image under
`tool_output_images`; `simple_agent` shows it to the model in a user message after the turn's tool outputs, and the
MCP endpoint returns it as image content. The prompt, tools and image layout match the Agentic-MME agent's.
Retrieval tools stay in the agent. `benchmarks/agentic_mme/prepare.py` writes these rows and their image files.

    AGENTIC_MME_IMAGE_ROOT=resources_servers/agentic_mme/data/example_images \
    gym env start --config benchmarks/agentic_mme/server_tools.yaml --model-type openai_model

    gym eval run --no-serve --agent agentic_mme_server_tools_agent \
      --input resources_servers/agentic_mme/data/example_server_tools.jsonl \
      --output /tmp/agentic-mme-server-tools.jsonl --num-repeats 1

Visual-only is an ablation, not full benchmark parity. To enable all 17 tools,
merge configs/agentic_mme_search.yaml after the base config and supply
agentic_mme_serpapi_api_key and agentic_mme_jina_api_key (the Jina key may be an
empty string). This explicitly allows images to be sent to SerpAPI. Only use it
with data approved for that service.

For deterministic offline retrieval, set retrieval.mode=replay and put an ordered
retrieval_replay list on each input row; entries contain tool_name, arguments and
output from a previous tool_trace, one per retrieval call, including calls rejected
before the provider (their recorded error is replayed). Replay must match tool and
arguments, as sent or once normalized. Mismatches fail visibly and never fall back to
the network.

### OpenCode in a lent sandbox

With `sandbox_provider` set, each episode's seed lends the harness a sandbox, and `task_images_in_sandbox` decides
whether it holds the task images. Two configs run OpenCode this way:

- `benchmarks/agentic_mme/opencode.yaml`: the sandbox holds the task images under `images/`, and OpenCode works on
  them with its own shell, Python (OpenCV, NumPy and Pillow) and read tool. Its scores are a separate track.
- `benchmarks/agentic_mme/opencode_server_tools.yaml`: the sandbox is empty (`task_images_in_sandbox: false`), this
  server serves the 14 image tools over MCP, and OpenCode's own shell, file and web tools are off. The prompt and
  tools match the Agentic-MME agent's, so its scores compare with `config.yaml`.

`sandbox_network_access` sets the lent sandbox's network. `opencode.yaml` uses `model_only`: the sandbox reaches only
the policy model (`sandbox_model_server`), so OpenCode cannot install tools or look answers up on the internet, and it
runs the copy of OpenCode that the sandbox image holds. `opencode_server_tools.yaml` uses `model_and_tools`, which
also lets the sandbox reach this server's `/mcp` route. Only OpenSandbox enforces these rules; with another provider,
set `sandbox_network_access: inherit`. A rule opens every port of an allowed host, so `opencode.yaml` gives the head
server (it returns the whole config, keys included), the judge, the agent, the Environment Server and (except with
server tools) this server the address `127.0.0.1`; a server's own host also wins over a launcher's `default_host`.
NeMo RL sets the head server's address itself, which wins over the recipe, so under NeMo RL the head server stays on
the node's address. The rules block raw IP addresses only on an OpenSandbox server that
enforces egress at the IP layer (`dns+nft`). The Environment Server admits 32 episodes at a time, the agent's
concurrency, because each seed starts a sandbox.

Both need a sandbox image built from `sandbox_image/Dockerfile` (`AGENTIC_MME_SANDBOX_IMAGE`) and the OpenSandbox
provider (`OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY`). The image holds OpenCode and OpenCode's plugin dependencies,
which `opencode.yaml` pins by SHA-256; without the dependencies, each offline OpenCode start waits about 70 s. They
come from the lockfile in `sandbox_image/opencode/`, a copy of `benchmarks/agentic_if/opencode/`. OpenCode calls the model server from inside the sandbox, and
with server tools also this server's `/mcp` route, so the servers it calls advertise the node's address:

    AGENTIC_MME_IMAGE_ROOT=resources_servers/agentic_mme/data/example_images \
    gym env start --config benchmarks/agentic_mme/opencode_server_tools.yaml --model-type openai_model \
      ++use_absolute_ip=true '++absolute_ip_server_types=[responses_api_models,resources_servers]'

    gym eval run --no-serve --agent agentic_mme_opencode_agent \
      --input resources_servers/agentic_mme/data/example_opencode_server_tools.jsonl \
      --output /tmp/agentic-mme-opencode-server-tools.jsonl --num-repeats 1

For `opencode.yaml`, use `example_opencode.jsonl` and `'++absolute_ip_server_types=[responses_api_models]'`.

### Web search providers and shared cache

`retrieval.search_provider` picks the text-search backend (`serper`, `serpapi`,
`tavily`, `exa`). Google Lens (google_lens_search) is always offered with text search,
as in the benchmark, and always runs through SerpAPI, so live retrieval needs
`serpapi_api_key` and `allow_image_upload: true` whatever the text-search provider. Every result is
normalised to the Serper shape, so the model sees the same tool output either way.
Ready-made overlays for `benchmarks/agentic_mme/config.yaml` (keys come from the
environment):

| overlay | text search | Lens | keys |
|---|---|---|---|
| retrieval_serpapi.yaml | SerpAPI | SerpAPI | SERPAPI_API_KEY |
| retrieval_tavily_serpapi_lens.yaml | Tavily | SerpAPI | TAVILY_API_KEY, SERPAPI_API_KEY |
| retrieval_exa_serpapi_lens.yaml | Exa (search + pages) | SerpAPI | EXA_API_KEY, SERPAPI_API_KEY |

Every overlay also passes `JINA_API_KEY` (if set) to fetch_webpage's Jina Reader. Anonymous
Jina is rate-limited per IP; under search-heavy runs it answers HTTP 401, so set the key there.
With `retrieval.fetch_provider: exa` fetch_webpage reads pages through Exa's `/contents`
endpoint instead (needs `exa_api_key`; no Jina), so Exa alone serves text search and pages.
Exa returns pages from its index, so a URL it has not crawled comes back as a tool error.

Benchmark-contamination guard (on by default): a result quoting Agentic-MME itself (its
dataset, repo or papers about it) would hand the model the answer key. Search and Lens items
matching `retrieval.contamination_patterns` (case-insensitive substrings over the whole item:
the `agentic-mme` spellings, `crystal1047`, `chos3ne11ven`) are dropped one by one. The model
is never told: a search left empty reads "No results.", and a page that mentions the benchmark
(or a fetch of a benchmark URL) fails with the same error as a page without content. Filtering
happens on every read, live or cached, and the cache keeps the raw answer. The tool trace
records `"contaminated": <n dropped>`; the model does not see it.
`contamination_patterns: []` disables the guard.

Google Lens needs the image online: it is uploaded to SerpAPI's own Image API (`POST /image`,
re-encoded as JPEG under 500 KB; Lens then takes the `image_id`). No other image host is used,
so Serper's Lens (which needs a public URL) is not offered. `allow_image_upload: true` is required.

Add `retrieval_cache.yaml` after one of them to answer from a shared on-disk cache
before calling the provider (`AGENTIC_MME_SEARCH_CACHE=<dir>`):

    export AGENTIC_MME_SEARCH_CACHE=/shared/agentic_mme_search_cache TAVILY_API_KEY=... SERPAPI_API_KEY=...
    config_paths="benchmarks/agentic_mme/config.yaml,benchmarks/agentic_mme/retrieval_tavily_serpapi_lens.yaml,benchmarks/agentic_mme/retrieval_cache.yaml"
    ng_prepare_benchmark "+config_paths=[$config_paths]"   # then run with the same config_paths

- google_search: exact normalised query. Fuzzy matching is off by default
  (`cache_fuzzy_threshold: 0`), because a near match can be a different question
  ('capital of australia' vs 'capital of austria'). With a threshold set, a query
  that misses exactly may match an earlier one (`token_sort_ratio >= threshold` over
  FTS5 candidates from `<dir>/index/master.db`, rejected when the two queries contain
  different numbers); the output then carries `"matched_query"`, so the model sees
  which question the results answer.
- google_lens_search: SHA-256 of the image pixels; fetch_webpage: normalised URL.
- Misses go to the provider and are written back (`cache_write: false` to only read);
  new query norms also go to `<dir>/index/search_norms_<host>_<pid>.txt`, and are
  fuzzy-matchable in that process at once. To share them with other processes, rebuild
  the index between runs: `python -m responses_api_agents.agentic_mme_agent.search_cache <dir>`. Files are content-addressed and written atomically, so many nodes
  can share one directory.
- Each retrieval output carries `"cache": "exact" | "fuzzy" | "miss"`, so a run's
  provider usage is the count of `miss` entries in `tool_trace`.

Layout: `search/`, `lens/`, `browse/` hold `<ab>/<cd>/<sha256(key)>.json` entries;
`index/master.db` is the FTS5 index. Everything (cache, fuzzy index, rebuild) lives in
`responses_api_agents/agentic_mme_agent/` and needs only `rapidfuzz` beyond the stdlib.

Live model/provider smoke testing and baselining are required before marking
verified=true. Unit tests are not evidence of benchmark performance.

## Licensing

New Gym code: Apache-2.0. The official dataset card declares Apache-2.0.
No reference-repository implementation or paper text is vendored: the reference
code checkout has no declared code license. Tool interfaces were independently
implemented using Pillow (HPND), NumPy (BSD) and OpenCV (Apache-2.0).
