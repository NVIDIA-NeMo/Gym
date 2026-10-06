# TimeWarp resources server

[TimeWarp](https://github.com/sparklabutah/timewarp) ([paper](https://arxiv.org/abs/2603.04949)) measures how
robust web agents are to changes in a website's UI over time. It has three sites, Wiki, News and Shop, each
rendered in six UI eras from early-2000s layouts to a minimal modern one. A task is a question about those
sites; the agent browses and answers in free text, and the answer is checked by deterministic verifiers.

This server gives each rollout an isolated browser context on one UI era, exposes browser tools to the policy,
and scores the policy's final message with TimeWarp's verifiers. The benchmark that uses it is
[`benchmarks/timewarp`](../../benchmarks/timewarp/README.md).

## Hosting the TimeWarp sites

The sites are TimeWarp's own Flask apps; this server does not start them. Set them up once with TimeWarp's
`setup.sh` (it downloads the site data from the
[`sparklabutah/timewarp-env-data`](https://huggingface.co/datasets/sparklabutah/timewarp-env-data) dataset and
builds the Shop's search index, which needs Java 21), then start the UI versions you need on the ports the
default config expects:

```bash
git clone https://github.com/sparklabutah/timewarp.git && cd timewarp && bash setup.sh && conda activate timewarp
bash /path/to/Gym/resources_servers/timewarp/scripts/start_sites.sh "$PWD" 1 2 3 4 5 6
```

| Site | UI version V is served at | Upstream variable |
| --- | --- | --- |
| Wiki | `http://localhost:510V` | `TW_WIKI` |
| News | `http://localhost:520V` | `TW_NEWS` |
| Shop | `http://localhost:530V/abc` | `TW_WEBSHOP` |

To serve the sites elsewhere, override `site_urls` (UI version to the three base URLs) in
`configs/timewarp.yaml`. A row whose `ui_version` has no entry fails at `seed_session`, as does a row whose
start site is unreachable.

Each Wiki process loads a ~0.5 GB index, so all six versions need several GB of memory. The Shop's search
engine (pyserini 1.3.0) constructs an OpenAI client when it is imported; if `OPENAI_API_KEY` is unset the Shop
fails to start, and any placeholder value works because the key is never used. The Shop themes load Bootstrap
and jQuery from public CDNs and product images from Amazon's image CDN, so the machine running the browser needs
internet access for the pages to render as they do upstream.

## Task contract

Each row carries `ui_version` (1-6), `start_site`, `sites`, `intent` and the task's TimeWarp `eval` block in
`verifier_metadata`. `seed_session` opens a browser context on the start site's home page for that UI version.

**Observation.** Playwright's AI-mode ARIA snapshot of the page (`aria_snapshot(mode="ai")`): an
accessibility tree where each actionable element carries a reference such as `[ref=e12]`, preceded by the URL
and title. Pages longer than `max_observation_chars` (default 12,000; a Wiki article is about 70,000) are
split into parts that the policy reads with `observe(part=N)`.

**Tools.**

| Tool | Arguments | Effect |
| --- | --- | --- |
| `observe` | `part` (optional) | Show the current page, or a later part of it |
| `open_site` | `site`: `wiki`, `news` or `shop` | Open that site's home page in this UI version |
| `goto` | `url` | Open a URL on this UI version's sites |
| `click` | `ref` | Click an element |
| `fill` | `ref`, `text` | Replace the text in an input |
| `press` | `ref`, `key` | Press a key, such as `Enter`, in an element |
| `select_option` | `ref`, `option` | Choose a dropdown option by value or label |
| `go_back`, `go_forward` | none | Move through the tab's history |

Every tool except `fill` returns the next observation. Tool errors (a stale ref, a timeout) come back to the
policy as text starting with `Error:`; they never fail the rollout.

**Sandbox.** The browser may only visit the three sites of the row's UI version. A `goto` elsewhere is refused,
and a link or redirect elsewhere is answered with HTTP 204 so the tab stays put; the next observation reports
the blocked URL. Upstream TimeWarp instead scores an episode 0 once a page leaves the TimeWarp sites.
Subresources are not filtered. Links and scripts that would open a new tab load in the current tab, so an
episode has one tab and `go_back` always works.

**Answer.** The episode ends when the policy replies with a message instead of a tool call; that message,
with any `<think>` block removed, is the answer. An episode stopped by the agent's `max_steps` while still
calling tools has no answer and scores 0, as an upstream episode with no `send_msg_to_user` does.

## Scoring

`verify` runs the verifiers listed in the task's `eval_types` and multiplies their scores (all must pass). The
verifier code is TimeWarp's, vendored in [`normalization.py`](normalization.py) and adapted in
[`scoring.py`](scoring.py); its matching rules are documented in the
[TimeWarp README](https://github.com/sparklabutah/timewarp#-how-tasks-are-scored).

| `eval_types` entry | Checks |
| --- | --- |
| `string_match` | `must_include` / `must_exclude` / `exact_match` on word boundaries, with `\|OR\|` alternatives, `^regex$` leaves and an optional `first_sentence` scope |
| `number_match` | Required numbers appear in any formatting (`7,000,000`, `7 million`, `seven million`), exactly or within a tolerance |
| `list_match` | Every listed item appears, in order when `ordered` is set |
| `exact_match` | Legacy whole-answer match after lowercasing and whitespace cleanup |
| `llm_judge` | TimeWarp's judge prompt against the `fuzzy_match` gold, sent to `judge_model_server` |

Only one task per split uses `llm_judge`. When `judge_model_server` is unset those rollouts are returned with
`mask_sample: true` and `failure_kind: timewarp:judge_not_configured`, so they are excluded from scores rather
than counted as failures. Upstream's default judge is GPT-5.1; any model server works, but keep it separate from
the policy. A malformed `eval` block raises instead of scoring 0, as upstream does.

Aggregate metrics add success rates per UI version (`v1/...` to `v6/...`) and per site (`wiki/`, `news/`,
`webshop/`, and `multi/` for cross-site tasks).

## Differences from the upstream harness

Upstream runs TimeWarp in BrowserGym with AgentLab's GenericAgent. This server keeps the tasks, the sites and
the verifiers, and replaces the harness with Gym-native tool calls so that rollouts go through a Gym model
server and can be used for RL training. Scores are therefore not directly comparable to the paper's:

- The observation is Playwright's ARIA snapshot, not BrowserGym's AXTree, and there are no screenshots, so the
  four test goals that need a product photo (ids 72-75) cannot be answered from the page text.
- The policy acts through function calls instead of BrowserGym action strings, and answers with its final
  message instead of `send_msg_to_user`.
- Navigation outside the sites is prevented instead of scored 0, and episodes are single-tab.
- `max_steps: 30` mirrors BrowserGym's step limit, but a Gym step is one model turn, which may issue several
  tool calls, and the first turn is usually spent on `observe`.

## Sessions and concurrency

The server keeps one Chromium per process and one browser context per rollout, so it requires
`num_workers: 1`. A context is closed when the rollout is verified or its cookie session is seeded again; a
rollout abandoned before `verify` keeps its context until the server exits. Playwright's Chromium is installed
on startup (`setup_chromium.py`); on Linux it also needs system libraries, which
`python -m playwright install-deps chromium` installs (root required). Sessions seeded by an Environment Server
(`ResourcesSeedSessionRequest`) are not supported yet; the default configs route rollouts through
`legacy_agent` and `simple_agent`.

## Running

```bash
gym env start --config resources_servers/timewarp/configs/timewarp.yaml --model-type openai_model
gym eval run --no-serve --agent timewarp_simple_agent \
  --input resources_servers/timewarp/data/example.jsonl \
  --output results/timewarp_smoke.jsonl --limit 5
```

`data/example.jsonl` holds five test-split tasks covering the three start sites, every deterministic verifier
and UI versions 1, 2, 3, 5 and 6, avoiding the goals that need a product photo; it was produced with `benchmarks/timewarp/prepare.py`'s `build_rows` and
`benchmarks/timewarp/prompt.yaml`.

Tests: `gym env test --resources-server timewarp`. They need no TimeWarp sites; `tests/test_browser.py`
drives a real Chromium against a small local site.

## Licensing

- Server code: Apache-2.0.
- `normalization.py` and the verifier logic in `scoring.py`: vendored from TimeWarp (commit `4978e69`), which
  declares the MIT License; the MIT notice is kept in both files and the component is listed in
  `ATTRIBUTIONS.md`.
- Task data ([`sparklabutah/timewarp`](https://huggingface.co/datasets/sparklabutah/timewarp)) and site data
  ([`sparklabutah/timewarp-env-data`](https://huggingface.co/datasets/sparklabutah/timewarp-env-data)): MIT.
  The site data is not redistributed here.
- Playwright: Apache-2.0.
