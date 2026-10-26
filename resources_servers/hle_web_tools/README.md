# HLE web tools

The two web tools of the HLE authors' with-tools protocol, `web_search` and `web_fetch`, behind their
blocklist, served to a sandboxed agent over MCP. The protocol is
[`docs/evaluation-with-tools.md`](https://github.com/centerforaisafety/hle/blob/22ed3074b1e7b134bcbc09028d0ba320839b0655/docs/evaluation-with-tools.md)
in `centerforaisafety/hle` at commit `22ed3074b1e7b134bcbc09028d0ba320839b0655`; `app.py` records the
SHA-256 of that document, of `docs/blocklist.json` and of `docs/requirements-tools.txt` at that commit,
and `blocklist.json` next to it is a byte-exact copy of the upstream blocklist whose digest the server
verifies at start (`blocklist_sha256`; set it to `null` to run a deliberately different policy file).

The server holds the search provider's credentials. The sandbox receives the per-rollout MCP session
token minted by `/seed_session`, and the token lists exactly these two tools. Gym's MCP exposure keeps
the plain `POST /web_search` and `POST /web_fetch` routes served next to `/mcp`, and a sandbox whose
egress policy admits the tool-server host can call them directly; a call without the rollout's cookie or
token lands in a fresh session that carries no question. The policy therefore does not depend on the
transport: every blocklist family applies on both routes, and `web_search` is refused for any session
that was not seeded with a question (`require_seeded_question`, default on), so the one session-dependent
rule fails closed instead of being skipped. What this server cannot tell apart is a seed sent by the
agent server from one sent by the sandbox itself: keeping the sandbox from seeding its own session is the
job of the egress policy, which the Diamond README discusses under Sandboxed agents.

## Tools

| Tool | Protocol | Here |
| --- | --- | --- |
| `web_search(query)` | URLs, titles and snippets; at most 20 results per call before filtering | Tavily `/search` with `max_results: 20`, `search_depth: advanced`, no generated answer, no page bodies, no images; every result is filtered by URL, title and snippet; the survivors (at most 20) are rendered as `[n] title / URL / Snippet` |
| `web_fetch(url)` | main content of a page or PDF as Markdown, capped at 60,000 characters with a truncation indicator; no image, audio or video payloads | Tavily `/extract` with `format: markdown` (PDFs come back as text); `data:` URIs are removed; the Markdown is cut at `max_fetch_chars` (60,000) with `[Content truncated at 60,000 characters; N characters omitted]` |

Tavily is the backend for both, which is the one residual difference from upstream's wrapper: the
protocol does not name a search or extraction provider, so which pages a query surfaces, how they are
ranked and how a page is turned into Markdown are Tavily's. The protocol's filtering, caps and refusals
are all applied here, after Tavily returns. The provider is reached through Gym's shared aiohttp client
(`nemo_gym.server_utils.request`) rather than the `tavily-python` SDK: two JSON endpoints need no SDK,
and the server keeps the key pool, retry budget and body redaction that `resources_servers/tavily_search`
also implements over its own transport.

## Blocklist semantics

Every family of `blocklist.json` is enforced with the meaning the protocol gives it:

| Family | Rules | Applied to | Semantics |
| --- | --- | --- | --- |
| `domains` | 220 | every URL (requested, returned, redirected, wrapped) and `site:` operands | exact hostname or any subdomain, after IDNA and lower-casing |
| `url_prefixes` | 66 | every URL | prefix of the URL with the scheme and a leading `www.` removed, ignoring case |
| `serp_url_patterns` | 12 | every URL | regexes anchored at the start of the scheme-less URL; `www.` is kept, the patterns allow subdomains themselves |
| `url_patterns` | 22 | queries, URLs, result titles and snippets, fetched titles and Markdown | case-insensitive regexes |
| `patterns_case_sensitive` | 4 | the same | regexes as written (`\bHLE\b ...`) |
| `answer_title_patterns` | 5 | result titles | case-insensitive; also tried on the title with each trailing ` \| Site`, ` - Site`, ` – Site`, ` · Site`, ` » Site`, ` :: Site` segment removed |
| `answer_title_allow_hosts` | 7 | result titles | hosts (and subdomains) exempt from `answer_title_patterns` only; every other family still applies |

Beyond the families, the protocol's own refusals:

- **Queries** are refused when they match `url_patterns` or `patterns_case_sensitive`, when a `site:`
  operand names a blocked domain (scheme, `www.` and path stripped before the check), or when they
  repeat a long verbatim span of the question. Upstream states no span length; this server refuses a
  query that contains **8 consecutive words of the question** (`verbatim_span_words`, default 8), where
  words are case-folded maximal runs of word characters, so punctuation, case and whitespace differences
  do not evade the rule. A question shorter than the threshold is protected as a whole. The question
  reaches the server through `/seed_session`: an Agent seeds its tool servers with its run request,
  whose extra fields are the task row (`question` for the HLE presets, configurable as
  `question_field`); the last user message of `responses_create_params.input` is the fallback, and an
  Environment Server's `task_data` is read the same way. A search from a session that was not seeded
  with a question (a seed without one, or a plain-HTTP call carrying no session) is refused with the
  family `unseeded_session`, and a seed without a question is logged at WARNING; set
  `require_seeded_question: false` to serve such sessions without the span rule instead. The words of a
  blocked span are never logged, only their count.
- **Fetch URLs** must be `http(s)` without credentials. The URL, its percent-decoded forms (up to four
  rounds, so double encoding does not help) and every URL wrapped inside any of them are checked
  against the URL-shaped families and the text families: the path remainder of the Wayback Machine
  (`web.archive.org/web/<timestamp>/<url>`), of `archive.org`, of `r.jina.ai` and of Google's cache
  (`webcache.googleusercontent.com/search?q=cache:<url>`); every query-string value that names a host
  (`?url=`, `?u=`, `?q=`, ...), with or without a scheme; and any `https?://` found inside the URL.
  The other wrappers upstream names (`archive.ph`, `archive.today`, `archive.is`, `archive.md`,
  `archive.li`, `archive.vn`, `archive.fo`, `translate.goog`, `translate.google.com`, `r.jina.ai`,
  `12ft.io`, `corsproxy.io`, `allorigins.win`, `codetabs.com`, `thingproxy.freeboard.io`,
  `whateverorigin.org`) are blocked outright by the `domains` family.
- **Fetched responses** are checked after the fetch: the final URL the backend reports (a redirect to a
  blocked host is refused), then the title and the full Markdown against `url_patterns` and
  `patterns_case_sensitive`. A page that fails is refused whole and evicted from the cache. The page
  cache (`max_cached_pages`, keyed by requested URL) runs exactly the same checks on a hit, so a policy
  that changes after a page was cached still applies.

Every refused call is logged at WARNING with the tool, the session, the family, the rule, where it
matched and the first 200 characters of the matched subject; every search result dropped by the filter
is logged at INFO in the same form:

```text
WARNING hle_web_tools blocked a web_search call (session 3f...): family=question_span rule='8 consecutive words of the question' where=query subject='8 words'
INFO    hle_web_tools dropped a web_search result (session 3f...): family=domains rule='huggingface.co' where=result URL subject='https://huggingface.co/datasets/...'
```

The model reads a refusal that names the reason at the family level ("the source is a blocked site",
"the query repeats 8 consecutive words of the question verbatim; rephrase it in your own words"), never
the rule text.

## Configuration

```yaml
hle_tools:
  resources_servers:
    hle_web_tools:
      entrypoint: app.py
      expose_tools_over_mcp: true
      tavily_api_key: ${oc.env:TAVILY_API_KEY}   # one key or a comma-separated pool
      verified: false
```

| Key | Default | Meaning |
| --- | --- | --- |
| `tavily_api_key` | required | one key, a list, or a comma-separated string; a key Tavily rejects (401, 432, 433) is skipped for the rest of the call |
| `blocklist_file_path` | `resources_servers/hle_web_tools/blocklist.json` | the policy file; resolved under the cwd or the install root |
| `blocklist_sha256` | upstream digest | the digest the file must have; `null` accepts any file |
| `verbatim_span_words` | 8 | consecutive question words that refuse a query |
| `question_field` | `question` | the seed-body field carrying the question |
| `require_seeded_question` | `true` | refuse `web_search` from a session that was not seeded with a question; `false` serves it without the span rule |
| `max_results` | 20 | results requested and returned per search (Tavily's and the protocol's cap) |
| `max_fetch_chars` | 60,000 | the Markdown cap |
| `max_query_chars` | 400 | Tavily rejects longer queries |
| `search_depth`, `extract_depth` | `advanced`, `basic` | Tavily depths |
| `max_cached_pages`, `max_cached_page_chars` | 128, 500,000 | the fetched-page cache |
| `max_http_attempts`, `http_timeout_s` | 3, 60 | the retry budget per call and the timeout per attempt |

`verify` is a stub: this is a tool server. A preset pairs it through the agent's `tool_servers` and
grades the rollout with its own `resources_server` (for HLE-Diamond, the equivalence judge). Should
`verify` be called anyway it returns reward 0 with `mask_sample: true` and a `failure_reason` saying so.

## Use in a preset

`benchmarks/hle_diamond/opencode_hle_tools.yaml` runs the OpenCode harness with this server as its only
tool server. The checks below need `TAVILY_API_KEY` in the environment of the process that starts the
servers and are described in that benchmark's README.

## Tests

```bash
pytest resources_servers/hle_web_tools/tests/test_app.py
```

Model-free; the Tavily transport is stubbed. Each rule family has one test built on rules taken from
the upstream blocklist, plus the two query refusals, the fail-closed refusal of a cookie-less
`POST /web_search` after a seed, wrapper decoding, the cap and marker, the exact two-tool list over a
real MCP `tools/list`, the 20-result cap, the cache re-check, the redirect check, key failover and the
blocklist digest pin.
