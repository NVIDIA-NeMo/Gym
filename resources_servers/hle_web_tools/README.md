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
token lands in a fresh session that carries no question. Every blocklist family applies on both routes,
and `web_search` is refused for any session that was not seeded with a question
(`require_seeded_question`, default on), so such a session fails closed rather than being served without
the span rule. A seeded session keeps its question while the server holds it (see the session cap under
Blocklist semantics): a retried seed with the same
question is served again, and a seed that would change or drop the question is refused with HTTP 409, so
a caller holding a session cannot swap the span check out from under it.

**Residual: sandbox-seeded sessions.** The span rule can still be bypassed. `/seed_session` is an
unauthenticated route, and a sandbox whose egress policy admits the tool-server host
(`network_access: model_and_tools` admits the whole host, so it always does) can `POST /seed_session` for
a **fresh** session with a question of its choice and search from that session; the 409 above refuses
re-seeding an existing session and does not cover a new one or one the session cap has forgotten. The
session cookie and MCP token are signed
with a key derived from public config (`get_session_middleware_key`: the server class name and the
instance name), so they are not a secret that could tell the agent server's seed from the sandbox's.
Closing this needs an authenticated seed channel, a secret the agent server holds and the sandbox never
sees; until then the span rule is a check on a cooperative harness, not on the sandboxed model. Tracking
issue: NVIDIA-NeMo/Gym#5845. The Diamond README lists this next to the backend and the oversized-page refusal
under Sandboxed agents (with tools) > Upstream web tools, and all three belong next to any with-tools number
from that preset.

## Tools

| Tool | Protocol | Here |
| --- | --- | --- |
| `web_search(query)` | URLs, titles and snippets; at most 20 results per call before filtering | Tavily `/search` with `max_results: 20`, `search_depth: advanced`, no generated answer, no page bodies, no images; every result is filtered by URL, title and snippet; the survivors (at most 20) are rendered as `[n] title / URL / Snippet`, a title longer than 1,000 characters cut with `…` (the text families still see the whole title) |
| `web_fetch(url)` | main content of a page or PDF as Markdown, capped at 60,000 characters with a truncation indicator; no image, audio or video payloads | Tavily `/extract` with `format: markdown` (PDFs come back as text); `data:` URIs (base64 or percent-encoded, with or without a media type of at most 127 characters per type and subtype, and with up to 16 `name=value` media-type parameters, each name at most 127 characters and each value at most 256; a URI past these bounds is left in place) are replaced by `[data URI removed]` up to the first whitespace, quote, `)` or `>`, so a payload wrapped over several lines keeps its later lines; the Markdown is cut at `max_fetch_chars` (60,000) with `[Content truncated at 60,000 characters; N characters omitted]`; the `Title:` line above it (the provider's title, else the first heading) is cut at 1,000 characters with `…` |

Tavily is the backend for both, a difference from upstream's wrapper next to the sandbox-seeded session
residual above and the oversized-page refusal below (a page longer than `max_page_chars` is refused rather
than served cut at 60,000 characters): the protocol does not name a search or extraction provider, so which
pages a query surfaces, how they are ranked and how a page is turned into Markdown are Tavily's. The
protocol's filtering, caps and refusals are all applied here, after Tavily returns. The provider is reached through Gym's shared aiohttp client
(`nemo_gym.client.http_client.request`) rather than the `tavily-python` SDK: two JSON endpoints need no SDK,
and the server keeps the key pool, retry budget and body redaction that `resources_servers/tavily_search`
also implements over its own transport.

## Blocklist semantics

Every family of `blocklist.json` is enforced with the meaning the protocol gives it:

| Family | Rules | Applied to | Semantics |
| --- | --- | --- | --- |
| `domains` | 220 | every URL (requested, returned, redirected, wrapped) and `site:` operands | exact hostname or any subdomain, after UTS #46 mapping (as browsers and `yarl` map a host), IDNA and lower-casing |
| `url_prefixes` | 66 | every URL | prefix of the URL with the scheme and a leading `www.` removed, ignoring case; tried on the URL as written and on its canonical form; a prefix with a query (`openreview.net/pdf?id=...`) also matches its path as a prefix with each of its `key=value` pairs among the URL's query pairs, in any order (`pdf?&id=...`, `pdf?x=1&id=...`) |
| `serp_url_patterns` | 12 | every URL | regexes anchored at the start of the scheme-less URL; `www.` is kept, the patterns allow subdomains themselves; tried on the URL as written and on its canonical form |
| `url_patterns` | 22 | queries, URLs, result titles and snippets, fetched titles and Markdown | case-insensitive regexes |
| `patterns_case_sensitive` | 4 | the same | regexes as written (`\bHLE\b ...`) |
| `answer_title_patterns` | 5 | result titles | case-insensitive; tried on the title as shown (cut at 1,000 characters, with and without the cut marker) and on the title with each trailing ` \| Site`, ` - Site`, ` – Site`, ` · Site`, ` » Site`, ` :: Site` segment removed whose separator starts within those 1,000 characters, however far the segment runs; an answer phrase that lies wholly past the cut is neither shown nor checked |
| `answer_title_allow_hosts` | 7 | result titles | hosts (and subdomains) exempt from `answer_title_patterns` only; every other family still applies |

The canonical form is the URL as a fetcher resolves it: the host UTS #46-mapped and IDNA-encoded, lower-cased and without a
trailing dot, userinfo and a default port (`:443`, `:80`) dropped, `.` and `..` path segments resolved, runs
of `/` collapsed, the fragment dropped. Two canonical forms are checked: dot segments resolved before slash
runs are collapsed (a WHATWG client resolves dots before it sends the path), and slash runs collapsed first
(httpx sends `%2e%2e` unresolved, and nginx's `merge_slashes` then reads `abs/x//../y` as `abs/y`). So `arxiv.org:443/abs/...`, `arxiv.org./abs/...`, `arxiv.org/x/../abs/...`,
`arxiv.org//abs/...` and `arxiv.org/abs/x//%2e%2e/...` all match the prefix `arxiv.org/abs/...`.

Beyond the families, the protocol's own refusals:

- **Queries** are refused (each check runs on the query as written and NFKC-folded, with format
  characters such as zero-width spaces, soft hyphens and word joiners, and every other default-ignorable
  code point such as variation selectors and Hangul fillers, read three ways: all removed, all as a space,
  and a third reading in which a zero-width space separates words and every other one is removed, which is
  how UAX #29 word segmentation reads format and extend characters; Hangul fillers are letters to it, and
  this server drops them too, as NFKC_Casefold does) when they match `url_patterns` or
  `patterns_case_sensitive`, when a `site:`
  operand names a blocked domain (the operand is read as host characters only, so `(site:a.org OR
  site:b.org)`, `site:a.org|site:b.org` and `site:b.org,` each yield the bare host; typographic quotes
  and guillemets around it (`site:“b.org”`, `site:«b.org»`) are read like ASCII quotes, and stacked
  openers (`site:((b.org))`) are skipped; one leading dot, the
  scheme, `www.`, any path and trailing punctuation (`site:huggingface.co!`) are stripped before the
  check; an operand that is then not a hostname of letters, digits, `-` and `_` labels, optionally after
  `*.`, is refused as `malformed_url` rather than skipped), or when they
  repeat a long verbatim span of the question. Upstream states no span length; this server refuses a
  query that contains **8 consecutive words of the question** (`verbatim_span_words`, default 8), where
  words are case-folded maximal runs of letters and digits after that folding and with combining marks
  removed, so punctuation, case, whitespace, underscores, full-width or mathematical letters, zero-width
  and other default-ignorable characters read in any of those three ways, and accents do not evade the rule. Letters from another script that look alike (a Cyrillic
  `о` for a Latin `o`) are not folded. A question shorter than the threshold is protected as a whole, in each of
  those three readings, so a reading that splits a word at an ignorable character never blocks a fragment of it on
  its own. The question
  reaches the server through `/seed_session`: an Agent seeds its tool servers with its run request,
  whose extra fields are the task row (`question` for the HLE presets, configurable as
  `question_field`); the last user message of `responses_create_params.input` is the fallback, and an
  Environment Server's `task_data` is read the same way. A search from a session that was not seeded
  with a question (a seed without one or whose question has no words, such as `"?!"`, or a plain-HTTP
  call carrying no session) is refused with the
  family `unseeded_session`, and a seed without a question is logged at WARNING; set
  `require_seeded_question: false` to serve such sessions without the span rule instead. A seeded
  session cannot be re-seeded with another question (HTTP 409; a retry with the same question is
  served). An Environment Server's episode seed (`resources_session_id`, `episode_id`, `task_id`,
  `task_data`) also binds the session to its episode and task: a seed of the same ID for another episode
  or task, or a close naming another episode, gets HTTP 409, and after a close the ID is fenced for
  `session_close_retry_window_seconds`, so a delayed seed cannot recreate it. At most 10,000 seeded
  sessions are held; past that the least recently seeded one is forgotten, question and episode binding
  together, with a WARNING naming it, and its searches are refused as `unseeded_session`; a forgotten
  episode session's ID is fenced as if its episode had closed it, so a later seed of that ID gets HTTP 409.
  Before then a slot is released only by an episode session's `/close_session` or, when checkpointing is
  configured, by the checkpoint controller retiring the attempt that seeded the session
  (`retire_session_state` releases every session of that attempt, whether an episode seed or an agent-run
  seed created it, and does not fence the ID). A session an Agent seeds
  with its run body, which is how the OpenCode preset seeds, is otherwise never closed, so held sessions add up over
  the server's life, and a rollout's question is forgotten once 10,000 later sessions are seeded on the same
  server while that rollout is still running, however few rollouts are open at once. For a harness that
  closes its sessions, the cap must exceed the peak number of sessions open at once. A forgotten session
  that an agent-run seed created is not fenced, and neither is an episode session's ID against a later
  agent-run seed: a later seed
  carrying its cookie installs whatever question it brings (HTTP 200), so once the cap forgets a session,
  a caller holding its cookie can swap its span check. This is the same unauthenticated-seed residual as a
  sandbox-seeded fresh session (above), and so is a flood of unauthenticated seeds past the cap, which
  makes the server forget other rollouts' questions, so their searches fail closed as `unseeded_session`.
  The words of a blocked span are never logged, only their count.
- **Fetch URLs** must be `http(s)` URLs with a host, at most 4,096 characters long, with no userinfo
  (no `@` anywhere in the authority, even an empty `https://@host`) and no backslash, whitespace or
  control character anywhere: those are the characters on which WHATWG fetchers and Python's
  `urlsplit` disagree about where the host ends (`https://blocked.example\.other.example/` is
  `blocked.example` to a browser). Anything else is refused as `malformed_url`. The URL, its
  percent-decoded forms and every URL wrapped inside any of them are then expanded breadth first and
  each is checked against the URL-shaped families and the text families: the path remainder of the
  Wayback Machine and every other `*.archive.org` host (`web.archive.org/web/[<timestamp>/]<url>`, where
  a first segment after `/web/` that is not shaped like a timestamp (`20240101000000`, `2024id_`, `2024*`,
  `*`) is read both as a timestamp and as the start of the URL, since archive.org redirects
  `/web/scale.com/x` to a snapshot of `scale.com/x`; `wayback.archive.org/...`), of `r.jina.ai`
  and of Google's cache (`webcache.googleusercontent.com/search?q=cache:<url>`), each read as a URL with
  or without a scheme
  and whatever the spelling of its host (`scale.com./x`, `x@scale.com/x` and full-width letters all name
  `scale.com`); every query-string value that names a host (`?url=`, `?u=`, `?q=`, ...), with or without
  a scheme; and any `http(s):/` found inside the URL, each taken up to the next one. A wrapped or decoded
  URL that Python cannot parse (`https://[::1]@scale.com/`, which browsers read as `scale.com`) is refused
  as `malformed_url`. A wrapped `https:/host`, `https:///host` or `https:host` is read as
  `https://host`, and a wrapped or decoded URL is also read as a WHATWG fetcher reads it (backslash as
  `/`, tab and newlines deleted). The expansion checks the original and four further rounds of decoding
  and unwrapping, and at most 64 distinct candidates; a URL that still has unexplored candidates when
  either bound is reached is refused as `malformed_url` instead of being certified on a partial check.
  The other wrappers upstream names (`archive.ph`, `archive.today`, `archive.is`, `archive.md`,
  `archive.li`, `archive.vn`, `archive.fo`, `translate.goog`, `translate.google.com`, `r.jina.ai`,
  `12ft.io`, `corsproxy.io`, `allorigins.win`, `codetabs.com`, `thingproxy.freeboard.io`,
  `whateverorigin.org`) are blocked outright by the `domains` family.
- **Fetched responses** are checked after the fetch: the final URL the backend reports in
  `results[].url` (a redirect to a blocked host is refused when Tavily reports the post-redirect URL
  there, which has not been verified against the live API), then the title and the content against
  `url_patterns` and
  `patterns_case_sensitive`. On a fresh fetch the content checked is the provider's whole page, before
  any data URI is removed and before any cut, and then the sanitized Markdown as well; a match anywhere,
  even past the 60,000 characters the model would read, refuses the page whole. That scan is bounded: a
  page longer than `max_page_chars` (1,000,000) is refused unread with the family `oversized_page`, and
  the page work (sanitizing, the heading fallback for the title, the content checks, and the rechecks on
  a cache hit) runs in a pool of two worker processes started with `spawn`, not on the event loop. A
  thread would not do: Python's `re` keeps the GIL for a whole scan, and one scan of an adversarial page
  takes seconds (about 4 s for 1,000,000 characters of `抱抱脸` against the pinned blocklist). The pool is
  created on the first page check and starts its workers on demand, up to two; each worker imports the
  server module afresh, so it is about the size of the server process (about 240 MB here), and the first
  fetch waits about 2 s for it. `web_search`'s checks of result URLs, titles and snippets still run on the
  event loop: a provider title or snippet of 1,000,000 such characters would hold it about 4 s (whether
  Tavily ever returns one that long is not verified). A refused page is never cached. The page cache
  (`max_cached_pages`, keyed by requested URL) keeps the sanitized Markdown whole and runs the same
  final-URL, title and content checks on a hit, so a policy that changes after a page was cached still
  applies; those cache-hit rechecks see the sanitized cached Markdown, not the provider's raw page. A
  page whose sanitized Markdown is longer than `max_cached_page_chars` is served but not cached.

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
| `tavily_api_key` | required | one key, a list, or a comma-separated string (a repeated key counts once); a key Tavily rejects (401, 432, 433) is skipped for the rest of the call without spending the retry budget, and the call fails once every key is rejected |
| `blocklist_file_path` | `resources_servers/hle_web_tools/blocklist.json` | the policy file; resolved under the cwd or the install root |
| `blocklist_sha256` | upstream digest | the digest the file must have; `null` accepts any file |
| `verbatim_span_words` | 8 | consecutive question words that refuse a query |
| `question_field` | `question` | the seed-body field carrying the question |
| `require_seeded_question` | `true` | refuse `web_search` from a session that was not seeded with a question; `false` serves it without the span rule |
| `max_results` | 20 | results requested and returned per search (Tavily's and the protocol's cap) |
| `max_fetch_chars` | 60,000 | the Markdown cap |
| `max_query_chars` | 400 | Tavily rejects longer queries |
| `search_depth`, `extract_depth` | `advanced`, `basic` | Tavily depths |
| `max_page_chars` | 1,000,000 | a fetched page longer than this (the provider's Markdown, before anything is removed or cut) is refused unread as `oversized_page` |
| `max_cached_pages`, `max_cached_page_chars` | 128, 500,000 | the fetched-page cache: at most this many pages, and a page longer than this is served but not cached |
| `max_http_attempts`, `http_timeout_s` | 3, 60 | the retry budget per call and the timeout per attempt |
| `tavily_base_url` | `https://api.tavily.com` | the Tavily API base URL, for a proxy or a stub |

Run the server single-worker (leave `num_workers` unset). The per-session state (the seeded question and the
episode binding) is keyed by the session ID, which session routing pins to the worker that owns the session (a
resources request by its cookie's `session_id`, an MCP tool call by its token's `sid`), so that state alone
would not need one worker. What does: with `num_workers > 1` uvicorn imports the entrypoint in each worker and
serves a module-level `app`, which this module does not define, and the page cache (keyed by URL) and the
page-check pool are per process, so each worker would fetch, scan and cache its own copy of a page. The server
says so through `multiple_workers_unsupported_reason`, so a launch with `num_workers > 1` is refused before
uvicorn starts, dry runs included (`<instance name> needs num_workers=1: ...`, for the Diamond preset
`hle_tools needs num_workers=1: the module defines no module-level app for uvicorn workers, ...`).

`verify` is a stub: this is a tool server. A preset pairs it through the agent's `tool_servers` and
grades the rollout with its own `resources_server` (for HLE-Diamond, the equivalence judge). Should
`verify` be called anyway it returns reward 0 with `mask_sample: true` and a `failure_reason` saying so.

## Use in a preset

`benchmarks/hle_diamond/opencode_hle_tools.yaml` runs the OpenCode harness with this server as its only
tool server. Live runs of that preset need `TAVILY_API_KEY` in the environment of the process that starts
the servers; see that benchmark's README.

## Example data

`data/example.jsonl` holds 5 synthetic placeholder rows (general-knowledge questions, not HLE items; each
row carries a `placeholder` field saying so) for the data validation CI runs on every resources server.
`task_data.py` declares the row schema that validation checks: an optional `question` (read by
`/seed_session`) and the optional `placeholder` note, which the server never reads.
The `hle_web_tools_simple_agent` and `hle_web_tools_environment_server` instances in
`configs/hle_web_tools.yaml` exist for that check and reach the tools over the plain HTTP routes.
`data/example_metrics.json` was generated from those rows with
`gym dataset collate ... +mode=example_validation`; `data/example_rollouts.jsonl` holds `<PLACEHOLDER ROW>`
lines, as in `resources_servers/tavily_search`, because no rollout was collected (that needs a policy
model and a Tavily key), and the server stays `verified: false`.

## Tests

```bash
pytest resources_servers/hle_web_tools/tests/test_app.py
```

Model-free; the Tavily transport is stubbed. Each rule family has one test built on rules taken from
the upstream blocklist, plus the two query refusals, the fail-closed refusal of a cookie-less
`POST /web_search` after a seed, the 409 on re-seeding a session with another question, the episode-session
contract (`nemo_gym.server_runtime.testing.session_conformance.check_resources_session_contract`), wrapper decoding, the cap and marker, the exact two-tool list over a
real MCP `tools/list`, the 20-result cap, the cache re-check, the redirect check, key failover and exhaustion of the key
pool, malformed provider bodies, including a 200 whose body is not JSON (a tool error, not an HTTP 500) and the blocklist digest pin. A table of URL
spellings from review (expansion overflow, a fourth decoded form, backslash, whitespace and userinfo,
canonical-form prefixes, archive.org variants) must be refused before the fetch while ordinary redirect and
query URLs stay allowed; `site:` operands are parsed to hostnames; long URLs and a 100,000-character heading
are handled within a second; a slow page check leaves the event loop free; data URIs are stripped on fresh and cached fetches without eating the prose
after them; and a blocked span past every cut still refuses the page.

## Licensing

`blocklist.json` is a byte-exact, unmodified copy of `docs/blocklist.json` from
[`centerforaisafety/hle`](https://github.com/centerforaisafety/hle) at commit
`22ed3074b1e7b134bcbc09028d0ba320839b0655`, which is MIT-licensed (Copyright (c) 2025 centerforaisafety). A
JSON file cannot carry the notice without changing the digest the server verifies, so the upstream
[`LICENSE`](https://github.com/centerforaisafety/hle/blob/22ed3074b1e7b134bcbc09028d0ba320839b0655/LICENSE)
at that commit is reproduced unchanged next to it in [`LICENSE-HLE`](LICENSE-HLE), and the file is listed
under Vendored Components in the repository's `ATTRIBUTIONS.md`. The rest of this server is NVIDIA code
under Apache-2.0.
