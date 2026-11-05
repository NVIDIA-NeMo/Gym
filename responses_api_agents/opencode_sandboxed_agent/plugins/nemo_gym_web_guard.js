// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Keeps benchmark answer keys and TDM opt-out domains out of the model's search results.
//
// TWO CONCERNS, ONE MODULE (ported from the nv-opencode fork's tool/webfilter.ts):
//
//   1. CONTAMINATION. BrowseComp's answers are published on HuggingFace, so a model that
//      searches for its own question can find the answer key. Measured on a BrowseComp-400
//      run, a noticeable share of trajectories contained a marker. A score produced without
//      this filter is inflated and is not comparable to the legacy stack's scrubbed numbers.
//   2. EXCLUDE-DOMAINS (TDM opt-out compliance). Exa's MCP endpoint ignores excludeDomains
//      server-side, so exclusion has to happen on the response, client-side.
//
// WHY A REPLACEMENT TOOL RATHER THAN THE tool.execute.after HOOK. For opencode's builtin tools,
// output truncation runs inside the tool wrapper BEFORE the after-hook, and the truncator writes
// the FULL, UNFILTERED text to a file it then tells the model it can Grep. Filtering after the
// fact would leave the answer key on disk, one Grep away. A plugin-registered tool is truncated
// AFTER its execute returns, so filtering here happens before anything is persisted. Registering
// a tool under an existing name replaces the builtin: the tool table is a plain map and plugin
// entries are written last.
//
// INVERSION TRAP: an MCP-provided tool overwrites a same-named plugin tool. If search ever moves
// to a config `mcp` server this shadowing silently stops applying — assert on the sentinel log.
//
// Unset OPENCODE_WEBSEARCH_EXCLUDE_DOMAINS makes concern 2 a no-op. The contamination guard is
// on unless NEMO_GYM_CONTAMINATION_GUARD=0, so that "off" is always a deliberate, logged arm.

const PLUGIN = "[nemo-gym-plugin]";
const EMPTY_RESULT = "No results returned"; // neutral: never says why something was dropped

const CONTAMINATION_PATTERNS = [
  "browsecomp",
  "browse_comp",
  "browse-comp",
  "simple-eval",
  "bcplus",
  "bc-plus",
  "bc_plus",
];

// Matched against the URL only. Folding these into the text list would drop any page that merely
// links to a HuggingFace dataset, which is a large false-positive class.
const CONTAMINATED_URL_SUBSTRINGS = ["huggingface.co/datasets", "datasets-server.huggingface.co"];

// "browse_comp" is a substring of "browse_companies" — legitimate finance pages, not leakage.
const BENIGN_BROWSE_COMP = /browse[-_]?compan/g;

let cachedDomains;
function excludedDomains() {
  if (cachedDomains !== undefined) return cachedDomains;
  const raw = (process.env.OPENCODE_WEBSEARCH_EXCLUDE_DOMAINS || "").trim();
  if (!raw) return (cachedDomains = []);
  let list = [];
  if (raw.startsWith("[")) {
    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) list = parsed.map(String);
    } catch {
      list = [];
    }
  } else {
    list = raw.split(/[,:]/);
  }
  cachedDomains = list.map((d) => d.trim().toLowerCase().replace(/^\*?\./, "")).filter(Boolean);
  return cachedDomains;
}

function isExcludedUrl(url) {
  const domains = excludedDomains();
  if (domains.length === 0) return false;
  let host;
  try {
    host = new URL(url).hostname.toLowerCase();
  } catch {
    return false;
  }
  // Suffix match on a label boundary: "allure.com" must match "www.allure.com" but not
  // "notallure.com".
  return domains.some((d) => host === d || host.endsWith("." + d));
}

function isContaminated(text) {
  const low = (text || "").toLowerCase();
  for (const p of CONTAMINATION_PATTERNS) {
    if (!low.includes(p)) continue;
    if (p.startsWith("browse") && !low.replace(BENIGN_BROWSE_COMP, "").includes(p)) continue;
    return p;
  }
  return undefined;
}

function guardEnabled() {
  return process.env.NEMO_GYM_CONTAMINATION_GUARD !== "0";
}

// Exa returns one pre-formatted blob rather than a results array, so per-result work means
// splitting on the record boundary: records start at a line-anchored "Title: " and carry a "URL:".
function filterSearchOutput(text, fn) {
  if (!text) return text;
  const records = text.split(/^(?=Title: )/m).filter((r) => r.length > 0);

  if (records.length === 0) {
    // Unrecognised shape (provider format drift). Fall back to whole-blob contamination so a
    // leak cannot slip through unchecked; exclusion is skipped, there being no URL to test.
    const p = guardEnabled() && isContaminated(text);
    if (p) {
      process.stderr.write(PLUGIN + " contamination fn=" + fn + " unparsed_blanked=1 pattern=" + p + "\n");
      return EMPTY_RESULT;
    }
    return text;
  }

  const kept = [];
  let droppedContam = 0;
  let droppedDomain = 0;
  for (const rec of records) {
    const url = (rec.match(/^URL:\s*(\S+)/m) || [])[1] || "";
    const lowUrl = url.toLowerCase();
    if (guardEnabled() && CONTAMINATED_URL_SUBSTRINGS.some((s) => lowUrl.includes(s))) {
      droppedContam++;
      continue;
    }
    if (guardEnabled() && isContaminated(rec)) {
      droppedContam++;
      continue;
    }
    if (url && isExcludedUrl(url)) {
      droppedDomain++;
      continue;
    }
    kept.push(rec);
  }

  if (droppedContam)
    process.stderr.write(
      PLUGIN + " contamination fn=" + fn + " dropped=" + droppedContam + " kept=" + kept.length + "\n",
    );
  if (droppedDomain)
    process.stderr.write(
      PLUGIN + " excludedomains fn=" + fn + " dropped=" + droppedDomain + " kept=" + kept.length + "\n",
    );
  if (droppedContam === 0 && droppedDomain === 0) return text;
  // Everything dropped should look like an empty provider response, never an explanation.
  return kept.length === 0 ? EMPTY_RESULT : kept.join("");
}

function filterFetchOutput(text, fn) {
  if (!guardEnabled()) return text;
  const pattern = isContaminated(text);
  if (pattern) {
    process.stderr.write(PLUGIN + " contamination fn=" + fn + " blanked=1 pattern=" + pattern + "\n");
    return EMPTY_RESULT;
  }
  return text;
}

// --- Exa search -------------------------------------------------------------------------------
// Two transports, routed exactly as the fork routes them (tool/websearch.ts:66-87):
//
//   OPENCODE_WEBSEARCH_TYPE unset  -> MCP  (mcp.exa.ai/mcp, `web_search_exa`)   <- the DEFAULT
//   OPENCODE_WEBSEARCH_TYPE set    -> REST (api.exa.ai/search with that `type`)
//
// The default matters for comparability: the published legacy arms never set
// OPENCODE_WEBSEARCH_TYPE (absent from the comparator's agent_apptainer_command_str), so every
// legacy search went through MCP and the model saw Exa's SERVER-rendered SERP -- including
// Highlights, which the REST path cannot produce. Running REST here was the single largest
// uncontrolled retrieval difference between the two arms.
//
// WHY THE PLUGIN SPEAKS MCP INSTEAD OF A CONFIG `mcp:` SERVER. opencode builds its tool table
// last-wins and MCP tools are installed AFTER plugin tools (session/tools.ts:482 runs after :95),
// so declaring an `mcp` server named `exa` would OVERWRITE this shadow tool and silently take the
// contamination guard and the TDM exclusion offline -- the failure would look exactly like a
// healthy run. Calling MCP from inside the tool keeps the filter on the response path, which is
// also precisely what the fork does.
//
// mcp.exa.ai's `web_search_exa` schema is {query, numResults} only; it accepts and then DROPS
// type/livecrawl/contextMaxCharacters (re-verified by the fork 2026-09-09). They are sent anyway,
// because the fork sends them -- the goal is the same bytes on the wire, not a tidier request.

const MCP_TIMEOUT_MS = 25_000; // fork: McpExa.call(..., "25 seconds")
// Fetching a page is slower than a SERP lookup; the fork's own webfetch allowed 30s by default.
const MCP_FETCH_TIMEOUT_MS = 60_000;
const REST_TIMEOUT_MS = 30_000; // fork: restSearch non-deep branch
const REST_DEEP_TIMEOUT_MS = 120_000; // fork: restSearch deep branch
const REST_DEEP_TYPES = ["deep", "deep-lite", "deep-reasoning"];
const REST_DEEP_ANSWER_MAX_CHARS = 5000;

// Bun's fetch has no usable default deadline (a missing one cost whole campaigns to a 300s hang),
// and ctx.abort alone only covers session cancellation. An explicit controller chained to the
// caller's signal gives both, on every runtime.
function withDeadline(ms, outer) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(new Error("timed out after " + ms + "ms")), ms);
  let onAbort;
  if (outer) {
    if (outer.aborted) ctl.abort(outer.reason);
    else {
      onAbort = () => ctl.abort(outer.reason);
      outer.addEventListener("abort", onAbort, { once: true });
    }
  }
  return {
    signal: ctl.signal,
    release() {
      clearTimeout(timer);
      if (onAbort && outer) outer.removeEventListener("abort", onAbort);
    },
  };
}

// The endpoint streams its JSON-RPC reply as server-sent events. Mirrors the fork's parseSse:
// first `data:` frame carrying result.content[0].text wins.
function parseMcpBody(body) {
  for (const line of body.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    let data;
    try {
      data = JSON.parse(line.substring(6));
    } catch {
      continue;
    }
    const text = data && data.result && data.result.content && data.result.content[0]?.text;
    if (text) return text;
  }
  // Not in the fork: tolerate a plain application/json reply rather than returning nothing if the
  // endpoint ever stops streaming. Logged so it is never mistaken for the normal path.
  try {
    const data = JSON.parse(body);
    const text = data && data.result && data.result.content && data.result.content[0]?.text;
    if (text) {
      process.stderr.write(PLUGIN + " exa_mcp_non_sse=1\n");
      return text;
    }
  } catch {
    /* fall through */
  }
  return undefined;
}

// One JSON-RPC call to the endpoint. Both tools go through here: same URL, same envelope, only
// the tool name and arguments differ.
async function exaMcpCall(toolName, args, timeoutMs, signal, label) {
  const key = process.env.EXA_API_KEY;
  // Key goes in the query string, not a header -- that is the endpoint's contract (fork url()).
  // Base URL is overridable so Gym can put a caching proxy in front of Exa. That proxy speaks the
  // same wire format in both directions, so NOTHING else in this plugin changes -- all cache
  // logic (lookup, fuzzy matching, storage, accounting) lives Gym-side where it can be changed
  // without shipping a new file into the sandbox. Unset = straight to Exa, exactly as before.
  const base = (process.env.NEMO_GYM_EXA_BASE_URL || "https://mcp.exa.ai/mcp").replace(/\/+$/, "");
  const url = key ? base + "?exaApiKey=" + encodeURIComponent(key) : base;
  const payload = { jsonrpc: "2.0", id: 1, method: "tools/call", params: { name: toolName, arguments: args } };

  const started = Date.now();
  const d = withDeadline(timeoutMs, signal);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json, text/event-stream" },
      body: JSON.stringify(payload),
      signal: d.signal,
    });
    if (!res.ok) {
      throw new Error("exa mcp " + toolName + " failed: " + res.status + " " + (await res.text()).slice(0, 300));
    }
    const text = parseMcpBody(await res.text());
    // One greppable line per call, same purpose as the fork's [exa-rest]: proves from the run log
    // alone which transport actually ran, without re-reading trajectories.
    process.stderr.write(
      PLUGIN +
        " [exa-mcp] tool=" +
        toolName +
        " ms=" +
        (Date.now() - started) +
        " bytes=" +
        (text ? text.length : 0) +
        " " +
        label +
        "\n",
    );
    return text;
  } finally {
    d.release();
  }
}

function exaMcpSearch(query, type, numResults, livecrawl, contextMaxCharacters, signal) {
  const args = { query, type, numResults, livecrawl };
  // Optional in the fork's SearchArgs: omitted entirely when the model did not ask for one.
  if (contextMaxCharacters !== undefined && contextMaxCharacters !== null) {
    args.contextMaxCharacters = contextMaxCharacters;
  }
  return exaMcpCall("web_search_exa", args, MCP_TIMEOUT_MS, signal, "query=" + JSON.stringify(query.slice(0, 120)));
}

// `urls` is an array server-side even for one page; the model's tool still takes a single url, so
// the shape it sees is unchanged.
function exaMcpFetch(url, maxCharacters, timeoutMs, signal) {
  const args = { urls: [url] };
  if (maxCharacters) args.maxCharacters = maxCharacters;
  return exaMcpCall(
    "web_fetch_exa",
    args,
    timeoutMs || MCP_FETCH_TIMEOUT_MS,
    signal,
    "url=" + JSON.stringify(url.slice(0, 160)),
  );
}

async function exaRestSearch(query, numResults, maxCharacters, type, livecrawl, signal) {
  const key = process.env.EXA_API_KEY;
  if (!key) throw new Error("EXA_API_KEY is not set");
  const deep = REST_DEEP_TYPES.includes(type);
  // Request body matched field-for-field to the fork's REST builder (tool/mcp-exa.ts RestBody),
  // including the deep-only summary + outputSchema pair -- without them deep behaves like auto.
  const body = {
    query,
    numResults,
    type,
    contents: { text: { maxCharacters }, livecrawl, ...(deep ? { summary: true } : {}) },
    ...(deep ? { outputSchema: { type: "text" } } : {}),
  };
  const started = Date.now();
  const d = withDeadline(deep ? REST_DEEP_TIMEOUT_MS : REST_TIMEOUT_MS, signal);
  let data;
  try {
    const res = await fetch("https://api.exa.ai/search", {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json", "x-api-key": key },
      body: JSON.stringify(body),
      signal: d.signal,
    });
    if (!res.ok) throw new Error("exa search failed: " + res.status + " " + (await res.text()).slice(0, 300));
    data = await res.json();
  } finally {
    d.release();
  }
  const results = data.results || [];
  process.stderr.write(
    PLUGIN +
      " [exa-rest] type=" +
      type +
      " ms=" +
      (Date.now() - started) +
      " n=" +
      results.length +
      " deep_answer=" +
      (data.output && data.output.content ? 1 : 0) +
      " query=" +
      JSON.stringify(query.slice(0, 120)) +
      "\n",
  );
  // Field order and labels match the fork's renderer exactly, including the "N/A" published
  // placeholder, the capped [Deep Answer] block and the Summary:/Text: prefixes.
  const blocks = [];
  const synthesis = data.output && data.output.content;
  if (synthesis) {
    blocks.push(
      "[Deep Answer]: " +
        (synthesis.length > REST_DEEP_ANSWER_MAX_CHARS
          ? synthesis.slice(0, REST_DEEP_ANSWER_MAX_CHARS) + "\n[...deep answer truncated...]"
          : synthesis),
    );
  }
  for (const r of results) {
    let entry = "Title: " + (r.title || "") + "\nURL: " + (r.url || "") + "\nPublished: " + (r.publishedDate || "N/A");
    if (r.summary) entry += "\nSummary: " + r.summary;
    if (r.text) entry += "\nText: " + r.text;
    blocks.push(entry);
  }
  return blocks.length ? blocks.join("\n\n") : undefined;
}

// --- webfetch ---------------------------------------------------------------------------------
// Also Exa, via `web_fetch_exa` on the same MCP endpoint. Exa's own tool description says it:
// "Read a webpage's full content as clean markdown. Use after web_search_exa when highlights are
// insufficient."
//
// This replaces ~90 lines of hand-rolled downloading. Exa does the retrieval, so their
// infrastructure handles what the fork had to hand-build in webfetch.ts -- browser user-agent,
// Cloudflare challenges, per-format Accept headers, size limits, HTML-to-markdown conversion.
// Our own version had none of it and was failing roughly twice as many fetches as the fork.
//
// Divergence from the legacy comparator, stated plainly: legacy fetched DIRECTLY from inside the
// sandbox using the fork's webfetch.ts. Routing through Exa is a deliberate change, not a port.
//
// RATIFIED 2026-10-15 -- KEEP THIS, do not 'fix' it back to the builtin.
// The builtin issues a raw HTTP GET with a browser User-Agent and even retries Cloudflare
// challenges as UA=opencode, i.e. it deliberately works around sites that decline bots.
// Exa's fetch respects those exclusions. That guardrail is worth more than byte-parity with
// the apptainer arm, so this difference is ACCEPTED and must be stated whenever these numbers
// are compared to an apptainer run rather than quietly reconciled.
//
// KNOWN COST, measured over 60 rows: builtin returns p50 4,869 / p90 39,661 / max 51,455 chars
// per page; Exa's default returns p50 1,240 / p90 3,207 / max 3,834. ~13x less text, and our
// 50KB truncation-recovery path (full text to disk + Grep) therefore never fires.
// If accuracy comes in low, the first lever is maxCharacters (NEMO_GYM_EXA_MAX_CHARS), NOT
// reverting to the builtin. Running on Exa's default for now, by decision.
//
// Arguments are the MCP server's own, not ours: `urls` (we send one) and an optional
// `maxCharacters`, forwarded only when the model asks for one so Exa applies its own default.

// --- bash ---------------------------------------------------------------------------------------
// The third retrieval channel, and until now the only unguarded one.
//
// websearch and webfetch are filtered; bash was not. Measured over the completed 400-row run:
// of 230 bash commands, 165 (72%) fetch a URL -- curl/wget straight to a page, whose output
// reached the model with nothing checked. The legacy arm had the identical hole. It was
// survivable only because ubuntu:22.04 shipped no curl and 64 of those attempts died with
// "command not found"; swapping to python:3.12 turns it into a working, unfiltered route to any
// page on the internet, including the mirrors that republish BrowseComp questions WITH answers.
//
// WHY A SHADOW TOOL AND NOT tool.execute.after. For BUILTIN tools, truncation runs INSIDE the
// tool wrapper before any after-hook (stock tool.ts:131 calls truncate.output), and
// truncate.ts:69-71 writes the FULL UNFILTERED text to TRUNCATION_DIR and hands the model a path
// to Grep. An after-hook would therefore sanitise the returned string while leaving the real
// output on disk -- a guard that looks like it works and does not. A plugin tool is truncated
// AFTER our execute returns, so filtering here happens pre-truncation and there is no second copy.
//
// Execution goes through the plugin API's BunShell ($), which is why this is possible at all
// without touching the binary.
const BASH_DEFAULT_TIMEOUT_MS = 60_000;
const BASH_MAX_TIMEOUT_MS = 10 * 60 * 1_000; // stock MAX_TIMEOUT_MS
const BASH_MAX_CAPTURE = 1024 * 1024; // stock MAX_CAPTURE_BYTES

export default async function nemoGymWebGuard({ $ }) {
  process.stderr.write(
    PLUGIN +
      " web_guard version=2 contamination=" +
      (guardEnabled() ? "on" : "off") +
      " exclude_domains=" +
      excludedDomains().length +
      // Stated at load so a transport regression is visible in the log without reading a single
      // trajectory. mcp = the legacy comparator's path; rest = a deliberate deep-mode override.
      " search_transport=" +
      ((process.env.OPENCODE_WEBSEARCH_TYPE || "").trim() ? "rest" : "mcp") +
      "\n",
  );

  const bashGuard = guardEnabled() && process.env.NEMO_GYM_BASH_GUARD !== "0";
  process.stderr.write(PLUGIN + " bash_guard=" + (bashGuard ? "on" : "off") + "\n");

  return {
    tool: {
      websearch: {
        // Description and argument surface copied from the fork (tool/websearch.txt and
        // websearch.ts Parameters) so the model sees the same tool it saw on the legacy runs.
        // This matters: the model picks numResults itself, and an earlier description saying
        // "default 10" made it ask for 10 every time against the legacy 8.
        description: [
          "- Search the web using Exa AI - performs real-time web searches and can scrape content from specific URLs",
          "- Provides up-to-date information for current events and recent data",
          "- Supports configurable result counts and returns the content from the most relevant websites",
          "- Use this tool for accessing information beyond knowledge cutoff",
          "- Searches are performed automatically within a single API call",
          "",
          "Usage notes:",
          "  - Supports live crawling modes: 'fallback' (backup if cached unavailable) or 'preferred' (prioritize live crawling)",
          "  - Search types: 'auto' (balanced), 'fast' (quick results), 'deep' (comprehensive search)",
          "  - Configurable context length for optimal LLM integration",
          "  - Domain filtering and advanced search options available",
          "",
          "The current year is " + new Date().getFullYear() + ". You MUST use this year when searching for recent information or current events",
        ].join("\n"),
        // Plain JSON Schema, not Zod: a bare "zod" import cannot resolve from the /tmp path the
        // plugin is loaded from, and opencode's registry treats non-Zod arg values as JSON Schema.
        args: {
          query: { type: "string", description: "Websearch query" },
          numResults: { type: "integer", description: "Number of search results to return (default: 8)" },
          livecrawl: {
            type: "string",
            enum: ["fallback", "preferred"],
            description:
              "Live crawl mode - 'fallback': use live crawling as backup if cached content unavailable, 'preferred': prioritize live crawling (default: 'fallback')",
          },
          type: {
            type: "string",
            enum: ["auto", "fast", "deep"],
            description:
              "Search type - 'auto': balanced search (default), 'fast': quick results, 'deep': comprehensive search",
          },
          contextMaxCharacters: {
            type: "integer",
            description: "Maximum characters for context string optimized for LLMs (default: 10000)",
          },
        },
        async execute(args, ctx) {
          // Defaults mirror the fork's request builder, not its description text: numResults || 8
          // and livecrawl || "fallback".
          const n = args.numResults || Number(process.env.NEMO_GYM_EXA_NUM_RESULTS || 8);
          const livecrawl = args.livecrawl || process.env.NEMO_GYM_EXA_LIVECRAWL || "fallback";
          // The fork's routing switch, verbatim: a forced type means REST, nothing means MCP.
          const forcedType = (process.env.OPENCODE_WEBSEARCH_TYPE || "").trim();
          let raw;
          try {
            if (forcedType) {
              // REST only: `type` has no effect over MCP, so the deep modes live here alone.
              // contextMaxCharacters ?? 2500 keeps a SERP near the MCP path's observed size.
              const chars = args.contextMaxCharacters || Number(process.env.NEMO_GYM_EXA_MAX_CHARS || 2500);
              raw = await exaRestSearch(args.query, n, chars, forcedType, livecrawl, ctx.abort);
            } else {
              // Pass contextMaxCharacters through ONLY when the model supplied one -- the fork
              // leaves it undefined otherwise, and MCP drops it either way.
              raw = await exaMcpSearch(
                args.query,
                args.type || process.env.NEMO_GYM_EXA_SEARCH_TYPE || "auto",
                n,
                livecrawl,
                args.contextMaxCharacters,
                ctx.abort,
              );
            }
          } catch (e) {
            // Surfaced to the model as an ordinary tool failure rather than a crash.
            return "Search failed: " + (e && e.message ? e.message : String(e));
          }
          const filtered = raw ? filterSearchOutput(raw, "websearch") : undefined;
          return filtered && filtered.trim() ? filtered : EMPTY_RESULT;
        },
      },

      // Shadows the builtin only when the guard is on; with NEMO_GYM_BASH_GUARD=0 the key is
      // absent from this object and opencode keeps its own bash, so an A/B arm costs one env var.
      ...(bashGuard
        ? {
            bash: {
              // Argument surface copied from stock (core/src/tool/bash.ts Input): command,
              // workdir, timeout. Tool text is prompt text, so it stays identical to what the
              // model would otherwise see.
              description: [
                "- Executes a shell command and returns its combined stdout and stderr",
                "- Use for inspecting files, running scripts, and fetching resources the other tools cannot reach",
                "- The command runs inside the task sandbox",
                "",
                "Usage notes:",
                "  - Prefer the dedicated search and fetch tools for web content where they suffice",
                "  - Output is truncated if very large",
              ].join("\n"),
              args: {
                command: { type: "string", description: "Shell command string to execute" },
                workdir: {
                  type: "string",
                  description:
                    "Working directory. Defaults to the active Location; relative paths resolve from that Location.",
                },
                timeout: {
                  type: "integer",
                  description:
                    "Timeout in milliseconds. Defaults to " +
                    BASH_DEFAULT_TIMEOUT_MS +
                    " and may not exceed " +
                    BASH_MAX_TIMEOUT_MS +
                    ".",
                },
              },
              async execute(args, ctx) {
                const cmd = String(args.command || "");
                if (!cmd.trim()) return "";

                // Block the fetch BEFORE it happens when the command names an opted-out or
                // known-contaminated host. Same input-side treatment webfetch gets, and the only
                // way to honour a TDM opt-out through this channel: filtering the response after
                // the fact would still have retrieved the page.
                const urls = cmd.match(/https?:\/\/[^\s'"`)\]}>]+/g) || [];
                for (const u of urls) {
                  if (isExcludedUrl(u)) {
                    process.stderr.write(PLUGIN + " excludedomains fn=bash blocked=1\n");
                    return EMPTY_RESULT;
                  }
                  if (guardEnabled() && CONTAMINATED_URL_SUBSTRINGS.some((x) => u.toLowerCase().includes(x))) {
                    process.stderr.write(PLUGIN + " contamination fn=bash blocked_url=1\n");
                    return EMPTY_RESULT;
                  }
                }

                const timeout = Math.min(Number(args.timeout) || BASH_DEFAULT_TIMEOUT_MS, BASH_MAX_TIMEOUT_MS);
                let out = "";
                try {
                  // `.nothrow()` so a non-zero exit returns its output rather than raising: a
                  // failing command's stderr is exactly what the model needs to see.
                  let sh = $`sh -c ${cmd}`.nothrow().quiet();
                  if (args.workdir) sh = sh.cwd(String(args.workdir));
                  const res = await sh;
                  out = (res.stdout ? res.stdout.toString() : "") + (res.stderr ? res.stderr.toString() : "");
                } catch (e) {
                  return "Command failed: " + (e && e.message ? e.message : String(e));
                }
                if (out.length > BASH_MAX_CAPTURE) out = out.slice(0, BASH_MAX_CAPTURE);

                // Whole-output contamination check, as webfetch gets. Not per-record: bash output
                // has no record structure to split on, so a single marker blanks the result.
                return filterFetchOutput(out, "bash");
              },
            },
          }
        : {}),

      webfetch: {
        // Description and args are the FORK's (tool/webfetch.txt and webfetch.ts Parameters),
        // copied verbatim, even though Exa now does the retrieving. What the model reads is part
        // of the prompt, so it stays identical to the legacy arm; only the implementation beneath
        // changed. `format` defaults to markdown in the fork, which is exactly what Exa returns.
        description: [
          "- Fetches content from a specified URL",
          "- Takes a URL and optional format as input",
          "- Fetches the URL content, converts to requested format (markdown by default)",
          "- Returns the content in the specified format",
          "- Use this tool when you need to retrieve and analyze web content",
          "",
          "Usage notes:",
          "  - IMPORTANT: if another tool is present that offers better web fetching capabilities, is more targeted to the task, or has fewer restrictions, prefer using that tool instead of this one.",
          "  - The URL must be a fully-formed valid URL",
          "  - HTTP URLs will be automatically upgraded to HTTPS",
          '  - Format options: "markdown" (default), "text", or "html"',
          "  - This tool is read-only and does not modify any files",
          "  - Results may be summarized if the content is very large",
        ].join("\n"),
        args: {
          url: { type: "string", description: "The URL to fetch content from" },
          format: {
            type: "string",
            enum: ["text", "markdown", "html"],
            description: "The format to return the content in (text, markdown, or html). Defaults to markdown.",
          },
          timeout: { type: "integer", description: "Optional timeout in seconds (max 120)" },
        },
        async execute(args, ctx) {
          // Both checks run on the INPUT url, before anything is retrieved -- the fork does the
          // same for the exclusion (webfetch.ts:43-47), because a page we must not mine must not
          // be requested at all, not merely filtered afterwards. Returning the same neutral string
          // an empty page produces tells the model nothing about why.
          if (isExcludedUrl(args.url)) {
            process.stderr.write(PLUGIN + " excludedomains fn=webfetch blocked=1\n");
            return EMPTY_RESULT;
          }
          if (guardEnabled() && CONTAMINATED_URL_SUBSTRINGS.some((s) => args.url.toLowerCase().includes(s))) {
            process.stderr.write(PLUGIN + " contamination fn=webfetch blocked_url=1\n");
            return EMPTY_RESULT;
          }

          // The fork's timeout rule, kept because the arg is exposed: whatever the model asked
          // for, defaulting to 30s, hard-capped at 120s (webfetch.ts:59).
          const timeoutMs = Math.min((args.timeout ?? 30) * 1000, 120_000);
          let raw;
          try {
            // No maxCharacters: the fork never exposed one, so Exa applies its own default.
            raw = await exaMcpFetch(args.url, undefined, timeoutMs, ctx.abort);
          } catch (e) {
            // Surfaced to the model as an ordinary tool failure rather than a crash.
            return "Fetch failed: " + (e && e.message ? e.message : String(e));
          }
          if (!raw || !raw.trim()) return EMPTY_RESULT;
          // Contamination check on the BODY. The fork wrote filterFetchOutput and then never
          // called it (webfetch.ts:7 imports only isExcludedUrl/EMPTY_RESULT), so this is
          // deliberately stricter than the source.
          return filterFetchOutput(raw, "webfetch");
        },
      },
    },
  };
}
