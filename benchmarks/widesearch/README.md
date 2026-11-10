# WideSearch

[WideSearch](https://github.com/ByteDance-Seed/WideSearch) evaluates broad web
research: collecting many facts and returning them in a structured Markdown
table. The public dataset contains 200 tasks in English and Chinese.

This integration runs a Gym harness in a sandbox with Exa MCP search. The
[`widesearch`](../../resources_servers/widesearch/README.md) resources server
parses and grades the resulting table.

## Prepare the data

Preparation downloads the task metadata and per-task gold CSV files from
[Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/WideSearch), then
writes `data/widesearch_benchmark.jsonl`.

```bash
gym eval prepare --benchmark widesearch
```

Five prepared tasks and their diagnostic rollouts are checked in under the
[`widesearch` resources server](../../resources_servers/widesearch/data/).

## Run

The Claude Code config defaults to `nvidia/qwen/qwen3.8-27b`. Use `--limit 1` for
an end-to-end check before running all 200 tasks.

Complete long-run config copies for all supported harnesses are in
[`configs`](configs). Each file includes the WideSearch verifier, agent
settings, sandbox settings, prompt, and dataset. The files do not inherit from
a shared agent config. There are two kinds:

- **Session harnesses** (Codex, Pi, Hermes, OpenCode, OpenClaw, Kilo Code) run
  under `single_agent_turn_legacy`. Each episode, the agent creates its own
  sandbox and gets the `web_search`, `find_in_page` and `scroll_page` tools over
  MCP from the `exa_search` tool server (`tool_servers`): a
  [`tavily_search`](../../resources_servers/tavily_search/README.md) server with
  `search_provider: exa`. Its exclusion policy,
  [`search_exclusions.json`](search_exclusions.json), keeps the dataset's
  Hugging Face pages out of the results and page reads of these three tools; it
  does not restrict the sandbox's own network access, so a shell or a harness's
  own fetch tool can still reach them. The sandbox calls the model
  server and `exa_search` on the Gym host, so these configs set
  `use_absolute_ip: true`: the sandbox must be able to reach that host. These
  copies use different search tools, result formats and exclusions than their
  earlier `harness_exa_search` versions, so their scores are not directly
  comparable with runs of those versions.
- **Claude Code, Cline, Prime Agent, Simple Strands, and NeMo Fabric
  DeepAgents** still run through the deprecated `harness_exa_search` agent
  behind `legacy_agent`, which starts the harness and its Exa integration in a
  sandbox and relays model calls itself. Claude Code takes no MCP tool grants
  in a session, and the other four have no session path yet. Their `agent`
  field selects a name from Gym's shared `harness_agent` registry, and
  `agent_kwargs` contains that agent's settings.

The session harnesses read `EXA_API_KEY`; export it, and pass a sandbox
provider config next to the config copy. The `harness_exa_search` copies set
`exa_api_key: null`, so set it in the copy to give those harnesses Exa.

The Claude Code configs install Gym and the Claude CLI in a fresh sandbox
using `setup_command`. No cluster runtime archive is required.

For example, select the OpenCode copy directly:

```bash
gym eval run \
  --benchmark widesearch/configs/widesearch_opencode \
  --model-type <model-type> \
  --model <model> \
  --limit 1
```

Set `exa_api_key` in the selected config before running it. The available
copies are Claude Code, Codex, OpenCode, Pi, Hermes, OpenClaw, Kilocode, Cline,
Prime Agent, Simple Strands, and NeMo Fabric DeepAgents.

## Verification

The verifier follows the upstream evaluation specification for required and
unique columns, preprocessing, entity alignment, exact, URL, numeric, date,
and LLM-judged comparisons. It reports row-level and item-level
precision/recall/F1. Reward is the official strict score: `1` only when both
the complete table and every item match.
