# Exa search tool server

Serves one tool, `exa_search`, that searches the web with [Exa](https://exa.ai) and returns each result's title, URL,
publication date, author and page text (up to `max_characters` per result). It keeps no session state and grades
nothing.

List it in an Environment Server's `tool_servers` (or the agent's) to grant the tool to a session agent over MCP:

```yaml
my_environment_server:
  environment_servers:
    single_agent_turn_legacy:
      tool_servers:
        - type: resources_servers
          name: exa_search
```

Each episode then seeds its own session on this server, grants `exa_search` as a required MCP tool, and closes the
session at cleanup. The harness runs in a sandbox and calls the tool there, so the sandbox must reach this server: set
`use_absolute_ip: true`. `benchmarks/deepsearchqa` and `benchmarks/widesearch` use it for their session harnesses.

| Config | Default | Meaning |
|---|---|---|
| `exa_api_key` | `${oc.env:EXA_API_KEY}` | Exa API key, sent as `x-api-key`. |
| `exa_base_url` | `https://api.exa.ai` | Exa API root; the tool posts to `<root>/search`. |
| `num_results` | `5` | Results per search when the call names no number (calls may ask for 1 to 10). |
| `max_characters` | `4000` | Page text kept per result. |
| `max_attempts` | `3` | Attempts per search on the statuses Gym retries (`RETRY_ERROR_CODES`, such as 429 and 503) and on transport failures. |
| `timeout_s` | `60` | Timeout of one Exa request. |

A failed search returns the HTTP status or the transport error to the agent as the tool's result, never Exa's error
body, so the agent can retry or answer without it.
