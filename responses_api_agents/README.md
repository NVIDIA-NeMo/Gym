# Compatibility namespace

Built-in agent harnesses moved to [`harnesses/`](../harnesses). This directory only preserves
historical Python imports such as `responses_api_agents.simple_agent`; new code should import
`harnesses.simple_agent`.

The YAML server-type key remains `responses_api_agents` for compatibility with the existing
configuration and runtime protocol.
