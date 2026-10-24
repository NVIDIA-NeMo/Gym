# RNA Design (MCP tools)

The tools and scoring of the [RNA design environment](../../environments/rna_design/README.md), served over MCP
so an outside harness such as Claude Code can run the design campaign. Each rollout gets its own tool worker process
and sees only the tools its task offers, with the same budget as the agent. The scoring stays on the Gym node, so
code the harness writes cannot reach it. `verify` re-scores the pool with the suite's own metric and gates, the
same reward as the agent.

A rollout where the model was never reached, where the model endpoint failed partway (Claude Code's
`API Error: <status>`) and nothing was solved, or where every tool call failed and nothing was designed, is masked.

Setup and run instructions are in the [environment README](../../environments/rna_design/README.md). Licences are in
[LICENSE-third-party](../../responses_api_agents/rna_design_agent/LICENSE-third-party).

## Tests

```bash
gym env test --resources-server rna_design
```
