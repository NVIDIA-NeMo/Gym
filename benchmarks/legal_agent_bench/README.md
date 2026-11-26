# Legal Agent Bench

Run Harvey LAB through Gym's Harbor Environment Server with LAB Native, Hermes,
Claude Code, or Codex. See the [LAB guide](../../fern/versions/latest/pages/evaluation/legal-agent-bench.mdx)
for preparation, judge configuration, harness and sandbox selection, and scoring.

The benchmark's agent is `legal_agent_bench_benchmark_agent`.
Use `lab.reward_mode=criteria_pass_rate` for diagnostic criterion scoring;
`full_task` is the default.

Run the LAB preparation, configuration, sampling, and verifier tests with:

```bash
gym env test --resources-server legal_agent_bench
```
