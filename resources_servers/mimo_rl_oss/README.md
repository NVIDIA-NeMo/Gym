# mimo_rl_oss

Agentic tasks from [XiaomiMiMo/MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss), set up
and graded with mimoagent's dataset environments in a Gym sandbox.

| Subset | Rows | Reward |
|---|---|---|
| code | 2,698 | hidden test patch, test command exit code |
| cyber | 1,000 | ARVO PoC crashes in the described function |
| terminal_bench | 64 | `/tests/test.sh` writes `/logs/verifier/reward.txt` |
| webdev | 2,093 | MiMo's pointwise `webdev_eval_v1` vision judge |
| general_agent | 925 | MiMo's verifier over MCP tool state plus an LLM rubric judge |

Music (1,000 rows) is `resources_servers/mimo_music`.

general_agent runs MiMo's main and sidecar containers as one box, so an agent running as root can reach the MCP
state databases that the verifier grades. Treat its rewards with that in mind until the agent runs unprivileged.

`seed_session` builds the box and runs mimoagent's setup, harness_agent runs any harness in it, and `verify` grades
in the same box. Build the data with `python -m resources_servers.mimo_rl_oss.prepare`. Results are in `RESULTS.md`.
