# mimoagent

Runs any [mimoagent](https://github.com/XiaomiMiMo/mimoagent) harness as a Gym agent. `profile` picks a file in
`profiles/` (the `agent` and `model` blocks of mimoagent's `example_configs`, MIT).

| Kind | Profiles |
|---|---|
| native loops | `default`, `bashonly-agent`, `cc-agent`, `codex-agent`, `mimocode-agent` |
| CLIs installed in the box | `claude-code`, `codex`, `mimocode`, `pi`, `grok`, `kimi-code`, `kimi-cli`, `kilocode`, `openclaw`, `opencode`, `omp`, `hermes`, `dsh`, `mini-swe-agent` |

Run it inside the task box with `harness_agent` (`agent: mimoagent`, `sandbox_python: /opt/mimo-rt/venv/bin/python`,
setup command `install_runtime.sh`). See `resources_servers/mimo_rl_oss/configs/mimo_rl_oss.yaml`.

Profile changes from upstream:

- `codex-agent` uses `ptc: false` because OpenAI-compatible gateways drop its freeform code-mode tools.
- `mimocode` is pinned to 0.1.15 because 0.1.12 is no longer published.
- `dsh` raises Node's heap for its npm install.

The harness sends the profile's model name unless `model` is set. Gym's model server substitutes its own model, and
dsh and grok reject names outside their own catalogs.
