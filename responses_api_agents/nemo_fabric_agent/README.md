# NeMo Fabric Agent

Run interchangeable agent harnesses through [NeMo Fabric](https://github.com/NVIDIA/NeMo-Fabric) inside NeMo Gym.

`configs/` has a preset per Python adapter: Claude, Codex, Deep Agents, Hermes, mini-SWE-agent, and OpenHands. The
OpenClaw adapter is installed but has no preset, because it also needs the OpenClaw CLI and Node.js, which this server
does not provision (`npm install --global openclaw@2026.9.34`, Node.js 24.22.1 or later).

## Relay

Set `relay: true` to record each run with [NeMo Relay](https://github.com/NVIDIA/NeMo-Relay). The response is then
built from Relay's ATIF trajectory, unless the adapter's own output has more tool outputs or reasoning (Relay's ATIF
records no reasoning), and `fabric_result.relay_atif` holds the trajectories, since the Relay files are removed
after the run. Adapters without Relay support, and Hermes without a Hermes Agent build that supports Relay, run
without it. Relay is experimental: in NeMo Fabric 0.6.1 several adapters record tool calls without their outputs,
and mini-SWE-agent runs fail more often with it. Claude and Codex record through a Relay gateway that accepts a
plain-HTTP model URL only on loopback, so give them an HTTPS or loopback model URL.
