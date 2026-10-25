# NeMo Fabric Agent

Run interchangeable agent harnesses through [NeMo Fabric](https://github.com/NVIDIA/NeMo-Fabric) inside NeMo Gym.

`configs/` has a preset per Python adapter: Claude, Codex, Deep Agents, Hermes, mini-SWE-agent, OpenClaw, and
OpenHands. OpenClaw also needs its CLI on `PATH`: `npm install --global openclaw@2026.9.4` (Node.js 24.16 or later).

## Relay

Set `relay: true` to record each run with [NeMo Relay](https://github.com/NVIDIA/NeMo-Relay). The response is then
built from Relay's ATIF trajectory, so every adapter with Relay support returns the same item shape, and
`fabric_result.relay_atif` holds the trajectories. Adapters without Relay support run without it. Relay is
experimental: in NeMo Fabric 0.5.0 several adapters record tool calls without their outputs, and mini-SWE-agent runs
fail more often with it. Claude and Codex record through a Relay gateway that accepts a plain-HTTP model URL only on
loopback, so give them an HTTPS or loopback model URL.
