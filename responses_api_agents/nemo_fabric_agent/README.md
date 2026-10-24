# NeMo Fabric Agent

Run interchangeable agent harnesses through [NeMo Fabric](https://github.com/NVIDIA/NeMo-Fabric) inside NeMo Gym.

`configs/` has a preset per Python adapter: Claude, Codex, Deep Agents, Hermes, mini-SWE-agent, OpenClaw, and
OpenHands. OpenClaw also needs the `openclaw` npm CLI on `PATH`.

## Relay

Set `relay: true` to record each run with [NeMo Relay](https://github.com/NVIDIA/NeMo-Relay). The response is then
built from Relay's ATIF trajectory, so every adapter with Relay support returns the same item shape, and
`fabric_result.relay_atif` holds the trajectories. Adapters without Relay output fall back to their native output.
