# Release-note examples

Use these patterns as guidance, not fixed templates.

## Lead with the outcome

Context: A resources server can declare tasksets that `gym dataset collate` prepares for evaluation. The user benefit is avoiding a separate conversion script.

Avoid:

> Materialize declared tasksets on resources servers.

Prefer:

> Prepare environment datasets without separate conversion scripts by materializing declared tasksets.

## Split unrelated changes

Context: The Local provider runs trusted commands directly on the host without isolation. Pause and resume are separate lifecycle capabilities implemented by OpenSandbox in this release.

Avoid:

> Run workloads locally and pause or resume sandboxes.

This incorrectly connects host-process execution with a lifecycle capability implemented by another provider.

Prefer:

> Run and debug trusted workloads directly on the host with the Local provider.
>
> Pause or resume OpenSandbox sandboxes to preserve long-lived task state.

## Separate architecture and infrastructure from shipped support

Context: The Compose adapter uses provider capability interfaces, but OpenSandbox is the only built-in provider in this release that implements the required networking, storage, runtime, and forwarding capabilities.

Avoid:

> Run Compose workloads across sandbox providers.

Prefer:

> Run multi-service workloads on OpenSandbox from supported Compose YAML.

The provider-neutral design belongs in the detailed changelog; the release note describes the implementation users can run.

Context: Environment Servers provide infrastructure for defining custom interaction patterns. This release includes a built-in single-agent-turn implementation, but it does not include ready-made multi-turn or multi-agent implementations.

Avoid:

> NeMo Gym introduces reusable single-turn, multi-turn, and multi-agent interaction patterns.

Prefer:

> NeMo Gym introduces infrastructure for defining reusable agent-environment interaction patterns.

State the built-in support separately: this release includes a single-agent-turn implementation.

## Avoid false causality

Context: TLS configuration, terminal recovery, and cancellation cleanup are independent OpenSandbox changes that share a reliability outcome.

Avoid:

> Configure TLS verification and benefit from improved terminal handling and cancellation cleanup.

Prefer:

> OpenSandbox adds configurable TLS verification, more reliable terminal recovery, and cancellation cleanup.

## Use emphasis sparingly

Context: This is an ordinary feature bullet in a short section, not a dense catalog that needs visual category labels.

Avoid:

> **Local execution:** Run trusted workloads directly on your machine.

Prefer:

> Run trusted workloads directly on your machine with the Local provider.

Default to plain text. Reserve bold for contributor handles or category labels when it materially improves scanning in a dense list.

## Give migrations context

Context: This release routes episodes through Environment Servers. Existing agent-only configurations still run through an automatically generated compatibility server, but users need enough context to understand the warning and migration.

Avoid:

> Existing agent-only configurations use `legacy_agent`. Add an Environment Server.

Prefer:

> NeMo Gym now routes each episode through an Environment Server, which coordinates the agent, resources, verification, and cleanup. Existing agent-only configurations continue to work through an automatically generated `legacy_agent` compatibility server but emit a migration warning. To migrate, add an explicit Environment Server or run `scripts/add_legacy_agent_environment_servers.py`.

## Distinguish the motivating use case from the contract

Context: [PR #2737](https://github.com/NVIDIA-NeMo/Gym/pull/2737) was motivated and validated with Relay trajectories. However, NeMo Gym consumes a bounded, text-only and stateless ATIF v1.7 profile without requiring Relay as the producer or dependency.

Avoid:

> Re-verify Relay trajectories with `gym eval reverify --input-format atif`.

Prefer:

> Re-verify supported ATIF v1.7 trajectories from external systems.
