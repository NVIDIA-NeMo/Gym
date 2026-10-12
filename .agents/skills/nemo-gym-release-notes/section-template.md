# Release-note section template

Use stable, workflow-based section names so readers can compare releases quickly. Omit an empty section instead of renaming it. Add a new section only when multiple important changes do not fit an existing category.

## Historical pattern

- v0.3 through v0.7 consistently use Release Summary, First-Time Contributors, agent harnesses, models, environments or benchmarks, deprecations, bug fixes, and documentation.
- Rollout sections evolved from “Rollout Collection & Profiling” to “Rollout Observability and Export.” “Inspect and Export Agent Runs” describes the workflow without requiring readers to know rollout terminology.
- Sandboxing has appeared since v0.4 under changing names. “Configure Sandboxes” aligns with the agent and model sections while covering providers, isolation, lifecycle operations, and multi-service workloads.
- “Evaluation and Training” first appeared in v0.6 and remains the appropriate home for evaluation execution, scoring, scheduling, and training-data production.
- Command Line Interface and Release Assets appeared in some older releases but are not stable product workflows. Prefer placing CLI changes under the workflow they enable, and include Release Assets only when users need release-specific artifacts.
- v0.6 introduced “Configure Tasks and Data.” “Build and Configure Environments” is a broader successor that also accommodates Environment Servers, validation, and server runtime configuration.

## Canonical order

### Release Summary

Write this after all detailed sections are approved. Open with exactly one standalone sentence that communicates the release's most important user outcome to someone who reads nothing else. Do not turn the sentence into a compressed list of the Highlights. Use parallel clauses for independent benefits; do not imply one enables another unless the evidence supports that relationship. Follow it with three to six outcome-focused Highlights.

### First-Time Contributors

List verified first-time contributors and their most important contributions. Include the total contributor count and exclude bots.

### Build and Configure Environments

Cover environment lifecycle, tasks and datasets, configuration, validation, manifests, and server runtime requirements.

### Configure Agent Harnesses

Cover new harnesses, harness interoperability, and meaningful harness capabilities. Put reliability-only corrections under Bug Fixes.

### Configure Models

Cover model integrations, serving, routing, scaling, and model-request behavior. Put dependency migrations under Deprecation and Compatibility Notices.

### Configure Sandboxes

Cover sandbox providers, isolation boundaries, lifecycle operations, multi-service workloads, and provider reliability.

### Evaluation and Training

Cover evaluation collection, scoring, scheduling, resumption, resource control, and production or preservation of training data.

### Inspect and Export Agent Runs

Cover trajectory integrity, health checks, diagnostics, tracing, observability, import, export, and reverification.

### Environments and Benchmarks

Group new or substantially expanded environments by user-facing domain. Briefly explain unfamiliar entries instead of listing names alone.

### Deprecation and Compatibility Notices

Explain what changed, who is affected, whether compatibility remains, and the required migration. Include dependency requirements here.

### Bug Fixes

Include only fixes with meaningful user impact, ordered by security, correctness, data preservation, operational reliability, and narrower benchmark fixes.

### Documentation

List substantial documentation additions that are useful to call out but are not product capabilities.

### Release Assets

Optional. Include only when the release publishes artifacts users must find or use.

## Rules for adapting the template

- Keep the canonical order among sections that are present.
- Omit sections with no meaningful content.
- Do not rename a section merely to match the vocabulary of one release.
- Use imperative verb phrases for setup and configuration workflows; use noun phrases for outcomes, catalogs, and reference information.
- Add a new section only when at least two important items share a user workflow not represented above.
- Place each change by user workflow, not by source directory, PR label, or CLI command.
