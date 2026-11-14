# WildClawBench Environment Server

Runs prepared flat task rows through the standard single-agent-turn session lifecycle, with
Resources-owned task state and native verification. Agent close precedes verification, and
Resources close removes the native container. Supports one episode at a time per deployment.

Adds persistent episode identity, identical-request sharing, saved replies and native suite
aggregation. Failed, masked or incomplete suites cannot publish a headline score. All three
components must share one trusted host and private workspace. The native runner, task rubrics
and image remain unchanged. See [benchmark documentation](../../benchmarks/wildclawbench/README.md).
