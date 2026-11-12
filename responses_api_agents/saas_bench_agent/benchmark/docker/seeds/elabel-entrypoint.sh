#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/docker/seeds/elabel-entrypoint.sh
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
# SQL Server aborts with "Invalid mapping of address ... reserved address space"
# when the soft stack limit is unlimited (legacy VA layout). Cap it, then hand
# off to the image's all-in-one entrypoint.
ulimit -s 8192 || true
exec /entrypoint-aio.sh
