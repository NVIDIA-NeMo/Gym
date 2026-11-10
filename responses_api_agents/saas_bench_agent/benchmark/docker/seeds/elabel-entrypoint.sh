#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Source: https://gitlab-master.nvidia.com/pritamb/saas-bench/-/blob/31177a7a80040edf588388dd982af4b678d1e544/docker/seeds/elabel-entrypoint.sh
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
# SQL Server aborts with "Invalid mapping of address ... reserved address space"
# when the soft stack limit is unlimited (legacy VA layout). Cap it, then hand
# off to the image's all-in-one entrypoint.
ulimit -s 8192 || true
exec /entrypoint-aio.sh
