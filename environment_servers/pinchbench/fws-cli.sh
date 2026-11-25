#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Limit the mock proxy and credentials to the task's CLI process. Harness installation and
# model requests in the same container still use their own endpoints.
set -eu
exports=$(fws server env)
eval "$exports"
case "${0##*/}" in
    gws) exec /usr/local/lib/node_modules/@googleworkspace/cli/run.js "$@" ;;
    gh) exec /usr/bin/gh "$@" ;;
esac
