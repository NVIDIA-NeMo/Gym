#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

gym_ci_sanitize_environment() {
    if [[ "$#" -ne 1 ]]; then
        echo "usage: gym_ci_sanitize_environment lint|core|server" >&2
        return 2
    fi

    case "$1" in
        lint)
            # CI lint always runs every configured hook.
            unset SKIP
            ;;
        core)
            # Keep test discovery, configuration, and imports rooted in this checkout.
            unset GYM_CI_DEV_VENV_DIR NEMO_GYM_EXTRA_ROOTS NEMO_GYM_CONFIG_DICT PYTHONPATH
            ;;
        server)
            # Keep discovery, installation mode, Python import behavior, and nested pytest
            # selection provider-neutral.
            unset GYM_CI_DEV_VENV_DIR NEMO_GYM_EXTRA_ROOTS NEMO_GYM_CONFIG_DICT
            unset NEMO_GYM_ALLOW_PRERELEASE
            unset PYTHONPATH PYTHONSAFEPATH PYTEST_ADDOPTS
            ;;
        *)
            echo "unknown Gym CI stage: $1" >&2
            return 2
            ;;
    esac
}
