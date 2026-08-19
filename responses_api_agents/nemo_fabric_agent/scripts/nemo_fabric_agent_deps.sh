#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_VERSION="${PYTHON_VERSION:-3.13.14}"
export PBS_RELEASE="${PBS_RELEASE:-20260610}"
source "${PORTABLE_PYTHON_SH:-$SCRIPT_DIR/_portable_python.sh}"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

install_portable_python
install_nemo_gym_deps

FABRIC_SPEC="${NEMO_FABRIC_SPEC:-nemo-fabric[mini-swe-agent]==0.2.0rc3}"
FABRIC_RUNTIME_SPEC="${NEMO_FABRIC_RUNTIME_SPEC:-nemo-fabric-runtime==0.2.0rc3}"
"$DEPS_DIR/bin/python3" -m pip install \
    "$FABRIC_SPEC" \
    "$FABRIC_RUNTIME_SPEC" \
    "nemo-fabric-adapter-contract==0.2.0rc3" \
    "nemo-fabric-adapters-common==0.2.0rc3" \
    "nemo-fabric-adapters-mini-swe-agent==0.2.0rc3"
"$DEPS_DIR/bin/python3" -m pip install "openai==${OPENAI_VERSION:-2.7.2}"
"$DEPS_DIR/bin/python3" -c "from nemo_fabric import Fabric, FabricConfig; print('nemo-fabric OK')"
