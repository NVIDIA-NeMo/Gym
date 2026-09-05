#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Node-local execution helpers for MARS shared-filesystem compliance.

[[ -n ${BASH_VERSION:-} ]] || { echo "MARS_STAGE_FAIL: bash is required" >&2; return 64 2>/dev/null || exit 64; }

MARS_PACKAGE_ID_EXPECTED=checkpoint-e2e-1.4.13-mars-local-r8-20260827
MARS_GYM_REVISION_EXPECTED=d3f146d386c7dfe07d4fabce32c4c8b14c7917d2

mars_fail() { echo "MARS_STAGE_FAIL: $*" >&2; return 64; }

mars_validate_token() {
    [[ $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]
}

mars_validate_source_dir() {
    local path=$1
    [[ $path == /lustre/* && $path != *:* && $path != *$'\n'* \
        && -d $path && ! -L $path ]]
}

mars_assert_node_local() {
    local path=$1 filesystem
    [[ $path == /raid/scratch/* && $path != *:* && $path != *$'\n'* \
        && -d $path && ! -L $path ]] || mars_fail "unsafe node-local directory: $path" || return
    filesystem=$(stat -f -c '%T' "$path") || mars_fail "could not inspect node-local filesystem: $path" || return
    [[ $filesystem != lustre && $filesystem != lustre_lite ]] \
        || mars_fail "node-local directory resolves to Lustre: $path"
}

mars_init() {
    local run_id=$1 role=$2 scope_material=${3:-$1} base scope
    mars_validate_token "$run_id" || mars_fail "unsafe run id: $run_id" || return
    mars_validate_token "$role" || mars_fail "unsafe role: $role" || return
    [[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ ]] || mars_fail "Slurm job id is required" || return
    MARS_USER=${SLURM_JOB_USER:-${USER:-}}
    [[ $MARS_USER =~ ^[A-Za-z0-9._-]+$ ]] || mars_fail "unsafe user name" || return
    [[ $scope_material == /lustre/* && $scope_material != *:* \
        && $scope_material != *$'\n'* ]] || mars_fail "unsafe campaign scope" || return
    scope=$(printf '%s' "$scope_material" | sha256sum | awk '{print substr($1,1,16)}')
    [[ $scope =~ ^[0-9a-f]{16}$ ]] || mars_fail "could not derive campaign scope" || return
    base=${CHECKPOINT_E2E_NODE_LOCAL_BASE:-/raid/scratch/$MARS_USER/gdpval-e2e}
    [[ $base == /raid/scratch/* && $base != *:* && $base != *$'\n'* && ! -L $base ]] \
        || mars_fail "node-local base must be below /raid/scratch: $base" || return
    install -d -m 0700 "$base" "$base/assets" "$base/locks" "$base/runs" "$base/jobs"
    mars_assert_node_local "$base" || return
    MARS_BASE=$base
    MARS_RUN_ROOT=$base/runs/${run_id}-${scope}
    MARS_JOB_ROOT=$base/jobs/${SLURM_JOB_ID}-${role}
    install -d -m 0700 "$MARS_RUN_ROOT" "$MARS_RUN_ROOT/config" \
        "$MARS_RUN_ROOT/component_venvs" "$MARS_JOB_ROOT" "$MARS_JOB_ROOT/tmp" \
        "$MARS_JOB_ROOT/cache/uv" "$MARS_JOB_ROOT/cache/xdg" \
        "$MARS_JOB_ROOT/cache/apptainer" "$MARS_JOB_ROOT/tmp/apptainer" \
        "$MARS_JOB_ROOT/logs" "$MARS_JOB_ROOT/preconvert"
    mars_assert_node_local "$MARS_RUN_ROOT" || return
    mars_assert_node_local "$MARS_JOB_ROOT" || return
    export MARS_USER MARS_BASE MARS_RUN_ROOT MARS_JOB_ROOT
    export TMPDIR=$MARS_JOB_ROOT/tmp
    # Ray appends a long session/socket suffix and Linux caps AF_UNIX paths at
    # 107 bytes. Keep this node-local prefix deliberately short.
    export RAY_TMPDIR=/raid/scratch/$MARS_USER/r/$SLURM_JOB_ID
    export UV_CACHE_DIR=$MARS_JOB_ROOT/cache/uv
    export XDG_CACHE_HOME=$MARS_JOB_ROOT/cache/xdg
    export APPTAINER_TMPDIR=$MARS_JOB_ROOT/tmp/apptainer
    export APPTAINER_CACHEDIR=$MARS_JOB_ROOT/cache/apptainer
    export PYTHONPYCACHEPREFIX=$MARS_JOB_ROOT/cache/pycache
    install -d -m 0700 "$RAY_TMPDIR" "$PYTHONPYCACHEPREFIX"
    (( ${#RAY_TMPDIR} <= 48 )) || mars_fail "Ray scratch prefix is too long" || return
}

mars_stage_package() {
    local source=$1 id marker lock temporary destination
    mars_validate_source_dir "$source" || mars_fail "unsafe package source: $source" || return
    [[ -f $source/MARS_PACKAGE_ID && ! -L $source/MARS_PACKAGE_ID \
        && $(<"$source/MARS_PACKAGE_ID") == "$MARS_PACKAGE_ID_EXPECTED" ]] \
        || mars_fail "package identity mismatch: $source" || return
    id=${MARS_PACKAGE_ID_EXPECTED//[^A-Za-z0-9._-]/_}
    destination=$MARS_BASE/assets/package-$id
    marker=$destination/.mars-ready
    lock=$MARS_BASE/locks/package-$id.lock
    exec {mars_package_lock_fd}>"$lock"
    flock -x "$mars_package_lock_fd" || mars_fail "could not lock package stage" || return
    if [[ ! -f $marker || -L $marker || $(<"$marker" 2>/dev/null) != "$MARS_PACKAGE_ID_EXPECTED" ]]; then
        temporary=$MARS_BASE/assets/.package-$id.${SLURM_JOB_ID}.$$
        rm -rf -- "$temporary"
        install -d -m 0700 "$temporary"
        cp -a -- "$source/." "$temporary/"
        chmod u+w "$temporary"
        [[ -f $temporary/MARS_PACKAGE_ID \
            && $(<"$temporary/MARS_PACKAGE_ID") == "$MARS_PACKAGE_ID_EXPECTED" ]] \
            || mars_fail "staged package identity mismatch" || return
        printf '%s\n' "$MARS_PACKAGE_ID_EXPECTED" > "$temporary/.mars-ready"
        chmod 0400 "$temporary/.mars-ready"
        chmod -R a-w "$temporary"
        rm -rf -- "$destination"
        mv "$temporary" "$destination"
    fi
    flock -u "$mars_package_lock_fd"
    exec {mars_package_lock_fd}>&-
    MARS_PACKAGE=$destination
    export MARS_PACKAGE
}

mars_stage_gym() {
    local source=$1 revision=$2 destination marker lock temporary item venv_source metadata
    mars_validate_source_dir "$source" || mars_fail "unsafe Gym source: $source" || return
    [[ $revision == "$MARS_GYM_REVISION_EXPECTED" ]] \
        || mars_fail "unexpected Gym revision: $revision" || return
    [[ $(git -C "$source" rev-parse HEAD) == "$revision" ]] \
        || mars_fail "Gym source revision mismatch" || return
    destination=$MARS_BASE/assets/gym-${revision:0:12}
    marker=$destination/.mars-ready
    lock=$MARS_BASE/locks/gym-${revision:0:12}.lock
    exec {mars_gym_lock_fd}>"$lock"
    flock -x "$mars_gym_lock_fd" || mars_fail "could not lock Gym stage" || return
    if [[ ! -f $marker || -L $marker || $(<"$marker" 2>/dev/null) != "$revision" ]]; then
        temporary=$MARS_BASE/assets/.gym-${revision:0:12}.${SLURM_JOB_ID}.$$
        rm -rf -- "$temporary"
        install -d -m 0700 "$temporary"
        for item in pyproject.toml README.md LICENSE nemo_gym; do
            [[ -e $source/$item && ! -L $source/$item ]] \
                || mars_fail "Gym staging input is missing: $source/$item" || return
            cp -a -- "$source/$item" "$temporary/$item"
        done
        venv_source=$(readlink -f -- "$source/.venv")
        [[ $venv_source == /lustre/* && -d $venv_source && ! -L $venv_source ]] \
            || mars_fail "Gym virtualenv target is invalid: $source/.venv" || return
        cp -a -- "$venv_source" "$temporary/.venv"
        # The source virtualenv is editable and embeds its original Lustre
        # checkout. Local PYTHONPATH supplies the staged packages, so disable
        # that fallback rather than ever importing code from the source tree.
        for metadata in "$temporary"/.venv/lib/python*/site-packages/__editable__.nemo_gym*; do
            [[ -e $metadata || -L $metadata ]] || continue
            rm -f -- "$metadata"
        done
        for item in benchmarks resources_servers responses_api_agents responses_api_models; do
            install -d -m 0700 "$temporary/$item"
            [[ ! -f $source/$item/__init__.py ]] \
                || cp -a -- "$source/$item/__init__.py" "$temporary/$item/__init__.py"
        done
        cp -a -- "$source/benchmarks/gdpval" "$temporary/benchmarks/gdpval"
        cp -a -- "$source/resources_servers/gdpval" "$temporary/resources_servers/gdpval"
        cp -a -- "$source/responses_api_agents/stirrup_agent" \
            "$temporary/responses_api_agents/stirrup_agent"
        cp -a -- "$source/responses_api_models/openai_model" \
            "$temporary/responses_api_models/openai_model"
        cp -a -- "$source/responses_api_models/vllm_model" \
            "$temporary/responses_api_models/vllm_model"
        printf '%s\n' "$revision" > "$temporary/.checkpoint_e2e_revision"
        printf '%s\n' "$revision" > "$temporary/.mars-ready"
        chmod 0400 "$temporary/.checkpoint_e2e_revision" "$temporary/.mars-ready"
        [[ -x $temporary/.venv/bin/python ]] \
            || mars_fail "staged Gym Python is unavailable" || return
        rm -rf -- "$destination"
        mv "$temporary" "$destination"
    fi
    flock -u "$mars_gym_lock_fd"
    exec {mars_gym_lock_fd}>&-
    MARS_GYM=$destination
    MARS_PYTHON=$destination/.venv/bin/python
    [[ -x $MARS_PYTHON ]] || mars_fail "staged Gym Python is unavailable" || return
    export MARS_GYM MARS_PYTHON
}

mars_stage_runtime() {
    local destination marker lock runtime_id
    [[ -n ${MARS_PACKAGE:-} && -n ${MARS_GYM:-} && -n ${MARS_PYTHON:-} ]] \
        || mars_fail "package and Gym must be staged before runtime" || return
    runtime_id=${MARS_PACKAGE_ID_EXPECTED//[^A-Za-z0-9._-]/_}
    destination=$MARS_BASE/assets/runtime-${MARS_GYM_REVISION_EXPECTED:0:12}-${runtime_id}
    marker=$destination.mars-ready
    lock=$MARS_BASE/locks/runtime-${MARS_GYM_REVISION_EXPECTED:0:12}-${runtime_id}.lock
    exec {mars_runtime_lock_fd}>"$lock"
    flock -x "$mars_runtime_lock_fd" || mars_fail "could not lock runtime stage" || return
    if [[ ! -f $marker || -L $marker \
        || $(<"$marker" 2>/dev/null) != "$MARS_PACKAGE_ID_EXPECTED" ]]; then
        rm -rf -- "$destination"
        PYTHONPATH="$MARS_GYM" "$MARS_PYTHON" "$MARS_PACKAGE/transport_runtime.py" materialize \
            --gym-root "$MARS_GYM" --runtime-root "$destination" --package-root "$MARS_PACKAGE"
        printf '%s\n' "$MARS_PACKAGE_ID_EXPECTED" > "$marker"
        chmod 0400 "$marker"
    fi
    PYTHONPATH="$destination:$MARS_GYM" "$MARS_PYTHON" \
        "$MARS_PACKAGE/transport_runtime.py" validate \
        --gym-root "$MARS_GYM" --runtime-root "$destination" --package-root "$MARS_PACKAGE" \
        >/dev/null
    flock -u "$mars_runtime_lock_fd"
    exec {mars_runtime_lock_fd}>&-
    MARS_RUNTIME=$destination
    export MARS_RUNTIME
    export NEMO_GYM_EXTRA_ROOTS=$MARS_RUNTIME
    export PYTHONPATH=$MARS_RUNTIME:$MARS_GYM${PYTHONPATH:+:$PYTHONPATH}
}

mars_stage_file() {
    local source=$1 destination=$2 temporary
    [[ $source == /lustre/* && $source != *:* && $source != *$'\n'* \
        && -f $source && ! -L $source ]] || mars_fail "unsafe staged file source: $source" || return
    [[ ( $destination == "$MARS_RUN_ROOT"/* || $destination == "$MARS_JOB_ROOT"/* ) \
        && $destination != *:* && $destination != *$'\n'* ]] \
        || mars_fail "unsafe staged file destination: $destination" || return
    install -d -m 0700 "$(dirname -- "$destination")"
    temporary=$destination.tmp.${SLURM_JOB_ID}.$$
    cp -- "$source" "$temporary"
    chmod 0400 "$temporary"
    mv -f "$temporary" "$destination"
}

mars_stage_uv() {
    local source=${1:-/home/$MARS_USER/.local/bin/uv} destination
    [[ $source == /* && $source != /lustre/* && $source != *:* \
        && $source != *$'\n'* && -f $source && ! -L $source && -x $source ]] \
        || mars_fail "unsafe uv staging source: $source" || return
    destination=$MARS_JOB_ROOT/bin
    install -d -m 0700 "$destination"
    cp -- "$source" "$destination/uv"
    chmod 0500 "$destination/uv"
    [[ $(stat -f -c '%T' "$destination/uv") != lustre \
        && $(stat -f -c '%T' "$destination/uv") != nfs ]] \
        || mars_fail "staged uv is not node-local" || return
    MARS_UV_DIR=$destination
    MARS_UV=$destination/uv
    export MARS_UV_DIR MARS_UV
}

mars_stage_container() {
    local source=$1 signature=$2 destination marker lock temporary actual
    [[ $source == /lustre/* && $source != *:* && $source != *$'\n'* \
        && -f $source && ! -L $source && $signature =~ ^[1-9][0-9]*:[1-9][0-9]*$ ]] \
        || mars_fail "unsafe container staging contract" || return
    destination=$MARS_BASE/assets/gdpval-${signature/:/-}.sif
    marker=$destination.mars-ready
    lock=$MARS_BASE/locks/gdpval-${signature/:/-}.lock
    exec {mars_container_lock_fd}>"$lock"
    flock -x "$mars_container_lock_fd" || mars_fail "could not lock container stage" || return
    if [[ ! -f $marker || -L $marker || ! -f $destination || -L $destination ]]; then
        temporary=$destination.tmp.${SLURM_JOB_ID}.$$
        cp -p --reflink=auto -- "$source" "$temporary"
        actual=$(stat -c '%s:%Y' "$temporary")
        [[ $actual == "${signature%:*}:$(( ${signature#*:} / 1000000000 ))" ]] \
            || { rm -f -- "$temporary"; mars_fail "staged container signature mismatch"; return; }
        chmod 0500 "$temporary"
        mv -f "$temporary" "$destination"
        printf '%s\n' "$signature" > "$marker"
        chmod 0400 "$marker"
    fi
    flock -u "$mars_container_lock_fd"
    exec {mars_container_lock_fd}>&-
    MARS_GDPVAL_SIF=$destination
    export MARS_GDPVAL_SIF
}

mars_stage_apptainer() {
    local source=$1 signature=$2 source_root destination marker lock temporary
    [[ $source == /lustre/* && $source != *:* && $source != *$'\n'* \
        && -d $source && ! -L $source && $signature =~ ^[1-9][0-9]*:[1-9][0-9]*$ ]] \
        || mars_fail "unsafe Apptainer staging contract" || return
    [[ ${source##*/} == bin && -x $source/apptainer ]] \
        || mars_fail "Apptainer source must be its bin directory" || return
    source_root=${source%/bin}
    [[ $source_root == /lustre/* && -d $source_root && ! -L $source_root ]] \
        || mars_fail "unsafe Apptainer installation root" || return
    destination=$MARS_BASE/assets/apptainer-${signature/:/-}
    marker=$destination/.mars-ready
    lock=$MARS_BASE/locks/apptainer-${signature/:/-}.lock
    exec {mars_apptainer_lock_fd}>"$lock"
    flock -x "$mars_apptainer_lock_fd" || mars_fail "could not lock Apptainer stage" || return
    if [[ ! -f $marker || -L $marker ]]; then
        temporary=$MARS_BASE/assets/.apptainer-${signature/:/-}.${SLURM_JOB_ID}.$$
        rm -rf -- "$temporary"
        install -d -m 0700 "$temporary"
        cp -a -- "$source_root/." "$temporary/"
        [[ -x $temporary/bin/apptainer ]] || mars_fail "staged Apptainer binary is unavailable" || return
        printf '%s\n' "$signature" > "$temporary/.mars-ready"
        chmod 0400 "$temporary/.mars-ready"
        rm -rf -- "$destination"
        mv "$temporary" "$destination"
    fi
    flock -u "$mars_apptainer_lock_fd"
    exec {mars_apptainer_lock_fd}>&-
    MARS_APPTAINER_BIN=$destination/bin
    export MARS_APPTAINER_BIN
}

mars_stage_bootstrap_helper() {
    local source=$1 destination bootstrap_user
    mars_validate_source_dir "$source" || { echo "MARS_STAGE_FAIL: unsafe bootstrap package: $source" >&2; return 64; }
    bootstrap_user=${SLURM_JOB_USER:-${USER:-}}
    [[ $bootstrap_user =~ ^[A-Za-z0-9._-]+$ ]] \
        || { echo "MARS_STAGE_FAIL: unsafe bootstrap user" >&2; return 64; }
    destination=/raid/scratch/${bootstrap_user}/gdpval-e2e-bootstrap-${SLURM_JOB_ID}
    install -d -m 0700 "$destination"
    cp -- "$source/mars_node_local.sh" "$destination/mars_node_local.sh"
    chmod 0500 "$destination/mars_node_local.sh"
    printf '%s\n' "$destination/mars_node_local.sh"
}
