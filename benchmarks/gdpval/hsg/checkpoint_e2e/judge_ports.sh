#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Select bounded Gym service-port slots outside the node's ephemeral range.

gdpval_select_judge_port_window() {
    local range_file=${1:-/proc/sys/net/ipv4/ip_local_port_range}
    local preferred_low=2000 preferred_high=5999 slot_width=20
    local ephemeral_low ephemeral_high extra
    local lower_start lower_end lower_slots=0
    local upper_start upper_end upper_slots=0

    GDPVAL_JUDGE_PORT_BASE=
    GDPVAL_JUDGE_PORT_SLOT_WIDTH=
    GDPVAL_JUDGE_PORT_SLOT_COUNT=
    GDPVAL_JUDGE_PORT_WINDOW_HIGH=

    [[ -f $range_file && ! -L $range_file && -r $range_file ]] || {
        echo "GDPVAL_JUDGE_PORT_FAIL: ephemeral port range is unreadable: $range_file" >&2
        return 64
    }
    IFS=$' \t' read -r ephemeral_low ephemeral_high extra < "$range_file" || {
        echo "GDPVAL_JUDGE_PORT_FAIL: could not read ephemeral port range: $range_file" >&2
        return 64
    }
    [[ $ephemeral_low =~ ^[0-9]+$ && $ephemeral_high =~ ^[0-9]+$ \
        && -z $extra && $ephemeral_low -ge 1 && $ephemeral_high -le 65535 \
        && $ephemeral_low -le $ephemeral_high ]] || {
        echo "GDPVAL_JUDGE_PORT_FAIL: malformed ephemeral port range: $range_file" >&2
        return 64
    }

    # Prefer the fixed 2000-5999 window: it also excludes Ray's conventional
    # 6379 GCS and 8265 dashboard ports. If a platform's ephemeral range cuts
    # through it, use whichever disjoint side contains more complete slots.
    lower_start=$preferred_low
    lower_end=$((ephemeral_low - 1))
    if (( lower_end > preferred_high )); then
        lower_end=$preferred_high
    fi
    if (( lower_end >= lower_start )); then
        lower_slots=$(((lower_end - lower_start + 1) / slot_width))
    fi

    upper_start=$((ephemeral_high + 1))
    if (( upper_start < preferred_low )); then
        upper_start=$preferred_low
    fi
    upper_end=$preferred_high
    if (( upper_end >= upper_start )); then
        upper_slots=$(((upper_end - upper_start + 1) / slot_width))
    fi

    if (( lower_slots == 0 && upper_slots == 0 )); then
        echo "GDPVAL_JUDGE_PORT_FAIL: no complete non-ephemeral 20-port slot exists in 2000-5999" >&2
        return 64
    fi
    if (( upper_slots > lower_slots )); then
        GDPVAL_JUDGE_PORT_BASE=$upper_start
        GDPVAL_JUDGE_PORT_SLOT_COUNT=$upper_slots
    else
        GDPVAL_JUDGE_PORT_BASE=$lower_start
        GDPVAL_JUDGE_PORT_SLOT_COUNT=$lower_slots
    fi
    GDPVAL_JUDGE_PORT_SLOT_WIDTH=$slot_width
    GDPVAL_JUDGE_PORT_WINDOW_HIGH=$((
        GDPVAL_JUDGE_PORT_BASE + GDPVAL_JUDGE_PORT_SLOT_COUNT * slot_width - 1
    ))

    # Keep the invariant beside the calculation: every port later handed to
    # Gym must be outside the kernel allocator's ephemeral interval.
    (( GDPVAL_JUDGE_PORT_WINDOW_HIGH < ephemeral_low \
        || GDPVAL_JUDGE_PORT_BASE > ephemeral_high )) || {
        echo "GDPVAL_JUDGE_PORT_FAIL: selected judge window overlaps the ephemeral range" >&2
        return 64
    }
}
