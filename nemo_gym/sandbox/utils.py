# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sandbox utility helpers."""

# Sandboxes can expose the host's CPU count despite a smaller CFS quota. Keep
# this POSIX-shell preamble silent and fail-open because images vary widely.
_CPU_PIN_PREAMBLE = r"""
_ng_cpu_quota=
_ng_cpu_period=
_ng_cpu_list=
if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r _ng_cpu_quota _ng_cpu_period < /sys/fs/cgroup/cpu.max || :
else
    for _ng_cpu_dir in /sys/fs/cgroup/cpu /sys/fs/cgroup/cpu,cpuacct; do
        if [ -r "$_ng_cpu_dir/cpu.cfs_quota_us" ]; then
            _ng_cpu_quota=$(cat "$_ng_cpu_dir/cpu.cfs_quota_us" 2>/dev/null || :)
            _ng_cpu_period=$(cat "$_ng_cpu_dir/cpu.cfs_period_us" 2>/dev/null || :)
            [ "x$_ng_cpu_quota" = "x-1" ] && _ng_cpu_quota=max
            break
        fi
    done
fi
if [ -r /proc/self/status ] && command -v awk >/dev/null 2>&1; then
    _ng_cpu_list=$(awk '$1 == "Cpus_allowed_list:" {print $2; exit}' /proc/self/status 2>/dev/null || :)
fi
if [ -z "$_ng_cpu_list" ] && [ -r /sys/fs/cgroup/cpuset.cpus.effective ]; then
    _ng_cpu_list=$(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || :)
elif [ -z "$_ng_cpu_list" ] && [ -r /sys/fs/cgroup/cpuset/cpuset.effective_cpus ]; then
    _ng_cpu_list=$(cat /sys/fs/cgroup/cpuset/cpuset.effective_cpus 2>/dev/null || :)
fi
[ -n "$_ng_cpu_list" ] || _ng_cpu_list="0-$(( $(nproc 2>/dev/null || echo 1) - 1 ))"
if [ -n "$_ng_cpu_quota" ] && [ "$_ng_cpu_quota" != "max" ] \
    && [ "${_ng_cpu_period:-0}" -gt 0 ] 2>/dev/null; then
    _ng_cpu_count=$(( (_ng_cpu_quota + _ng_cpu_period - 1) / _ng_cpu_period ))
    [ "$_ng_cpu_count" -ge 1 ] || _ng_cpu_count=1
    if command -v awk >/dev/null 2>&1; then
        _ng_cpu_available=$(printf %s "$_ng_cpu_list" | awk -F, '
            {for (i=1;i<=NF;i++) {
                split($i,r,"-"); lo=r[1]+0; hi=(r[2]==""?lo:r[2]+0)
                total += hi-lo+1
            }}
            END {print total+0}' 2>/dev/null || :)
        if [ "${_ng_cpu_available:-0}" -gt 0 ] 2>/dev/null \
            && [ "$_ng_cpu_count" -gt "$_ng_cpu_available" ]; then
            _ng_cpu_count=$_ng_cpu_available
        fi
    fi
    export OMP_NUM_THREADS=$_ng_cpu_count OPENBLAS_NUM_THREADS=$_ng_cpu_count \
        MKL_NUM_THREADS=$_ng_cpu_count NUMEXPR_NUM_THREADS=$_ng_cpu_count \
        VECLIB_MAXIMUM_THREADS=$_ng_cpu_count UV_THREADPOOL_SIZE=$_ng_cpu_count \
        MAKEFLAGS=-j$_ng_cpu_count
    if command -v taskset >/dev/null 2>&1 && command -v awk >/dev/null 2>&1; then
        _ng_cpu_seed=$$
        if [ -r /dev/urandom ] && command -v od >/dev/null 2>&1 && command -v tr >/dev/null 2>&1; then
            _ng_cpu_random=$(od -An -N2 -tu2 /dev/urandom 2>/dev/null | tr -dc 0-9 2>/dev/null || :)
            case $_ng_cpu_random in
                ''|*[!0-9]*) ;;
                *) _ng_cpu_seed=$(( _ng_cpu_random + $$ )) ;;
            esac
        fi
        _ng_cpu_selection=$(printf %s "$_ng_cpu_list" | awk \
            -v n="$_ng_cpu_count" -v seed="$_ng_cpu_seed" -F, '
            {for (i=1;i<=NF;i++) {
                split($i,r,"-"); lo=r[1]+0; hi=(r[2]==""?lo:r[2]+0)
                for (cpu=lo;cpu<=hi;cpu++) allowed[count++]=cpu
            }}
            END {
                if (!count) exit
                if (n>count) n=count
                srand(seed%2147483647); offset=int(rand()*count)
                for (i=0;i<n;i++) selected=selected (i?",":"") allowed[(offset+i)%count]
                print selected
            }')
        if [ -n "$_ng_cpu_selection" ]; then
            taskset -pc "$_ng_cpu_selection" $$ >/dev/null 2>&1 || :
        fi
    fi
fi
""".strip()


def wrap_command_with_cpu_pin(command: str) -> str:
    """Size thread pools to the cgroup quota and best-effort pin ``command``."""
    return f"{_CPU_PIN_PREAMBLE}\n{command}"


def rewrite_image(image: str | None, rewrites: list[dict[str, str]]) -> str | None:
    """Apply ordered image-prefix rewrites used by sandbox configs."""
    if image is None:
        return None
    for rewrite in rewrites:
        from_prefix = rewrite["from"]
        to_prefix = rewrite["to"]
        if image.startswith(from_prefix):
            return to_prefix + image[len(from_prefix) :]
    return image
