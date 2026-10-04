#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Start TimeWarp's Wiki, News and Shop sites for the given UI versions on the ports that
# configs/timewarp.yaml expects: Wiki 510V, News 520V, Shop 530V (V is the version, 1-6).
#
# Usage: start_sites.sh /path/to/timewarp [VERSION ...]    (default: all six versions)
#
# Run it inside TimeWarp's own Python environment after TimeWarp's setup.sh has downloaded the
# site data. Each site runs in the background; logs go to $TIMEWARP_LOG_DIR (default
# ./timewarp-site-logs). Stop them with: pkill -f 'wiki_app.py|news_app.py|web_agent_site.app'
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 /path/to/timewarp [VERSION ...]" >&2
  exit 2
fi
timewarp_dir="$(cd "$1" && pwd)"
shift
versions=("$@")
if [ ${#versions[@]} -eq 0 ]; then
  versions=(1 2 3 4 5 6)
fi

log_dir="${TIMEWARP_LOG_DIR:-$PWD/timewarp-site-logs}"
mkdir -p "$log_dir"

for v in "${versions[@]}"; do
  case "$v" in
    [1-6]) ;;
    *) echo "UI version must be 1-6, got '$v'" >&2; exit 2 ;;
  esac
  # Same entry points and flags as TimeWarp's scripts/environment/run_all_env.sh, with fixed ports.
  (cd "$timewarp_dir/env/wiki" && nohup python wiki_app.py "-$v" "--port=510$v" > "$log_dir/wiki$v.log" 2>&1 &)
  (cd "$timewarp_dir/env/news" && nohup python news_app.py "-$v" "--port=520$v" > "$log_dir/news$v.log" 2>&1 &)
  (cd "$timewarp_dir/env/webshop" && nohup python -m web_agent_site.app "$v" "--port=530$v" --log --attrs > "$log_dir/webshop$v.log" 2>&1 &)
  echo "UI version $v: wiki http://localhost:510$v  news http://localhost:520$v  shop http://localhost:530$v/abc"
done
echo "Logs: $log_dir (the Wiki and News sites take a minute to load their indexes)"
