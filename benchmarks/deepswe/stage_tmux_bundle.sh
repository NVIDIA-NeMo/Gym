#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]
then
  echo "usage: $0 OUTPUT_TAR_GZ" >&2
  exit 2
fi

output=$1
script_dir=$(cd "$(dirname "$0")" && pwd)
tmux_bin=$(command -v tmux)
temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT

mkdir -p "$temp_dir/bin" "$temp_dir/lib" "$(dirname "$output")"
cp -L "$tmux_bin" "$temp_dir/bin/tmux-real"
cp "$script_dir/runtime/tmux-wrapper.sh" "$temp_dir/bin/tmux"
chmod 755 "$temp_dir/bin/tmux" "$temp_dir/bin/tmux-real"

for library in libutempter.so.0 libtinfo.so.6 libevent_core-2.1.so.7
do
  path=$(ldd "$tmux_bin" | awk -v library="$library" '$1 == library {print $3}')
  if [[ -z "$path" || ! -f "$path" ]]
  then
    echo "could not resolve $library for $tmux_bin" >&2
    exit 1
  fi
  cp -L "$path" "$temp_dir/lib/$library"
done

tar -C "$temp_dir" -czf "$output" bin lib
echo "staged $(tmux -V) bundle at $output"
