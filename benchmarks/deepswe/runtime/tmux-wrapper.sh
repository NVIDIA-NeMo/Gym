#!/bin/sh
export LD_LIBRARY_PATH="/opt/deep-swe-tmux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /opt/deep-swe-tmux/bin/tmux-real "$@"
