#!/bin/bash
# Build one snapshotted entry file in a fresh copy of the image's read-only template (guard G4). Uploaded by the
# resources server at every verify; never taken from the image or the agent's workspace.
# Usage: ui_gen_verify.sh <stack> <snapshot file> <build dir> <timeout seconds>
# Exit 0 = built; 124 = build timed out; 90 = the verifier itself could not set up (not the model's fault);
# anything else = the build failed.
set -u
STACK=$1; SNAP=$2; DIR=$3; TIMEOUT=$4
TEMPLATE=/opt/ui-gen/template
export HOME="$DIR/.home" NEXT_TELEMETRY_DISABLED=1
mkdir -p "$HOME" || exit 90
case "$STACK" in
  ui-static)
    # The dataset's own static "build": a complete HTML document.
    grep -qi '</html>' "$SNAP" && { echo "ok"; exit 0; }
    echo "build: no </html> in the page" >&2; exit 1 ;;
  ui-next|ui-vite)
    for f in $(ls -A "$TEMPLATE"); do
      [ "$f" = node_modules ] || cp -a "$TEMPLATE/$f" "$DIR/" || exit 90
    done
    ln -s "$TEMPLATE/node_modules" "$DIR/node_modules" || exit 90
    chmod -R u+w "$DIR" || exit 90
    cd "$DIR" || exit 90
    if [ "$STACK" = ui-next ]; then
      cp "$SNAP" app/sample.tsx || exit 90
      exec timeout "$TIMEOUT" ./node_modules/.bin/next build
    else
      cp "$SNAP" src/App.tsx || exit 90
      exec timeout "$TIMEOUT" ./node_modules/.bin/vite build
    fi ;;
  *) echo "unknown stack $STACK" >&2; exit 90 ;;
esac
