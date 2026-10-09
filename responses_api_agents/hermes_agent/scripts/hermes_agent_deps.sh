#!/bin/bash
# Install hermes_agent deps into $DEPS_DIR (mounted read-only at /agent_deps_mount).
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PORTABLE_PYTHON_SH:-$SCRIPT_DIR/_portable_python.sh}"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

# Pin must match hermes_agent/app.py's AIAgent API; override only for experiments.
HERMES_REQ="$NEMO_GYM_ROOT/responses_api_agents/hermes_agent/requirements.txt"
HERMES_SPEC="${HERMES_SPEC:-$(sed -n 's/^hermes-agent @ //p' "$HERMES_REQ")}"
: "${HERMES_SPEC:?could not read hermes-agent pin from $HERMES_REQ}"

install_portable_python
install_nemo_gym_deps

echo "Installing hermes-agent ($HERMES_SPEC)"
"$DEPS_DIR/bin/python3" -m pip install --force-reinstall --no-deps "$HERMES_SPEC"
"$DEPS_DIR/bin/python3" -m pip install "$HERMES_SPEC"

# ``python -c`` normally prepends the caller's working directory to sys.path.
# NeMo RL has its own top-level ``tools`` package, which can shadow Hermes'
# ``tools.registry`` when Gym is launched from the RL checkout. Safe-path mode
# makes this health check resolve imports only from the portable runtime. A
# freshly installed package on shared NFS can also be briefly invisible to a
# new process, so retry the import before declaring the runtime unusable.
hermes_health_ok=0
for attempt in {1..12}; do
    if "$DEPS_DIR/bin/python3" -P -c "import model_tools; from run_agent import AIAgent; print('hermes-agent OK')"; then
        hermes_health_ok=1
        break
    fi
    echo "Hermes import health check failed (attempt $attempt/12); retrying in 5s" >&2
    sleep 5
done
if [ "$hermes_health_ok" -ne 1 ]; then
    echo "Hermes import health check failed after 12 attempts" >&2
    exit 1
fi

echo "hermes_agent deps ready at $DEPS_DIR"
