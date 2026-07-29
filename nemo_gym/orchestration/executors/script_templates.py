import re
import shlex

_HEALTH_WAIT_MULTI = """\
# Wait for {name} (try multiple health endpoints)
echo "Waiting for {name} at {url}..."
{name_upper}_READY=0
for _i in $(seq 1 {max_attempts}); do
    if curl -sf "{url}/health" > /dev/null 2>&1 || curl -sf "{url}/openapi.json" > /dev/null 2>&1; then
        echo "  {name} ready."
        {name_upper}_READY=1
        break
    fi
    if [ -n "${{{name_upper}_PID:-}}" ] && ! kill -0 ${name_upper}_PID 2>/dev/null; then
        echo "  {name} died during startup."
        exit 1
    fi
    sleep 5
done
if [ ${name_upper}_READY -eq 0 ]; then
    echo "ERROR: {name} did not become healthy after {max_attempts} attempts."
    exit 1
fi
"""


def bash_var(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "_", name.upper())


def render_health_check(name: str, port: int, timeout: int) -> str:
    return _HEALTH_WAIT_MULTI.format(
        name=name,
        name_upper=bash_var(name),
        url=f"http://localhost:{port}",
        max_attempts=timeout // 5,
    )


def render_gym_cmd(benchmark_name: str, run_args: list[str]) -> str:
    """Render a GYM_CMD bash array with each argument on its own line."""
    args = ["gym eval run", f"--benchmark {shlex.quote(benchmark_name)}", *run_args]
    return "GYM_CMD=(\n    " + "\n    ".join(args) + "\n)"


def render_gym_install(repo: str, ref: str) -> str:
    """Render a bash -c wrapper that installs uv + gym before exec-ing GYM_CMD."""
    install_cmd = shlex.quote(f"git+{repo}@{ref}")
    return (
        "bash -c '\n"
        "    curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        '    source "$HOME/.local/bin/env"\n'
        f"    uv pip install --system {install_cmd}\n"
        "    exec \"$@\"\n"
        "' -- \"${GYM_CMD[@]}\""
    )
