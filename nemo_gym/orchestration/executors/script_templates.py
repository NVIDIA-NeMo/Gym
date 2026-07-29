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


def render_gym_cmd(subcommand: str, var_name: str, args: list[str]) -> str:
    """Render a bash array with each argument on its own line."""
    entries = [f"gym {subcommand}", *args]
    return f"{var_name}=(\n    " + "\n    ".join(entries) + "\n)"


def render_driver_entrypoint(
    repo: str | None,
    ref: str | None,
    prepare_cmd: str | None,
) -> str:
    """Render the srun entrypoint for the driver step.

    When either gym_install or prepare is needed, wraps everything in a single
    bash -c so prepare and run happen in the same srun step and container.
    """
    preamble: list[str] = []

    if repo and ref:
        install_cmd = shlex.quote(f"git+{repo}@{ref}")
        preamble += [
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            'source "$HOME/.local/bin/env"',
            f"uv pip install --system {install_cmd}",
        ]

    if prepare_cmd:
        preamble.append(prepare_cmd)

    if not preamble:
        return '"${GYM_CMD[@]}"'

    preamble.append('exec "$@"')
    body = "\n    ".join(preamble)
    return f"bash -c '\n    {body}\n' -- \"${{GYM_CMD[@]}}\""
