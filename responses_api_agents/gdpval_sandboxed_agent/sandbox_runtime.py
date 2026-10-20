"""Shared startup contract for the remote GDPVal OpenHands harness."""
from shlex import quote

DEFAULT_RUNTIME_DIR = '/mnt/s3-data/data/harness/gdpval-openhands/linux-amd64/v2'
DEFAULT_PYTHON = '/opt/gdpval-openhands/venv/bin/python'


def install_command(script_path: str, archive_path: str) -> str:
    if not script_path or not archive_path:
        raise ValueError('Both staged runtime installer and archive paths are required')
    if not script_path.startswith('/') or not archive_path.startswith('/'):
        raise ValueError('Staged runtime paths must be absolute sandbox paths')
    return f'bash {quote(script_path)} --archive {quote(archive_path)}'


async def install_runtime(sandbox, *, script_path, archive_path, timeout_s, attempts=4, backoff_s=(15, 30, 60)):
    """Install the harness in the sandbox, retrying transient failures.

    2026-10-10..12: the install (bash install.sh --archive runtime.tar.gz, both on the sandbox's
    S3 mount) failed with a bare "exit status 2" for 1,000-3,000 sandboxes per 4 h training copy
    while several jobs started sandboxes at once. Each failure used to abort the whole 16-rollout
    group (a 500 on /run, re-dispatched by NeMo RL, three strikes and the prompt is dropped). The
    failure is transient per sandbox, so retry in place; on the final failure include a listing of
    the archive and the sandbox's disk so the cause is visible in the agent log.
    """
    import asyncio, logging
    log = logging.getLogger(__name__)
    last = None
    for i in range(attempts):
        result = await sandbox.exec(install_command(script_path, archive_path), timeout_s=timeout_s)
        if not result.return_code:
            if i:
                log.warning('OpenHands runtime installed after %d retries', i)
            return
        last = result
        log.warning('OpenHands runtime installation attempt %d/%d failed (exit %s): %s',
                    i + 1, attempts, result.return_code, (result.stderr or result.stdout or '')[:300])
        if i + 1 < attempts:
            await asyncio.sleep(backoff_s[min(i, len(backoff_s) - 1)])
    diag = ''
    try:
        probe = await sandbox.exec(
            f'ls -la {quote(archive_path)} {quote(script_path)}; df -h /tmp / 2>/dev/null | tail -3; '
            f'mount | grep -m1 s3 || echo no-s3-mount', timeout_s=60)
        diag = (probe.stdout or '') + (probe.stderr or '')
    except Exception as exc:  # diagnostics must not mask the real error
        diag = f'(diagnostics failed: {exc!r})'
    raise RuntimeError(
        f'OpenHands runtime installation failed after {attempts} attempts (exit {last.return_code}): '
        f'{last.stderr or last.stdout} | diag: {diag[:800]}'
    )
