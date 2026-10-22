"""Shared startup contract for the remote GDPVal OpenHands harness."""
import asyncio
import logging
import random
from posixpath import basename
from shlex import quote

DEFAULT_RUNTIME_DIR = '/mnt/s3-data/data/harness/gdpval-openhands/linux-amd64/v2'
DEFAULT_PYTHON = '/opt/gdpval-openhands/venv/bin/python'
# install.sh unpacks here and writes .installed-sha256 last.
INSTALL_PREFIX = '/opt/gdpval-openhands'
LOCAL_ARCHIVE_DIR = '/tmp/gdpval-runtime-archive'

log = logging.getLogger(__name__)


def install_command(script_path: str, archive_path: str, *, timeout_s: int = 900,
                    prefix: str = INSTALL_PREFIX, local_dir: str = LOCAL_ARCHIVE_DIR) -> str:
    """One install attempt, safe to repeat in the same sandbox.

    - A directory under ``prefix`` without .installed-sha256 is a killed attempt's leftovers;
      install.sh refuses it ("does not match this runtime"), so it is removed first.
    - The archive and its .sha256 sidecar are copied to local disk, and install.sh checks and
      unpacks the copy, so the S3 mount is read once instead of twice.
    - The attempt is killed 30 s before the caller's exec deadline, so no stale install is still
      writing when the next attempt starts.
    - stderr is merged into stdout, so failures carry the installer's own message.
    """
    if not script_path or not archive_path:
        raise ValueError('Both staged runtime installer and archive paths are required')
    if not script_path.startswith('/') or not archive_path.startswith('/'):
        raise ValueError('Staged runtime paths must be absolute sandbox paths')
    local_archive = f'{local_dir}/{basename(archive_path)}'
    attempt = ' && '.join([
        f'if [ -e {quote(prefix)} ] && [ ! -f {quote(prefix + "/.installed-sha256")} ]; then '
        f'echo "removing partial install at {prefix}"; rm -rf {quote(prefix)}; fi',
        f'rm -rf {quote(local_dir)}',
        f'mkdir -p {quote(local_dir)}',
        f'cp {quote(archive_path)} {quote(archive_path + ".sha256")} {quote(local_dir)}/',
        f'bash {quote(script_path)} --archive {quote(local_archive)}',
    ])
    budget_s = max(timeout_s - 30, 60)
    return (f'timeout -k 10 {budget_s} bash -c {quote(attempt)} 2>&1; rc=$?; '
            f'rm -rf {quote(local_dir)}; exit $rc')


def _output(result) -> str:
    return '\n'.join(part for part in (result.stdout, result.stderr) if part).strip()


async def install_runtime(sandbox, *, script_path, archive_path, timeout_s, attempts=4, backoff_s=(15, 30, 60)):
    """Install the harness in the sandbox, retrying failed attempts in place.

    Installs fail transiently when many sandboxes read the archive from the S3 mount at once.
    Without a retry, each failure aborts the rollout's whole group. Backoffs get up to 50% jitter
    so a burst of failed sandboxes does not retry in lockstep. The final error carries the last
    attempt's output and a listing of the archive, install directory and disk.
    """
    command = install_command(script_path, archive_path, timeout_s=timeout_s)
    last = None
    for i in range(attempts):
        result = await sandbox.exec(command, timeout_s=timeout_s)
        if not result.return_code:
            if i:
                log.warning('OpenHands runtime installed after %d retries', i)
            return
        last = result
        log.warning('OpenHands runtime installation attempt %d/%d failed (exit %s): %s',
                    i + 1, attempts, result.return_code, _output(result)[-600:])
        if i + 1 < attempts:
            delay = backoff_s[min(i, len(backoff_s) - 1)]
            await asyncio.sleep(delay * (1 + random.random() / 2))
    diag = ''
    try:
        probe = await sandbox.exec(
            f'ls -la {quote(archive_path)} {quote(script_path)}; ls -la {quote(INSTALL_PREFIX)} 2>&1 | head -5; '
            f'df -h /tmp / 2>/dev/null | tail -3; mount | grep -m1 s3 || echo no-s3-mount', timeout_s=60)
        diag = _output(probe)
    except Exception as exc:  # diagnostics must not mask the real error
        diag = f'(diagnostics failed: {exc!r})'
    raise RuntimeError(
        f'OpenHands runtime installation failed after {attempts} attempts (exit {last.return_code}): '
        f'{_output(last)[-1500:]} | diag: {diag[:800]}'
    )
