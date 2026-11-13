"""Inspect Gym's distribution and require its installed CLI to report its version."""

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlsplit


def installed_source(distribution, official_image=False):
    """Use installation metadata and the checkout's actual remote, not clone syntax."""
    direct = json.loads(distribution.read_text("direct_url.json") or "null")
    if direct is None:
        # Ordinary index installs have no direct_url.json. This establishes package
        # identity, not a cryptographic proof of which index served the files.
        return {"route": "pypi", "source": "nemo-gym distribution"}
    url = direct.get("url", "")
    source_dir = None
    if urlsplit(url).scheme == "file" and "dir_info" in direct:
        source_dir = Path(unquote(urlsplit(url).path))
        if official_image:
            return {"route": "container", "source_dir": str(source_dir)}
        remote = subprocess.run(
            ["git", "-C", str(source_dir), "remote", "get-url", "origin"], capture_output=True, text=True, timeout=10
        )
        url = remote.stdout.strip() if remote.returncode == 0 else ""
    if re.fullmatch(r"(?:https://github\.com/|git@github\.com:)NVIDIA-NeMo/Gym(?:\.git)?/?", url, re.I):
        return {"route": "git", "source": url, "source_dir": str(source_dir) if source_dir else None}
    if urlsplit(url).hostname in {"files.pythonhosted.org", "pypi.org"}:
        return {"route": "pypi", "source": url}
    return {"route": "unknown", "source": url, "source_dir": str(source_dir) if source_dir else None}


def inspect_installation(expected_version, official_image=False):
    """Check package/source metadata and files, then run the installed gym --version.

    RECORD associates the gym file with the distribution. Check its recorded hash
    when available so replacing the installed script with a version-printing stub
    does not pass. This is a targeted consistency check, not a supply-chain audit.
    """
    try:
        distribution = metadata.distribution("nemo-gym")
    except metadata.PackageNotFoundError:
        return {"passed": False, "reason": "nemo-gym is not installed in this Python environment"}
    if re.sub(r"[-_.]+", "-", distribution.metadata.get("Name", "")).lower() != "nemo-gym":
        return {"passed": False, "reason": "Installed distribution identity is not nemo-gym"}
    if not distribution.version or distribution.version.lower() != expected_version.lower():
        return {"passed": False, "reason": "Installed package version differs from the trajectory version"}
    entry = next(
        (item for item in distribution.entry_points if item.group == "console_scripts" and item.name == "gym"), None
    )
    if entry is None:
        return {"passed": False, "reason": "Installed distribution has no gym console entrypoint"}
    files = distribution.files or []
    scripts = [item for item in files if Path(str(item)).name == "gym"]
    script = next((item for item in scripts if distribution.locate_file(item).is_file()), None)
    if script is None:
        return {"passed": False, "reason": "The installed gym console script is missing"}
    cli = distribution.locate_file(script).resolve()
    if script.hash:
        digest = hashlib.new(script.hash.mode, cli.read_bytes()).digest()
        if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != script.hash.value:
            return {"passed": False, "reason": "The installed gym script differs from its RECORD hash"}
    source = installed_source(distribution, official_image)
    module = entry.module.replace(".", "/")
    roots = [Path(distribution.locate_file(""))]
    if source.get("source_dir"):
        roots.extend([Path(source["source_dir"]), Path(source["source_dir"]) / "src"])
    if not any((root / (module + ".py")).is_file() or (root / module / "__init__.py").is_file() for root in roots):
        return {"passed": False, "reason": "The console entrypoint module is missing"}
    if source["route"] == "unknown" and not official_image:
        return {"passed": False, "reason": "Installed source is not an accepted PyPI or official Git source"}
    if official_image:
        source = {"route": "container", "source": "official Gym image (digest checked by verifier)"}
    command = [str(cli), "--version"]
    env = {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"passed": False, "reason": "Installed Gym CLI could not run: " + str(error)}
    output = result.stdout + result.stderr
    version = re.search(r"(?im)^\s*NeMo\s+Gym\s+v?(\d[^\s]*)\s*$", output)
    cli_check = {"command": command, "exit_code": result.returncode, "output": output}
    if result.returncode != 0 or version is None or version[1].lower() != expected_version.lower():
        return {
            "passed": False,
            "reason": "Installed gym --version failed or reported a different version",
            "cli_check": cli_check,
        }
    return {
        "passed": True,
        **source,
        "package_version": distribution.version,
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "distribution_location": str(distribution.locate_file("")),
        "cli": str(cli),
        "entrypoint": entry.value,
        "version_output": output,
        "cli_check": cli_check,
    }


if __name__ == "__main__":
    print(json.dumps(inspect_installation(sys.argv[1], official_image="--official-image" in sys.argv[2:])))
