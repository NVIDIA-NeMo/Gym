# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select sufficient Apptainer tmpfs capacity without editing shared configuration.

Run in the owned GDP controller. If necessary, copy its actual active config to
<run-root>/private/apptainer.conf and change only sessiondir max size. For an
unprivileged installation or an effective-root controller, the returned override
uses Apptainer's documented APPTAINER_CONFIG_FILE setting.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


_SIZE = re.compile(r"^(?P<prefix>[ \t]*sessiondir max size[ \t]*=[ \t]*)(?P<size>\d+)(?P<tail>[ \t]*(?:#.*)?)$", re.M)


def prepare_apptainer_config(
    *, run_root: Path, minimum_mib: int = 8192, binary: str = "apptainer"
) -> dict[str, object]:
    """Return the active or private config path; refuse unsupported or conflicting copies."""
    if minimum_mib < 8192:
        raise ValueError("GDP runtime requires a sessiondir capacity of at least 8192MiB")
    run_root = run_root.resolve(strict=True)
    owners = {os.geteuid()}
    if os.environ.get("SLURM_JOB_UID", "").isdigit():
        owners.add(int(os.environ["SLURM_JOB_UID"]))
    if not run_root.is_dir() or run_root.stat().st_uid not in owners:
        raise ValueError("run_root must be an existing directory owned by this controller/job user")
    executable = shutil.which(binary)
    if executable is None:
        raise FileNotFoundError(f"Apptainer executable not found: {binary}")
    result = subprocess.run([executable, "buildcfg"], text=True, capture_output=True, timeout=15, check=True)
    build = dict(re.findall(r"^([A-Z0-9_]+)=(.*)$", result.stdout, re.M))
    active = os.environ.get("APPTAINER_CONFIG_FILE") or build.get("APPTAINER_CONF_FILE", "").strip('"')
    if not active:
        raise ValueError("apptainer buildcfg did not identify APPTAINER_CONF_FILE")
    source = Path(active).resolve(strict=True)
    original = source.read_text()
    matches = list(_SIZE.finditer(original))
    if len(matches) > 1:
        raise ValueError("Active Apptainer config has duplicate sessiondir max size directives")
    current_mib = int(matches[0]["size"]) if matches else 64
    output = source
    override_required = bool(os.environ.get("APPTAINER_CONFIG_FILE"))
    created = False
    if override_required or current_mib < minimum_mib:
        suid = build.get("APPTAINER_SUID_INSTALL", "").strip('"').lower()
        if os.geteuid() != 0 and suid not in {"0", "false", "no"}:
            raise PermissionError(
                "Private apptainer.conf requires effective root or a verified non-setuid installation"
            )
    if current_mib < minimum_mib:
        private = run_root / "private"
        if private.is_symlink():
            raise ValueError("Private configuration directory must not be a symlink")
        private.mkdir(mode=0o700, exist_ok=True)
        if private.stat().st_uid not in owners:
            raise ValueError("Private configuration directory is not owned by this controller/job user")
        output = private / "apptainer.conf"
        updated = (
            _SIZE.sub(lambda match: match["prefix"] + str(minimum_mib) + match["tail"], original)
            if matches
            else original + ("" if original.endswith("\n") else "\n") + f"sessiondir max size = {minimum_mib}\n"
        )
        if output.exists() or output.is_symlink():
            if output.is_symlink() or output.stat().st_uid not in owners or output.read_text() != updated:
                raise FileExistsError(
                    "Private Apptainer config already exists with different contents; inspect it first"
                )
        else:
            fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(updated)
            created = True
        override_required = True
        current_mib = minimum_mib
    return {
        "config_path": str(output),
        "source_config_path": str(source),
        "config_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "sessiondir_max_mib": current_mib,
        "override_required": override_required,
        "created_private_copy": created,
        "environment_variable": "APPTAINER_CONFIG_FILE" if override_required else None,
        "shared_configuration_modified": False,
    }


def main() -> None:
    """Print only nonsecret JSON; the caller applies the returned environment override."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--minimum-mib", type=int, default=8192)
    parser.add_argument("--apptainer", default="apptainer")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_apptainer_config(run_root=args.run_root, minimum_mib=args.minimum_mib, binary=args.apptainer)
        )
    )


if __name__ == "__main__":
    main()
